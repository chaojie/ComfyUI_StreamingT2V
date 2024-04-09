from __future__ import annotations
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re
import gc
import json
import torch
import inspect
import PIL.Image
import numpy as np

from typing import Any, Callable, Dict, List, Optional, Union
from contextlib import ExitStack
from einops import rearrange

from transformers import CLIPTextModel, CLIPTextConfig, CLIPTokenizer
from transformers.modeling_utils import no_init_weights

from diffusers.image_processor import VaeImageProcessor, PipelineImageInput
from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL
from diffusers.schedulers import KarrasDiffusionSchedulers, DDIMScheduler
from diffusers.utils import (
    PIL_INTERPOLATION,
    is_accelerate_available,
    is_xformers_available,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.text_to_video_synthesis import TextToVideoSDPipelineOutput

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

from huggingface_hub import hf_hub_download

from vidxtend.models import (
    UNet3DConditionModel,
    ControlNetModel,
    NoiseGenerator,
    MaskGenerator,
    FrozenOpenCLIPImageEmbedder,
    ImageEmbeddingContextResampler,
)

from vidxtend.utils import logger, iterate_state_dict

def tensor2vid(
    video: torch.Tensor,
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],
    output_type="list"
) -> List[np.ndarray]:
    # This code is copied from https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/pipelines/multi_modal/text_to_video_synthesis_pipeline.py#L78
    # reshape to ncfhw
    mean = torch.tensor(mean, device=video.device).reshape(1, -1, 1, 1, 1)
    std = torch.tensor(std, device=video.device).reshape(1, -1, 1, 1, 1)
    # unnormalize back to [0,1]
    video = video.mul_(std).add_(mean)
    video.clamp_(0, 1)
    # prepare the final outputs
    i, c, f, h, w = video.shape
    images = video.permute(2, 3, 0, 4, 1).reshape(
        f, h, i * w, c
    )  # 1st (frames, h, batch_size, w, c) 2nd (frames, h, batch_size * w, c)
    if output_type == "list":
        # prepare a list of indvidual (consecutive frames)
        images = images.unbind(dim=0)
        images = [
            (image.cpu().numpy() * 255).astype("uint8") for image in images
        ]  # f h w c
    elif output_type == "pt":
        pass
    return images

class VidXTendPipelineOutput(TextToVideoSDPipelineOutput):
    pass

class VidXTendPipeline(DiffusionPipeline, TextualInversionLoaderMixin):
    r"""
    Pipeline for text-to-video generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Same as Stable Diffusion 2.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet3DConditionModel`]): Conditional U-Net architecture to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """
    _exclude_from_cpu_offload = []
    _optional_components = ["resampler"]
    model_cpu_offload_seq = "text_encoder->unet->vae"

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        controlnet: ControlNetModel,
        scheduler: KarrasDiffusionSchedulers,
        noise_generator: Optional[NoiseGenerator]=None,
        resampler: Optional[ImageEmbeddingContextResampler]=None,
        num_frames: int=16,
        num_frames_conditioning: int=8,
        temporal_self_attention_only_on_conditioning: bool=False,
        temporal_self_attention_mask_included_itself: bool=False,
        spatial_attend_on_condition_frames: bool=False,
        temp_attend_on_uncond_include_past: bool=False,
        temp_attend_on_neighborhood_of_condition_frames: bool=False,
        image_encoder_version: str="laion2b_s32b_b79k",
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            resampler=resampler,
            noise_generator=noise_generator,
        )
        self.register_to_config(
            num_frames=num_frames,
            num_frames_conditioning=num_frames_conditioning,
            temporal_self_attention_only_on_conditioning=temporal_self_attention_only_on_conditioning,
            temporal_self_attention_mask_included_itself=temporal_self_attention_mask_included_itself,
            spatial_attend_on_condition_frames=spatial_attend_on_condition_frames,
            temp_attend_on_uncond_include_past=temp_attend_on_uncond_include_past,
            temp_attend_on_neighborhood_of_condition_frames=temp_attend_on_neighborhood_of_condition_frames,
            image_encoder_version=image_encoder_version,
        )
        self.mask_generator = MaskGenerator(
            num_frames=num_frames,
            num_frames_conditioning=num_frames_conditioning,
            temporal_self_attention_only_on_conditioning=temporal_self_attention_only_on_conditioning,
            temporal_self_attention_mask_included_itself=temporal_self_attention_mask_included_itself,
            temp_attend_on_uncond_include_past=temp_attend_on_uncond_include_past
        )
        self.image_encoder = FrozenOpenCLIPImageEmbedder(version=image_encoder_version)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def set_use_memory_efficient_attention_xformers(
        self, 
        valid: bool=True,
        attention_op: Optional[Callable]=None
    ) -> None:
        """
        Set the memory efficient attention for the model.
        """
        from vidxtend.models.processor import set_use_memory_efficient_attention_xformers
        for model in [self.unet, self.controlnet]:
            set_use_memory_efficient_attention_xformers(
                model,
                num_frames_conditioning=self.config.num_frames_conditioning,
                num_frames=self.config.num_frames,
                temporal_self_attention_only_on_conditioning=self.config.temporal_self_attention_only_on_conditioning,
                temporal_self_attention_mask_included_itself=self.config.temporal_self_attention_mask_included_itself,
                spatial_attend_on_condition_frames=self.config.spatial_attend_on_condition_frames,
                temp_attend_on_neighborhood_of_condition_frames=self.config.temp_attend_on_neighborhood_of_condition_frames,
                temp_attend_on_uncond_include_past=self.config.temp_attend_on_uncond_include_past,
                valid=valid,
                attention_op=attention_op,
            )

    @classmethod
    def from_single_file(
        cls,
        file_path_or_repository: str,
        filename: str="vidxtend.safetensors",
        config_filename: str="config.json",
        variant: Optional[str]=None,
        subfolder: Optional[str]=None,
        device: Optional[Union[str, torch.device]]=None,
        torch_dtype: Optional[torch.dtype]=None,
        cache_dir: Optional[str]=None,
    ) -> TextToVideoSDPipeline:
        """
        Load a streaming text-to-video pipeline from a file or repository.
        """
        if variant is not None:
            filename, ext = os.path.splitext(filename)
            filename = f"{filename}.{variant}{ext}"

        if device is None:
            device = "cpu"
        else:
            device = str(device)

        if os.path.isdir(file_path_or_repository):
            model_dir = file_path_or_repository
            if subfolder:
                model_dir = os.path.join(model_dir, subfolder)
            file_path = os.path.join(model_dir, filename)
            config_path = os.path.join(model_dir, config_filename)
        elif os.path.isfile(file_path_or_repository):
            file_path = file_path_or_repository
            if os.path.isfile(config_filename):
                config_path = config_filename
            else:
                config_path = os.path.join(os.path.dirname(file_path), config_filename)
                if not os.path.exists(config_path) and subfolder:
                    config_path = os.path.join(os.path.dirname(file_path), subfolder, config_filename)
        elif re.search(r"^[a-zA-Z0-9_-]+\/[a-zA-Z0-9_-]+$", file_path_or_repository):
            file_path = hf_hub_download(
                file_path_or_repository,
                filename,
                subfolder=subfolder,
                cache_dir=cache_dir,
            )
            try:
                config_path = hf_hub_download(
                    file_path_or_repository,
                    config_filename,
                    subfolder=subfolder,
                    cache_dir=cache_dir,
                )
            except:
                config_path = hf_hub_download(
                    file_path_or_repository,
                    config_filename,
                    cache_dir=cache_dir,
                )
        else:
            raise FileNotFoundError(f"File {file_path_or_repository} is not a repository that can be downloaded or a file that can be loaded.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"File {config_path} not found.")

        with open(config_path, "r") as f:
            vidxtend_config = json.load(f)

        # Create the scheduler
        scheduler = DDIMScheduler.from_config(vidxtend_config["scheduler"])

        # Create tokenizer (downloaded)
        tokenizer = CLIPTokenizer.from_pretrained(
            vidxtend_config["tokenizer"]["model"],
            subfolder=vidxtend_config["tokenizer"].get("subfolder", None),
            cache_dir=cache_dir,
        )

        # Create the base models
        context = ExitStack()
        if is_accelerate_available():
            context.enter_context(no_init_weights())
            context.enter_context(init_empty_weights())

        with context:
            # UNet
            unet = UNet3DConditionModel.from_config(vidxtend_config["unet"])

            # VAE
            vae = AutoencoderKL.from_config(vidxtend_config["vae"])

            # Text encoder
            text_encoder = CLIPTextModel(CLIPTextConfig(**vidxtend_config["text_encoder"]))

            # Resampler
            resampler = ImageEmbeddingContextResampler.from_config(vidxtend_config["resampler"])

        # Load the weights
        state_dicts = {}
        for key, value in iterate_state_dict(file_path):
            try:
                module, _, key = key.partition(".")
                if is_accelerate_available():
                    if module == "unet":
                        set_module_tensor_to_device(unet, key, device=device, value=value)
                    elif module == "vae":
                        set_module_tensor_to_device(vae, key, device=device, value=value)
                    elif module == "text_encoder":
                        set_module_tensor_to_device(text_encoder, key, device=device, value=value)
                    elif module == "resampler":
                        set_module_tensor_to_device(resampler, key, device=device, value=value)
                    elif module == "controlnet":
                        if "controlnet" not in state_dicts:
                            state_dicts["controlnet"] = {}
                        state_dicts["controlnet"][key] = value
                    else:
                        raise ValueError(f"Unknown module: {module}")
                else:
                    if module not in state_dicts:
                        state_dicts[module] = {}
                    state_dicts[module][key] = value
            except (AttributeError, KeyError, ValueError) as ex:
                logger.warning(f"Skipping module {module} key {key} due to {type(ex)}: {ex}")

        if not is_accelerate_available():
            try:
                unet.load_state_dict(state_dicts["unet"], strict=False)
                vae.load_state_dict(state_dicts["vae"], strict=False)
                text_encoder.load_state_dict(state_dicts["text_encoder"], strict=False)
                resampler.load_state_dict(state_dicts["resampler"], strict=False)
            except KeyError as ex:
                raise RuntimeError(f"File did not provide a state dict for {ex}")

        # Create controlnet
        controlnet = ControlNetModel.from_unet(
            unet,
            **vidxtend_config["controlnet"]
        )

        # Load controlnet state dict
        controlnet.load_state_dict(state_dicts["controlnet"], strict=False)

        # Cleanup
        del state_dicts
        gc.collect()

        # Create the pipeline
        pipeline = cls(
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            scheduler=scheduler,
            controlnet=controlnet,
            tokenizer=tokenizer,
            resampler=resampler,
            num_frames=vidxtend_config.get("num_frames", 16),
            num_frames_conditioning=vidxtend_config.get("num_frames_conditioning", 8),
            temporal_self_attention_only_on_conditioning=vidxtend_config.get("temporal_self_attention_only_on_conditioning", False),
            temporal_self_attention_mask_included_itself=vidxtend_config.get("temporal_self_attention_mask_included_itself", False),
            spatial_attend_on_condition_frames=vidxtend_config.get("spatial_attend_on_condition_frames", False),
            temp_attend_on_uncond_include_past=vidxtend_config.get("temp_attend_on_uncond_include_past", False),
            temp_attend_on_neighborhood_of_condition_frames=vidxtend_config.get("temp_attend_on_neighborhood_of_condition_frames", False),
            image_encoder_version=vidxtend_config.get("image_encoder_version", "laion2b_s32b_b79k"),
        )

        if torch_dtype is not None:
            pipeline.to(torch_dtype)

        return pipeline

    def to(self, *args, **kwargs):
        """
        Move the model to the specified device.
        We manually move the image encoder to the same device as the model, as it's
        created as a frozen module and not added to the normal diffusers pipeline modules.
        """
        super().to(*args, **kwargs)
        kwargs.pop("silence_dtype_warnings", None)
        self.image_encoder.to(*args, **kwargs)

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance,
        cfg_text_image=False,
    ):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            if isinstance(image[0], PIL.Image.Image):
                images = []

                for image_ in image:
                    image_ = image_.convert("RGB")
                    image_ = image_.resize(
                        (width, height), resample=PIL_INTERPOLATION["lanczos"]
                    )
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)

                image = images

                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        image_vq_enc = (
            self.vae.encode(
                rearrange(image, "B F C W H -> (B F) C W H")
            ).latent_dist.sample()
            * self.vae.config.scaling_factor
        )
        image_vq_enc = rearrange(
            image_vq_enc, "(B F) C W H -> B F C W H", B=image_batch_size
        )
        if do_classifier_free_guidance:
            if cfg_text_image:
                image = torch.cat([torch.zeros_like(image), image], dim=0)
            else:
                image = torch.cat([image] * 2)
            # image_vq_enc = torch.cat([image_vq_enc] * 2)

        return image, image_vq_enc

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_tiling
    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding.

        When this option is enabled, the VAE will split the input tensor into tiles to compute decoding and encoding in
        several steps. This is useful to save a large amount of memory and to allow the processing of larger images.
        """
        self.vae.enable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_tiling
    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        img_cond: Optional[torch.FloatTensor] = None,
        img_cond_unc: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )
        max_length = prompt_embeds.shape[1]
        if img_cond is not None:
            if img_cond.ndim == 2:
                img_cond = img_cond.unsqueeze(1)
            prompt_embeds = torch.cat([prompt_embeds, img_cond], dim=1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            # max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=self.text_encoder.dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            if img_cond_unc is not None:
                if img_cond_unc.ndim == 2:
                    img_cond_unc = img_cond_unc.unsqueeze(1)
                negative_prompt_embeds = torch.cat(
                    [negative_prompt_embeds, img_cond_unc], dim=1
                )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        batch_size, channels, num_frames, height, width = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, height, width
        )

        image = self.vae.decode(latents).sample
        video = (
            image[None, :]
            .reshape(
                (
                    batch_size,
                    num_frames,
                    -1,
                )
                + image.shape[2:]
            )
            .permute(0, 2, 1, 3, 4)
        )
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        video = video.float()
        return video

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if (callback_steps is None) or (
            callback_steps is not None
            and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        num_frames,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        content=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            num_frames,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if self.noise_generator is not None:
            latents = self.noise_generator.sample_noise(
                shape=shape, generator=generator, device=device, dtype=dtype, content=content
            )
        elif latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def reset_noise_generator_state(self):
        if self.noise_generator is not None and hasattr(
            self.noise_generator, "reset_noise"
        ):
            self.noise_generator.reset_noise_generator_state()

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        # the image input for the controlnet branch
        image: Optional[torch.FloatTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 16,
        num_inference_steps: int = 50,
        guidance_scale: float = 9.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        no_text_condition_control: bool = False,
        weight_control_sample: float = 1.0,
        use_controlnet_mask: bool = False,
        skip_controlnet_branch: bool = False,
        input_frames_conditioning = None,
        cfg_text_image: bool = False,
        use_of: bool = False,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the video generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated video.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated video.
            num_frames (`int`, *optional*, defaults to 16):
                The number of video frames that are generated. Defaults to 16 frames which at 8 frames per seconds
                amounts to 2 seconds of video.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality videos at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate videos that are closely linked to the text `prompt`,
                usually at the expense of lower video quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`. Latents should be of shape
                `(batch_size, num_channel, num_frames, height, width)`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generate video. Choose between `torch.FloatTensor` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.TextToVideoSDPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.TextToVideoSDPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.TextToVideoSDPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated frames.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_images_per_prompt = 1
        controlnet_mask = None
        device = self._execution_device

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        if image is not None:
            image = self.image_processor.preprocess(image).unsqueeze(0).to(
                device=device, dtype=self.controlnet.dtype
            ) # 1 f c h w

        if input_frames_conditioning is not None:
            input_frames_conditioning = self.image_processor.preprocess(input_frames_conditioning).unsqueeze(0).to(
                device=device, dtype=self.controlnet.dtype
            ) # 1 f c h w

        if self.image_encoder is not None and self.resampler is not None and image is not None:
            self.image_encoder.to(device)
            self.resampler.to(device)
            bsz = image.shape[0]
            image_for_conditioning = rearrange(input_frames_conditioning, "B F C W H -> (B F) C W H")
            image_enc = self.image_encoder(image_for_conditioning)
            img_cond = self.resampler(image_enc, batch_size=bsz)
            image_enc_unc = self.image_encoder(torch.zeros_like(image_for_conditioning))
            img_cond_unc = self.resampler(image_enc_unc, batch_size=bsz)
            self.image_encoder.to("cpu")
            self.resampler.to("cpu")
        else:
            img_cond = None
            img_cond_unc = None

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            img_cond=img_cond,
            img_cond_unc=img_cond_unc,
        )

        skip_conditioning = image is None or skip_controlnet_branch

        if not skip_conditioning:
            num_condition_frames = image.shape[1]
            image, image_vq_enc = self.prepare_image(
                image=image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=self.controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                cfg_text_image=cfg_text_image,
            )

            if len(image.shape) == 5:
                image = rearrange(image, "B F C H W -> (B F) C H W")

            if use_controlnet_mask:
                # num_condition_frames = all possible frames, e.g. 16
                assert num_condition_frames == num_frames
                image = rearrange(
                    image, "(B F) C H W -> B F C H W", F=num_condition_frames
                )
                # image = torch.cat([image, image], dim=1)
                controlnet_mask = torch.zeros(
                    (image.shape[0], num_frames), device=image.device, dtype=image.dtype
                )
                # TODO HARDCODED number of frames!
                controlnet_mask[:, :8] = 1.0
                image = rearrange(image, "B F C H W -> (B F) C H W")

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        of_channels = 2 if use_of else 0
        num_channels_ctrl = self.unet.config.in_channels
        num_channels_latents = num_channels_ctrl + of_channels
        if not skip_conditioning:
            image_vq_enc = rearrange(image_vq_enc, "B F C H W -> B C F H W")

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            #content=image_vq_enc if not skip_conditioning else None,
        ).to(device=device, dtype=prompt_embeds.dtype)

        if self.unet.concat:
            image_latents = (
                self.vae.encode(
                    rearrange(image, "B F C W H -> (B F) C W H")
                ).latent_dist.sample()
                * self.vae.config.scaling_factor
            )
            image_latents = rearrange(
                image_latents, "(B F) C W H -> B C F W H", B=latents.shape[0]
            )
            image_shape = image_latents.shape
            image_shape = [ax_dim for ax_dim in image_shape]
            image_shape[2] = 16 - image_shape[2]
            image_latents = torch.cat(
                [
                    image_latents,
                    torch.zeros(
                        image_shape,
                        dtype=image_latents.dtype,
                        device=image_latents.device,
                    ),
                ],
                dim=2,
            )
            controlnet_mask = torch.zeros(
                image_latents.shape,
                device=image_latents.device,
                dtype=image_latents.dtype,
            )
            controlnet_mask[:, :, :8] = 1.0
            image_latents = image_latents * controlnet_mask
            # torch.cat([latents, image_latents, controlnet_mask[:, :1]], dim=1)
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        if self.mask_generator is not None:
            attention_mask = self.mask_generator.get_mask(
                device=latents.device,
                use_half=self.controlnet.dtype is torch.float16 or self.controlnet.dtype is torch.bfloat16
            )
        else:
            attention_mask = None

        if not skip_conditioning:
            self.controlnet.to(device)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                if self.unet.concat:
                    latent_model_input = torch.cat(
                        [
                            latent_model_input,
                            image_latents.repeat(2, 1, 1, 1, 1),
                            controlnet_mask[:, :1].repeat(2, 1, 1, 1, 1),
                        ],
                        dim=1,
                    )

                if not skip_conditioning:
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        latent_model_input[:, :num_channels_ctrl],
                        t,
                        encoder_hidden_states=(
                            prompt_embeds
                            if (not no_text_condition_control)
                            else torch.stack([prompt_embeds[0], prompt_embeds[0]])
                        ),
                        controlnet_cond=image,
                        attention_mask=attention_mask,
                        vq_gan=self.vae,
                        weight_control_sample=weight_control_sample,
                        return_dict=False,
                        controlnet_mask=controlnet_mask,
                    )
                else:
                    down_block_res_samples = None
                    mid_block_res_sample = None

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    down_block_additional_residuals=(
                        [
                            sample.to(dtype=latent_model_input.dtype)
                            for sample in down_block_res_samples
                        ]
                        if down_block_res_samples is not None
                        else None
                    ),
                    mid_block_additional_residual=(
                        mid_block_res_sample.to(dtype=latent_model_input.dtype)
                        if mid_block_res_sample is not None
                        else None
                    ),
                    fps=None,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # reshape latents
                bsz, channel, frames, width, height = latents.shape
                latents = latents.permute(0, 2, 1, 3, 4).reshape(
                    bsz * frames, channel, width, height
                )
                noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(
                    bsz * frames, channel, width, height
                )

                # compute the previous noisy sample x_t -> x_t-1
                scheduler_step = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                )
                latents = scheduler_step.prev_sample

                # reshape latents back
                latents = (
                    latents[None, :]
                    .reshape(bsz, frames, channel, width, height)
                    .permute(0, 2, 1, 3, 4)
                )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not skip_conditioning:
            self.controlnet.to("cpu")

        latents_video = latents[:, :num_channels_ctrl]
        if of_channels > 0:
            latents_of = latents[:, num_channels_ctrl:]
            latents_of = rearrange(latents_of, "B C F W H -> (B F) C W H")

        video_tensor = self.decode_latents(latents_video)

        if output_type == "pt":
            video = video_tensor
        elif output_type == "pt_t2v":
            video = tensor2vid(video_tensor, output_type="pt")
            video = rearrange(video, "f h w c -> f c h w")
        elif output_type == "pil":
            video = tensor2vid(video_tensor, output_type="list")
            video = [
                PIL.Image.fromarray(frame)
                for frame in video
            ]
        elif output_type == "concat_image":
            image_video = image.unsqueeze(2)[0:1].repeat([1, 1, 24, 1, 1])
            video_tensor_concat = torch.concat([image_video, video_tensor], dim=4)
            video = tensor2vid(video_tensor_concat)
        else:
            video = tensor2vid(video_tensor)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.controlnet.to("cpu")
            self.final_offload_hook.offload()

        if not return_dict:
            if of_channels == 0:
                return video
            else:
                return video, latents_of

        return TextToVideoSDPipelineOutput(frames=video)
