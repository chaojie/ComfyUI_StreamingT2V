# General
import os
import gc
from os.path import join as opj
import datetime
from pathlib import Path
import torch
import tempfile
import yaml
from .model.video_ldm import VideoLDM
from typing import List, Optional
from .model.callbacks import SaveConfigCallback
from PIL.Image import Image, fromarray

from einops import rearrange, repeat

import folder_paths
comfy_path = os.path.dirname(folder_paths.__file__)
result_fol = f'{comfy_path}/output'

import sys
sys.path.insert(0,f'{comfy_path}/custom_nodes/ComfyUI_StreamingT2V/thirdparty')
sys.path.insert(0,f'{comfy_path}/custom_nodes/ComfyUI_StreamingT2V')

from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
import imageio
import pathlib
import numpy as np

# Utilities
from .inference_utils import *
from .model_init import *
from .model_func import *

class StreamingT2VLoaderModelscopeT2V:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"default": "streaming_t2v.ckpt"}),
                "device":(["cuda","cpu"],{"default":"cuda"}),
                "vram_not_enough":("BOOLEAN",{"default":True}),
            },
        }

    RETURN_TYPES = ("StreamingT2VModel",)
    FUNCTION = "run"
    CATEGORY = "StreamingT2V"

    def run(self,ckpt_name,device,vram_not_enough):
        sdxl_model=None
        base_model="ModelscopeT2V"
        result_fol = folder_paths.get_output_directory()
        ckpt_file_streaming_t2v = folder_paths.get_full_path("checkpoints", ckpt_name)
        cfg_v2v = {'downscale': 1, 'upscale_size': (1280, 720), 'model_id': 'damo/Video-to-Video', 'pad': True}
        stream_cli, stream_model = init_streamingt2v_model(Path(ckpt_file_streaming_t2v).absolute(), Path(result_fol).absolute(),vram_not_enough)
        if base_model == "ModelscopeT2V":
            model = init_modelscope(device)
        elif base_model == "AnimateDiff":
            model = init_animatediff(device)
        elif base_model == "SVD":
            model = init_svd(device)
            sdxl_model = init_sdxl(device)
        
        msxl_model = init_v2v_model(cfg_v2v,device)
        return ((model,sdxl_model,msxl_model,base_model,stream_cli, stream_model),)

class StreamingT2VLoaderAnimateDiff:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"default": "streaming_t2v.ckpt"}),
                "device":(["cuda","cpu"],{"default":"cuda"}),
                "vram_not_enough":("BOOLEAN",{"default":True}),
            },
        }

    RETURN_TYPES = ("StreamingT2VModel",)
    FUNCTION = "run"
    CATEGORY = "StreamingT2V"

    def run(self,ckpt_name,device,vram_not_enough):
        sdxl_model=None
        base_model="AnimateDiff"
        result_fol = folder_paths.get_output_directory()
        ckpt_file_streaming_t2v = folder_paths.get_full_path("checkpoints", ckpt_name)
        cfg_v2v = {'downscale': 1, 'upscale_size': (1280, 720), 'model_id': 'damo/Video-to-Video', 'pad': True}
        stream_cli, stream_model = init_streamingt2v_model(Path(ckpt_file_streaming_t2v).absolute(), Path(result_fol).absolute(),vram_not_enough)
        if base_model == "ModelscopeT2V":
            model = init_modelscope(device)
        elif base_model == "AnimateDiff":
            model = init_animatediff(device)
        elif base_model == "SVD":
            model = init_svd(device)
            sdxl_model = init_sdxl(device)
        
        msxl_model = init_v2v_model(cfg_v2v,device)
        return ((model,sdxl_model,msxl_model,base_model,stream_cli, stream_model),)

class StreamingT2VLoaderSVD:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"default": "streaming_t2v.ckpt"}),
                "device":(["cuda","cpu"],{"default":"cuda"}),
                "vram_not_enough":("BOOLEAN",{"default":True}),
            },
        }

    RETURN_TYPES = ("StreamingT2VModelSVD",)
    FUNCTION = "run"
    CATEGORY = "StreamingT2V"

    def run(self,ckpt_name,device,vram_not_enough):
        sdxl_model=None
        base_model="SVD"
        result_fol = folder_paths.get_output_directory()
        ckpt_file_streaming_t2v = folder_paths.get_full_path("checkpoints", ckpt_name)
        cfg_v2v = {'downscale': 1, 'upscale_size': (1280, 720), 'model_id': 'damo/Video-to-Video', 'pad': True}
        stream_cli, stream_model = init_streamingt2v_model(Path(ckpt_file_streaming_t2v).absolute(), Path(result_fol).absolute(),vram_not_enough)

        predevice=device
        if vram_not_enough:
            device='cpu'
        
        if base_model == "ModelscopeT2V":
            model = init_modelscope(device)
        elif base_model == "AnimateDiff":
            model = init_animatediff(device)
        elif base_model == "SVD":
            model = init_svd(device)
            sdxl_model = init_sdxl(device)

        device=predevice
        msxl_model = init_v2v_model(cfg_v2v,device)
        return ((model,sdxl_model,msxl_model,base_model,stream_cli, stream_model),)


class StreamingT2VRunT2V:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "StreamingT2VModel": ("StreamingT2VModel",),
                "prompt":("STRING",{"default":"A cat running on the street"}),
                "negative_prompt":("STRING",{"default":""}),
                "num_frames": ("INT", {"default": 24}),
                "num_steps": ("INT", {"default": 50}),
                "image_guidance": ("FLOAT", {"default": 9.0}),
                "seed": ("INT", {"default": 33}),
                "chunk": ("INT", {"default": 56}),
                "overlap": ("INT", {"default": 32}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "StreamingT2V"

    def run(self,StreamingT2VModel,prompt,negative_prompt,num_frames,num_steps,image_guidance,seed,chunk,overlap):
        result_fol = folder_paths.get_output_directory()
        model,sdxl_model,msxl_model,base_model,stream_cli, stream_model=StreamingT2VModel
        
        inference_generator = torch.Generator(device="cuda")

        now = datetime.datetime.now()
        name = prompt[:100].replace(" ", "_") + "_" + str(now.time()).replace(":", "_").replace(".", "_")

        inference_generator = torch.Generator(device="cuda")
        inference_generator.manual_seed(seed)
        
        if base_model == "ModelscopeT2V":
            short_video = ms_short_gen(prompt, model, inference_generator)
        elif base_model == "AnimateDiff":
            short_video = ad_short_gen(prompt, model, inference_generator)
        elif base_model == "SVD":
            short_video = svd_short_gen(image, prompt, model, sdxl_model, inference_generator)

        n_autoreg_gen = (num_frames-8)//8
        stream_long_gen(prompt, short_video, n_autoreg_gen, seed, num_steps, image_guidance, name, stream_cli, stream_model)

        cfg_v2v = {'downscale': 1, 'upscale_size': (1280, 720), 'model_id': 'damo/Video-to-Video', 'pad': True}
        ret=f'{result_fol}/{name}.mp4'
        if num_frames > 80:
            ret=video2video_randomized(prompt, opj(result_fol, name+".mp4"), result_fol, cfg_v2v, msxl_model, chunk_size=chunk, overlap_size=overlap)
        else:
            ret=video2video(prompt, opj(result_fol, name+".mp4"), result_fol, cfg_v2v, msxl_model)
        return (ret,)


class StreamingT2VRunI2V:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "StreamingT2VModelSVD": ("StreamingT2VModelSVD",),
                "image": ("IMAGE",),
                "prompt":("STRING",{"default":"A cat running on the street"}),
                "negative_prompt":("STRING",{"default":""}),
                "num_frames": ("INT", {"default": 24}),
                "num_steps": ("INT", {"default": 50}),
                "image_guidance": ("FLOAT", {"default": 9.0}),
                "seed": ("INT", {"default": 33}),
                "chunk": ("INT", {"default": 56}),
                "overlap": ("INT", {"default": 32}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "StreamingT2V"

    def run(self,StreamingT2VModelSVD,image,prompt,negative_prompt,num_frames,num_steps,image_guidance,seed,chunk,overlap):
        image = 255.0 * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        input_fol = folder_paths.get_input_directory()
        image_path=f'{input_fol}/i2v.png'
        image.save(image_path)
        image=image_path

        result_fol = folder_paths.get_output_directory()
        model,sdxl_model,msxl_model,base_model,stream_cli, stream_model=StreamingT2VModelSVD
        
        inference_generator = torch.Generator(device="cuda")

        now = datetime.datetime.now()
        name = prompt[:100].replace(" ", "_") + "_" + str(now.time()).replace(":", "_").replace(".", "_")

        inference_generator = torch.Generator(device="cuda")
        inference_generator.manual_seed(seed)
        
        if base_model == "ModelscopeT2V":
            short_video = ms_short_gen(prompt, model, inference_generator)
        elif base_model == "AnimateDiff":
            short_video = ad_short_gen(prompt, model, inference_generator)
        elif base_model == "SVD":
            short_video = svd_short_gen(image, prompt, model, sdxl_model, inference_generator)

        n_autoreg_gen = (num_frames-8)//8
        stream_long_gen(prompt, short_video, n_autoreg_gen, seed, num_steps, image_guidance, name, stream_cli, stream_model)

        cfg_v2v = {'downscale': 1, 'upscale_size': (1280, 720), 'model_id': 'damo/Video-to-Video', 'pad': True}
        ret=f'{result_fol}/{name}.mp4'
        if num_frames > 80:
            ret=video2video_randomized(prompt, opj(result_fol, name+".mp4"), result_fol, cfg_v2v, msxl_model, chunk_size=chunk, overlap_size=overlap)
        else:
            ret=video2video(prompt, opj(result_fol, name+".mp4"), result_fol, cfg_v2v, msxl_model)
        return (ret,)


class StreamingT2VLoaderModelscopeModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device":(["cuda","cpu"],{"default":"cuda"}),
            },
        }

    RETURN_TYPES = ("T2VModel",)
    FUNCTION = "run"
    CATEGORY = "StreamingT2V"

    def run(self,device):
        model = init_modelscope(device)
        return (model,)

class StreamingT2VLoaderAnimateDiffModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device":(["cuda","cpu"],{"default":"cuda"}),
            },
        }

    RETURN_TYPES = ("T2VModel",)
    FUNCTION = "run"
    CATEGORY = "StreamingT2V"

    def run(self,device):
        model = init_animatediff(device)
        return (model,)

class StreamingT2VLoaderSVDModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device":(["cuda","cpu"],{"default":"cuda"}),
            },
        }

    RETURN_TYPES = ("I2VModel",)
    FUNCTION = "run"
    CATEGORY = "StreamingT2V"

    def run(self,device):
        model = init_svd(device)
        return (model,)

class StreamingT2VLoaderEnhanceModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device":(["cuda","cpu"],{"default":"cuda"}),
            },
        }

    RETURN_TYPES = ("msxl_model",)
    FUNCTION = "run"
    CATEGORY = "StreamingT2V"

    def run(self,device):
        cfg_v2v = {'downscale': 1, 'upscale_size': (1280, 720), 'model_id': 'damo/Video-to-Video', 'pad': True}
        msxl_model = init_v2v_model(cfg_v2v,device)
        return (msxl_model,)

class StreamingT2VLoaderStreamModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"default": "streaming_t2v.ckpt"}),
                "device":(["cuda","cpu"],{"default":"cuda"}),
            },
        }

    RETURN_TYPES = ("stream_cli", "stream_model",)
    FUNCTION = "run"
    CATEGORY = "StreamingT2V"

    def run(self,ckpt_name,device):
        vram_not_enough=False
        if device=="cpu":
            vram_not_enough=True
        result_fol = folder_paths.get_output_directory()
        ckpt_file_streaming_t2v = folder_paths.get_full_path("checkpoints", ckpt_name)
        cfg_v2v = {'downscale': 1, 'upscale_size': (1280, 720), 'model_id': 'damo/Video-to-Video', 'pad': True}
        stream_cli, stream_model = init_streamingt2v_model(Path(ckpt_file_streaming_t2v).absolute(), Path(result_fol).absolute(),vram_not_enough)
        
        return (stream_cli, stream_model,)


class StreamingT2VLoaderVidXTendModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device":(["cuda","cpu"],{"default":"cuda"}),
            },
        }

    RETURN_TYPES = ("VidXTendPipeline",)
    RETURN_NAMES = ("VidXTendPipeline",)
    FUNCTION = "run"
    CATEGORY = "StreamingT2V"

    def run(self,device):
        from vidxtend import VidXTendPipeline
        pipeline = VidXTendPipeline.from_single_file(
            "benjamin-paine/vidxtend",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipeline.set_use_memory_efficient_attention_xformers()
        pipeline.to(device, dtype=torch.float16)

        return (pipeline,)

class StreamingT2VRunShortStepModelscopeT2V:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("T2VModel",),
                "prompt":("STRING",{"default":"A cat running on the street"}),
                "seed": ("INT", {"default": 33}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("short_video",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "StreamingT2V"

    def run(self,model,prompt,seed):
        inference_generator = torch.Generator(device="cuda")

        inference_generator = torch.Generator(device="cuda")
        inference_generator.manual_seed(seed)
        
        short_video = ms_short_gen(prompt, model, inference_generator)
        short_video = short_video.permute(0,2,3,1)

        return (short_video,)

class StreamingT2VRunShortStepAnimateDiff:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("T2VModel",),
                "prompt":("STRING",{"default":"A cat running on the street"}),
                "seed": ("INT", {"default": 33}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("short_video",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "StreamingT2V"

    def run(self,model,prompt,seed):
        inference_generator = torch.Generator(device="cuda")

        inference_generator = torch.Generator(device="cuda")
        inference_generator.manual_seed(seed)
        
        short_video = ad_short_gen(prompt, model, inference_generator)
        print(f'{short_video.shape}')
        short_video = short_video.permute(0,2,3,1)

        return (short_video,)

class StreamingT2VRunShortStepSVD:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("I2VModel",),
                "image": ("IMAGE",),
                "prompt":("STRING",{"default":"A cat running on the street"}),
                "seed": ("INT", {"default": 33}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("short_video",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "StreamingT2V"

    def run(self,model,image,prompt,seed):
        image = 255.0 * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        input_fol = folder_paths.get_input_directory()
        image_path=f'{input_fol}/i2v.png'
        image.save(image_path)
        image=image_path

        inference_generator = torch.Generator(device="cuda")

        inference_generator = torch.Generator(device="cuda")
        inference_generator.manual_seed(seed)
        
        short_video = svd_short_gen(image, prompt, model, None, inference_generator)
        short_video = short_video.permute(0,2,3,1)

        return (short_video,)

class StreamingT2VRunLongStepVidXTendPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "VidXTendPipeline": ("VidXTendPipeline",),
                "short_video":("IMAGE",),
                "prompt":("STRING",{"default":"A cat running on the street"}),
                "num_frames": ("INT", {"default": 24}),
                "num_steps": ("INT", {"default": 50}),
                "image_guidance": ("FLOAT", {"default": 9.0}),
                "seed": ("INT", {"default": 33}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "StreamingT2V"

    def run(self,VidXTendPipeline,short_video,prompt,num_frames,num_steps,image_guidance,seed):
        images = []
        for image in short_video:
            image = 255.0 * image.cpu().numpy()
            image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
            images.append(image)
        #images=short_video.permute(0,3,1,2)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        added_frames = len(images)
        while added_frames < num_frames:
            result = VidXTendPipeline(
                prompt=prompt,
                #num_frames=num_frames,
                negative_prompt=None, # Optionally use negative prompt
                image=images[-8:], # Use final 8 frames of video
                input_frames_conditioning=images[:1], # Use first frame of video
                eta=1.0,
                guidance_scale=image_guidance,
                generator=generator,
                output_type="pil"
            ) # Remove the first 8 frames from the output as they were used as guide for final 8
            images.extend(result.frames[8:])
            added_frames += 8
            # Clear memory between iterations
            torch.cuda.empty_cache()
            gc.collect()
        images = [torch.unsqueeze(torch.tensor(np.array(image).astype(np.float32) / 255.0), 0) for image in images]
        return torch.cat(tuple(images[:num_frames]), dim=0).unsqueeze(0)

class StreamingT2VRunLongStep:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stream_cli": ("stream_cli",),
                "stream_model": ("stream_model",),
                "short_video":("IMAGE",),
                "prompt":("STRING",{"default":"A cat running on the street"}),
                "num_frames": ("INT", {"default": 24}),
                "num_steps": ("INT", {"default": 50}),
                "image_guidance": ("FLOAT", {"default": 9.0}),
                "seed": ("INT", {"default": 33}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("low_video_path",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "StreamingT2V"

    def run(self,stream_cli, stream_model,short_video,prompt,num_frames,num_steps,image_guidance,seed):
        short_video=short_video.permute(0,3,1,2)
        print(f'{short_video.shape}')

        result_fol = folder_paths.get_output_directory()
        now = datetime.datetime.now()
        name = prompt[:100].replace(" ", "_") + "_" + str(now.time()).replace(":", "_").replace(".", "_")
        
        n_autoreg_gen = (num_frames-8)//8
        stream_long_gen(prompt, short_video, n_autoreg_gen, seed, num_steps, image_guidance, name, stream_cli, stream_model)

        return (opj(result_fol, name+".mp4"),)

class StreamingT2VRunEnhanceStep:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "msxl_model": ("msxl_model",),
                "low_video_path":("STRING",{"default":""}),
                "prompt":("STRING",{"default":"A cat running on the street"}),
                "num_frames": ("INT", {"default": 24}),
                "chunk": ("INT", {"default": 56}),
                "overlap": ("INT", {"default": 32}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "StreamingT2V"

    def run(self,msxl_model,low_video_path,prompt,num_frames,chunk,overlap):
        result_fol = folder_paths.get_output_directory()

        cfg_v2v = {'downscale': 1, 'upscale_size': (1280, 720), 'model_id': 'damo/Video-to-Video', 'pad': True}
        ret=f''
        if num_frames > 80:
            ret=video2video_randomized(prompt, low_video_path, result_fol, cfg_v2v, msxl_model, chunk_size=chunk, overlap_size=overlap)
        else:
            ret=video2video(prompt, low_video_path, result_fol, cfg_v2v, msxl_model)
        return (ret,)


NODE_CLASS_MAPPINGS = {
    "StreamingT2VLoaderModelscopeT2V":StreamingT2VLoaderModelscopeT2V,
    "StreamingT2VLoaderAnimateDiff":StreamingT2VLoaderAnimateDiff,
    "StreamingT2VLoaderSVD":StreamingT2VLoaderSVD,
    "StreamingT2VRunT2V":StreamingT2VRunT2V,
    "StreamingT2VRunI2V":StreamingT2VRunI2V,
    "StreamingT2VLoaderModelscopeModel":StreamingT2VLoaderModelscopeModel,
    "StreamingT2VLoaderAnimateDiffModel":StreamingT2VLoaderAnimateDiffModel,
    "StreamingT2VLoaderSVDModel":StreamingT2VLoaderSVDModel,
    "StreamingT2VLoaderEnhanceModel":StreamingT2VLoaderEnhanceModel,
    "StreamingT2VLoaderStreamModel":StreamingT2VLoaderStreamModel,
    "StreamingT2VRunShortStepModelscopeT2V":StreamingT2VRunShortStepModelscopeT2V,
    "StreamingT2VRunShortStepAnimateDiff":StreamingT2VRunShortStepAnimateDiff,
    "StreamingT2VRunShortStepSVD":StreamingT2VRunShortStepSVD,
    "StreamingT2VRunLongStep":StreamingT2VRunLongStep,
    "StreamingT2VRunEnhanceStep":StreamingT2VRunEnhanceStep,
    "StreamingT2VLoaderVidXTendModel":StreamingT2VLoaderVidXTendModel,
    "StreamingT2VRunLongStepVidXTendPipeline":StreamingT2VRunLongStepVidXTendPipeline,
}

import logging
logging_level = logging.INFO

logging.basicConfig(format="%(message)s", level=logging_level)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))