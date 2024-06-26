# General
import os
import gc
from os.path import join as opj
import datetime
from pathlib import Path
import torch
import tempfile
import yaml
#from .model.video_ldm import VideoLDM
from typing import List, Optional
#from .model.callbacks import SaveConfigCallback
from PIL.Image import Image, fromarray

from einops import rearrange, repeat

import folder_paths
comfy_path = os.path.dirname(folder_paths.__file__)
result_fol = f'{comfy_path}/output'

import sys
sys.path.insert(0,f'{comfy_path}/custom_nodes/ComfyUI_StreamingT2V/thirdparty')
sys.path.insert(0,f'{comfy_path}/custom_nodes/ComfyUI_StreamingT2V')

#from modelscope.pipelines import pipeline
#from modelscope.outputs import OutputKeys
import imageio
import pathlib
import numpy as np

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
        # Utilities
        from .model_init import init_streamingt2v_model,init_modelscope,init_animatediff,init_sdxl,init_v2v_model

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
        # Utilities
        from .model_init import init_streamingt2v_model,init_modelscope,init_animatediff,init_sdxl,init_v2v_model
        
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
        # Utilities
        from .model_init import init_streamingt2v_model,init_modelscope,init_animatediff,init_sdxl,init_v2v_model
        
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
                "upscale_width": ("INT", {"default": 1280}),
                "upscale_height": ("INT", {"default": 720}),
                "upscale_pad": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "StreamingT2V"

    def run(self,StreamingT2VModel,prompt,negative_prompt,num_frames,num_steps,image_guidance,seed,chunk,overlap,upscale_width,upscale_height,upscale_pad):
        # Utilities
        from .model_func import ms_short_gen,ad_short_gen,svd_short_gen,stream_long_gen,video2video_randomized,video2video

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

        cfg_v2v = {'downscale': 1, 'upscale_size': (upscale_width,upscale_height), 'model_id': 'damo/Video-to-Video', 'pad': upscale_pad}
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
                "upscale_width": ("INT", {"default": 1280}),
                "upscale_height": ("INT", {"default": 720}),
                "upscale_pad": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "StreamingT2V"

    def run(self,StreamingT2VModelSVD,image,prompt,negative_prompt,num_frames,num_steps,image_guidance,seed,chunk,overlap,upscale_width,upscale_height,upscale_pad):
        # Utilities
        from .model_func import ms_short_gen,ad_short_gen,svd_short_gen,stream_long_gen,video2video_randomized,video2video

        image = 255.0 * image[0].cpu().numpy()
        image = fromarray(np.clip(image, 0, 255).astype(np.uint8))
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

        cfg_v2v = {'downscale': 1, 'upscale_size': (upscale_width,upscale_height), 'model_id': 'damo/Video-to-Video', 'pad': upscale_pad}
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
        # Utilities
        from .model_init import init_modelscope

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
        # Utilities
        from .model_init import init_animatediff

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
        # Utilities
        from .model_init import init_svd

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
        # Utilities
        from .model_init import init_v2v_model

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
        # Utilities
        from .model_init import init_streamingt2v_model

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
        pipeline.enable_model_cpu_offload()
        pipeline.enable_vae_slicing()
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
        # Utilities
        from .model_func import ms_short_gen

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
        # Utilities
        from .model_func import ad_short_gen

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
        # Utilities
        from .model_func import svd_short_gen

        image = 255.0 * image[0].cpu().numpy()
        image = fromarray(np.clip(image, 0, 255).astype(np.uint8))
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
                "negative_prompt":("STRING",{"default":"worst quality, normal quality, low quality, low res, blurry, text,watermark, logo, banner, extra digits, cropped,jpeg artifacts, signature, username, error,sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "StreamingT2V"

    def run(self,VidXTendPipeline,short_video,prompt,num_frames,num_steps,image_guidance,seed,negative_prompt):
        images = []
        for image in short_video:
            image = 255.0 * image.cpu().numpy()
            image = fromarray(np.clip(image, 0, 255).astype(np.uint8))
            images.append(image)
        #images=short_video.permute(0,3,1,2)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        added_frames = len(images)
        while added_frames < num_frames:
            result = VidXTendPipeline(
                prompt=prompt,
                #num_frames=num_frames,
                num_inference_steps=num_steps,
                negative_prompt=negative_prompt,
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

class StreamingT2VRunLongStepVidXTendPipelineCustomRef:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "VidXTendPipeline": ("VidXTendPipeline",),
                "short_video":("IMAGE",),
                "prompt":("STRING",{"default":"A cat running on the street"}),
                "ref_frames":("IMAGE",),
                "num_frames": ("INT", {"default": 24}),
                "num_steps": ("INT", {"default": 50}),
                "image_guidance": ("FLOAT", {"default": 9.0}),
                "seed": ("INT", {"default": 33}),
                "negative_prompt":("STRING",{"default":"worst quality, normal quality, low quality, low res, blurry, text,watermark, logo, banner, extra digits, cropped,jpeg artifacts, signature, username, error,sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "StreamingT2V"

    def run(self,VidXTendPipeline,short_video,prompt,ref_frames,num_frames,num_steps,image_guidance,seed,negative_prompt):
        images = []
        for image in short_video:
            image = 255.0 * image.cpu().numpy()
            image = fromarray(np.clip(image, 0, 255).astype(np.uint8))
            images.append(image)
        
        refimages = []
        for image in ref_frames:
            image = 255.0 * image.cpu().numpy()
            image = fromarray(np.clip(image, 0, 255).astype(np.uint8))
            refimages.append(image)

        #images=short_video.permute(0,3,1,2)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        added_frames = len(images)
        while added_frames < num_frames:
            result = VidXTendPipeline(
                prompt=prompt,
                #num_frames=num_frames,
                num_inference_steps=num_steps,
                negative_prompt=negative_prompt,
                image=images[-8:], # Use final 8 frames of video
                input_frames_conditioning=refimages, # Use first frame of video
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

class StreamingT2VRunLongStepVidXTendPipelineCustomRefOutExtendOnly:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "VidXTendPipeline": ("VidXTendPipeline",),
                "short_video":("IMAGE",),
                "prompt":("STRING",{"default":"A cat running on the street"}),
                "ref_frames":("IMAGE",),
                "num_frames": ("INT", {"default": 24}),
                "num_steps": ("INT", {"default": 50}),
                "image_guidance": ("FLOAT", {"default": 9.0}),
                "seed": ("INT", {"default": 33}),
                "negative_prompt":("STRING",{"default":"worst quality, normal quality, low quality, low res, blurry, text,watermark, logo, banner, extra digits, cropped,jpeg artifacts, signature, username, error,sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "StreamingT2V"

    def run(self,VidXTendPipeline,short_video,prompt,ref_frames,num_frames,num_steps,image_guidance,seed,negative_prompt):
        images = []
        for image in short_video:
            image = 255.0 * image.cpu().numpy()
            image = fromarray(np.clip(image, 0, 255).astype(np.uint8))
            images.append(image)
        
        refimages = []
        for image in ref_frames:
            image = 255.0 * image.cpu().numpy()
            image = fromarray(np.clip(image, 0, 255).astype(np.uint8))
            refimages.append(image)

        #images=short_video.permute(0,3,1,2)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        added_frames = len(images)
        while added_frames < num_frames:
            result = VidXTendPipeline(
                prompt=prompt,
                #num_frames=num_frames,
                num_inference_steps=num_steps,
                negative_prompt=negative_prompt,
                image=images, # Use final 8 frames of video
                input_frames_conditioning=refimages, # Use first frame of video
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

class StreamingT2VRunLongStepVidXTendPipelinePromptTravel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "VidXTendPipeline": ("VidXTendPipeline",),
                "short_video":("IMAGE",),
                "prompt":("STRING",{"default":"", "multiline": True}),
                "ref_frames":("IMAGE",),
                "num_frames": ("INT", {"default": 24}),
                "num_steps": ("INT", {"default": 50}),
                "image_guidance": ("FLOAT", {"default": 9.0}),
                "seed": ("INT", {"default": 33}),
                "negative_prompt":("STRING",{"default":"worst quality, normal quality, low quality, low res, blurry, text,watermark, logo, banner, extra digits, cropped,jpeg artifacts, signature, username, error,sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "StreamingT2V"

    def run(self,VidXTendPipeline,short_video,prompt,ref_frames,num_frames,num_steps,image_guidance,seed,negative_prompt):
        import json
        promptstr="{"+prompt+"}"
        prompts={}
        input_frames_conditionings={}
        promptjson=json.loads(promptstr)
 
        images = []
        for image in short_video:
            image = 255.0 * image.cpu().numpy()
            image = fromarray(np.clip(image, 0, 255).astype(np.uint8))
            images.append(image)
        
        input_frames = []
        for image in ref_frames:
            image = 255.0 * image.cpu().numpy()
            image = fromarray(np.clip(image, 0, 255).astype(np.uint8))
            input_frames.append(image)

        ind=len(images)
        preprompt=list(promptjson.values())[0]
        preimage=input_frames[0]
        prompts[str(ind)]=preprompt
        while ind < num_frames:
            if str(ind) in list(promptjson.keys()):
                prompts[str(ind)]=promptjson[str(ind)]
            else:
                prompts[str(ind)]=preprompt
            preprompt=prompts[str(ind)]

            if ind/8<len(input_frames):
                input_frames_conditionings[str(ind)]=input_frames[int(ind/8)]
            else:
                input_frames_conditionings[str(ind)]=preimage
            preimage=input_frames_conditionings[str(ind)]
            ind+=8
        #images=short_video.permute(0,3,1,2)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        added_frames = len(images)
        while added_frames < num_frames:
            prompt=prompts[str(added_frames)]
            result = VidXTendPipeline(
                prompt=prompt,
                #num_frames=num_frames,
                num_inference_steps=num_steps,
                negative_prompt=negative_prompt,
                image=images[-8:], # Use final 8 frames of video
                input_frames_conditioning=[input_frames_conditionings[str(added_frames)]], # Use first frame of video
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
        # Utilities
        from .model_func import stream_long_gen

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
                "upscale_width": ("INT", {"default": 1280}),
                "upscale_height": ("INT", {"default": 720}),
                "upscale_pad": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "StreamingT2V"

    def run(self,msxl_model,low_video_path,prompt,num_frames,chunk,overlap,upscale_width,upscale_height,upscale_pad):
        # Utilities
        from .model_func import video2video_randomized,video2video
        
        result_fol = folder_paths.get_output_directory()

        cfg_v2v = {'downscale': 1, 'upscale_size': (upscale_width,upscale_height), 'model_id': 'damo/Video-to-Video', 'pad': upscale_pad}
        ret=f''
        if num_frames > 80:
            ret=video2video_randomized(prompt, low_video_path, result_fol, cfg_v2v, msxl_model, chunk_size=chunk, overlap_size=overlap)
        else:
            ret=video2video(prompt, low_video_path, result_fol, cfg_v2v, msxl_model)
        return (ret,)

class VHS_FILENAMES_STRING_StreamingT2V:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "filenames": ("VHS_FILENAMES",),
                    }
                }

    RETURN_TYPES = ("STRING",)
    CATEGORY = "StreamingT2V"
    FUNCTION = "run"

    def run(self, filenames):
        return (filenames[1][-1],)

class PromptTravelIndex:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt":("STRING",{"default":""}),
                "index": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("STRING",)
    CATEGORY = "StreamingT2V"
    FUNCTION = "run"

    def run(self, prompt, index):
        import json
        promptstr="{"+prompt+"}"
        promptjson=json.loads(promptstr)

        keys=list(promptjson.keys())
        ret=''
        if index<=int(keys[len(keys)-1]):
            for ind in range(len(keys)):
                if index>=int(keys[ind]):
                    ret=list(promptjson.values())[ind]

        else:
            ret=list(promptjson.values())[len(keys)-1]
        return (ret,)

def get_allowed_dirs():
    import json
    dir = os.path.abspath(os.path.join(__file__, ".."))
    file = os.path.join(dir, "text_file_dirs.json")
    with open(file, "r") as f:
        return json.loads(f.read())


def get_valid_dirs():
    return get_allowed_dirs().keys()

def get_dir_from_name(name):
    dirs = get_allowed_dirs()
    if name not in dirs:
        raise KeyError(name + " dir not found")

    path = dirs[name]
    path = path.replace("$input", folder_paths.get_input_directory())
    path = path.replace("$output", folder_paths.get_output_directory())
    path = path.replace("$temp", folder_paths.get_temp_directory())
    return path


def is_child_dir(parent_path, child_path):
    parent_path = os.path.abspath(parent_path)
    child_path = os.path.abspath(child_path)
    return os.path.commonpath([parent_path]) == os.path.commonpath([parent_path, child_path])


def get_real_path(dir):
    dir = dir.replace("/**/", "/")
    dir = os.path.abspath(dir)
    dir = os.path.split(dir)[0]
    return dir

def get_file(root_dir, file):
    if file == "[none]" or not file or not file.strip():
        raise ValueError("No file")

    root_dir = get_dir_from_name(root_dir)
    root_dir = get_real_path(root_dir)
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    full_path = os.path.join(root_dir, file)

    #if not is_child_dir(root_dir, full_path):
    #    raise ReferenceError()

    return full_path

class TextFileNode:
    RETURN_TYPES = ("STRING","BOOLEAN",)
    CATEGORY = "utils"

    def load_text(self, **kwargs):
        self.file = get_file(kwargs["root_dir"], kwargs["file"])
        if not os.path.exists(self.file):
            return ("",False,)
        with open(self.file, "r") as f:
            return (f.read(),True, )


class LoadText_StreamingT2V(TextFileNode):
    @classmethod
    def IS_CHANGED(self, **kwargs):
        return os.path.getmtime(self.file)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "root_dir": (list(get_valid_dirs()), {"default":"output"}),
                "file": ("STRING", {"default": "dragtest_1.txt"}),
            },
        }

    FUNCTION = "load_text"

class SaveText_StreamingT2V(TextFileNode):
    @classmethod
    def IS_CHANGED(self, **kwargs):
        return float("nan")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "root_dir": (list(get_valid_dirs()), {"default":"output"}),
                "file": ("STRING", {"default": "dragtest_1.txt"}),
                "text": ("STRING", {"forceInput": True, "multiline": True})
            },
        }

    FUNCTION = "write_text"

    def write_text(self, **kwargs):
        self.file = get_file(kwargs["root_dir"], kwargs["file"])
        with open(self.file, "w") as f:
            f.write(kwargs["text"])

        return super().load_text(**kwargs)

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
    "StreamingT2VRunLongStepVidXTendPipelineCustomRef":StreamingT2VRunLongStepVidXTendPipelineCustomRef,
    "StreamingT2VRunLongStepVidXTendPipelineCustomRefOutExtendOnly":StreamingT2VRunLongStepVidXTendPipelineCustomRefOutExtendOnly,
    "StreamingT2VRunLongStepVidXTendPipelinePromptTravel":StreamingT2VRunLongStepVidXTendPipelinePromptTravel,
    "VHS_FILENAMES_STRING_StreamingT2V":VHS_FILENAMES_STRING_StreamingT2V,
    "PromptTravelIndex":PromptTravelIndex,
    "LoadText_StreamingT2V":LoadText_StreamingT2V,
    "SaveText_StreamingT2V":SaveText_StreamingT2V,
}

#import logging
#logging_level = logging.INFO

#logging.basicConfig(format="%(message)s", level=logging_level)
#logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))