from __future__ import annotations

import os
import gc
import sys
import math
import click
import random
import logging
import traceback

from PIL import Image

from typing import Optional

@click.command()
@click.argument("video", type=click.Path(exists=True, dir_okay=False))
@click.argument("prompt", type=str)
@click.option("--frame-rate", "-fps", type=int, default=None, help="Video FPS. Will default to the input FPS.", show_default=True)
@click.option("--seconds", "-s", type=float, default=1.0, help="The total number of seconds to add to the video. Multiply this number by frame rate to determine total number of new frames generated.", show_default=True)
@click.option("--negative-prompt", "-np", type=str, default=None, help="Negative prompt for the diffusion process.", show_default=True)
@click.option("--guidance-scale", "-cfg", type=float, default=7.5, help="Guidance scale for the diffusion process.", show_default=True)
@click.option("--num-inference-steps", "-ns", type=int, default=50, help="Number of diffusion steps.", show_default=True)
@click.option("--seed", "-r", type=int, default=None, help="Random seed.")
@click.option("--model", "-m", type=str, default="benjamin-paine/vidxtend", help="HuggingFace model name.")
@click.option("--no-half", "-nh", is_flag=True, default=False, help="Do not use half precision.", show_default=True)
@click.option("--no-offload", "-no", is_flag=True, default=False, help="Do not offload to the CPU to preserve GPU memory.", show_default=True)
@click.option("--no-slicing", "-ns", is_flag=True, default=False, help="Do not use VAE slicing.", show_default=True)
@click.option("--gpu-id", "-g", type=int, default=0, help="GPU ID to use.")
@click.option("--model-single-file", "-sf", is_flag=True, default=False, help="Download and use a single file instead of a directory.")
@click.option("--config-file", "-cf", type=str, default="config.json", help="Config file to use when using the model-single-file option. Accepts a path or a filename in the same directory as the single file. Will download from the repository passed in the model option if not provided.", show_default=True)
@click.option("--model-filename", "-mf", type=str, default="vidxtend.safetensors", help="The model file to download when using the model-single-file option.", show_default=True)
@click.option("--remote-subfolder", "-rs", type=str, default=None, help="Remote subfolder to download from when using the model-single-file option.")
@click.option("--cache-dir", "-cd", type=click.Path(exists=True, file_okay=False), help="Cache directory to download to. Default uses the huggingface cache.", default=None)
@click.option("--output", "-o", type=click.Path(exists=False, dir_okay=False), help="Output file.", default="output.mp4", show_default=True)
@click.option("--fit", "-f", type=click.Choice(["actual", "cover", "contain", "stretch"]), default="cover", help="Image fit mode.", show_default=True)
@click.option("--anchor", "-a", type=click.Choice(["top-left", "top-center", "top-right", "center-left", "center-center", "center-right", "bottom-left", "bottom-center", "bottom-right"]), default="top-left", help="Image anchor point.", show_default=True)
def main(
    video: str,
    prompt: str,
    frame_rate: Optional[int]=None,
    seconds: float=1.0,
    guidance_scale: float=7.5,
    num_inference_steps: int=50,
    negative_prompt: Optional[str]=None,
    seed: Optional[int]=None,
    model: str="benjamin-paine/vidxtend",
    no_half: bool=False,
    no_offload: bool=False,
    no_slicing: bool=False,
    gpu_id: int=0,
    model_single_file: bool=False,
    config_file: str="config.json",
    model_filename: str="vidxtend.safetensors",
    remote_subfolder: Optional[str]=None,
    cache_dir: Optional[str]=None,
    output: str="output.mp4",
    fit: str="cover",
    anchor: str="top-left",
) -> None:
    """
    Run VidXtend on a video file, concatenating the generated frames to the end of the video.
    """
    if os.path.exists(output):
        base, ext = os.path.splitext(os.path.basename(output))
        dirname = os.path.dirname(output)
        suffix = 1
        while os.path.exists(os.path.join(dirname, f"{base}-{suffix}{ext}")):
            suffix += 1
        new_output_filename = f"{base}-{suffix}{ext}"
        click.echo(f"Output file {output} already exists. Writing to {new_output_filename} instead.")
        output = os.path.join(dirname, new_output_filename)

    import torch
    from diffusers.utils.import_utils import is_xformers_available
    from vidxtend import VidXTendPipeline
    from vidxtend.utils import Video, fit_image, human_size

    device = (
        torch.device("cuda", index=gpu_id)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if no_half:
        variant = None
        torch_dtype = None
    else:
        variant = "fp16"
        torch_dtype = torch.float16

    if model_single_file:
        pipeline = VidXTendPipeline.from_single_file(
            model,
            filename=model_filename,
            config_filename=config_file,
            variant=variant,
            subfolder=remote_subfolder,
            cache_dir=cache_dir,
            device=device,
            torch_dtype=torch_dtype,
        )
    else:
        pipeline = VidXTendPipeline.from_pretrained(
            model,
            variant=variant,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
        )

    if is_xformers_available():
        pipeline.set_use_memory_efficient_attention_xformers()

    if torch_dtype is not None:
        pipeline.to(torch_dtype)

    if no_offload:
        pipeline.to(device)
    else:
        pipeline.enable_model_cpu_offload(gpu_id=gpu_id)

    if not no_slicing:
        pipeline.enable_vae_slicing()

    image_size = pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    video = Video.from_file(video)
    images = fit_image(
        video.frames_as_list,
        width=image_size,
        height=image_size,
        fit=fit,
        anchor=anchor,
    )
    num_frames = math.ceil(video.frame_rate * seconds)
    num_iterations = math.ceil(num_frames / 8)

    # Create a random number generator
    generator = torch.Generator(device=device)
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        click.echo(f"Using random seed: {seed}")
    generator.manual_seed(seed)

    click.echo(f"Generating {num_frames} frames in {num_iterations} iterations.")
    added_frames = 0
    while added_frames < num_frames:
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            image=images[-8:],  # Use the last 8 frames as the image
            input_frames_conditioning=images[:1], # Use the first overall frame as the conditioning frame,
            generator=generator,
            eta=1.0, # Set to 1.0 for deterministic results
            output_type="pil",
        )
        images.extend(result.frames[8:]) # Add the newly generated 8 frames
        added_frames += 8
        # Clear memory between iterations
        torch.cuda.empty_cache()
        gc.collect()

    video.frames = images[:num_frames]
    bytes_written = Video(images).save(output, rate=8, overwrite=True)
    click.echo(f"Wrote {len(images)} frames to {output} ({human_size(bytes_written)})")

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as ex:
        sys.stderr.write(f"{ex}\r\n")
        sys.stderr.write(traceback.format_exc())
        sys.stderr.flush()
        sys.exit(5)
