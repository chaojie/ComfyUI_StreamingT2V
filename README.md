# ComfyUI-StreamingT2V

I have added a `vram_not_enough` option, which requires 20GB VRAM if turned on (slow), and 30GB VRAM if not

Download model from https://huggingface.co/PAIR/StreamingT2V/blob/main/streaming_t2v.ckpt

Put streaming_t2v.ckpt to ComfyUI/model/checkpoints folder

## workflow

ModelscopeT2V (T2V)

https://github.com/chaojie/ComfyUI_StreamingT2V/blob/main/wf.json

<img src="wf.png" raw=true>

AnimateDiff (T2V)

https://github.com/chaojie/ComfyUI_StreamingT2V/blob/main/wf_ad.json

<img src="wf_ad.png" raw=true>

SVD (I2V)

https://github.com/chaojie/ComfyUI_StreamingT2V/blob/main/wf_svd.json

<img src="wf_svd.png" raw=true>

## [StreamingT2V](https://github.com/Picsart-AI-Research/StreamingT2V)