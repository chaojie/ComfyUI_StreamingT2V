# ComfyUI-StreamingT2V

15G VRAM Required

Download model from https://huggingface.co/PAIR/StreamingT2V/blob/main/streaming_t2v.ckpt

Put streaming_t2v.ckpt to ComfyUI/model/checkpoints folder

## workflow

### Step 1 Generate short video (16x256x256):

https://github.com/chaojie/ComfyUI_StreamingT2V/blob/main/wf_short_ad.json

<img src="wf_short_ad.png" raw=true>

https://github.com/chaojie/ComfyUI_StreamingT2V/blob/main/wf_short_ms.json

<img src="wf_short_ms.png" raw=true>

https://github.com/chaojie/ComfyUI_StreamingT2V/blob/main/wf_short_svd.json

<img src="wf_short_svd.png" raw=true>

### Step 2 Generate long video (nx256x256):

https://github.com/chaojie/ComfyUI_StreamingT2V/blob/main/wf_long.json

<img src="wf_long.png" raw=true>

### Step 3 Generate long enhanced video (nx512x512):

https://github.com/chaojie/ComfyUI_StreamingT2V/blob/main/wf_enhance.json

<img src="wf_enhance.png" raw=true>

### You can generate all by one workflow

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