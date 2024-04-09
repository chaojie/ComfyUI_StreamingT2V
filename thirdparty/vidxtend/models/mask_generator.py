from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = ["MaskGenerator"]

class MaskGenerator:
    def __init__(
        self,
        num_frames_conditioning: int,
        num_frames: int,
        temporal_self_attention_only_on_conditioning: bool = False,
        temporal_self_attention_mask_included_itself: bool = False,
        temp_attend_on_uncond_include_past: bool = False,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.num_frames_conditioning = num_frames_conditioning
        self.temporal_self_attention_only_on_conditioning = temporal_self_attention_only_on_conditioning
        self.temporal_self_attention_mask_included_itself = temporal_self_attention_mask_included_itself
        self.temp_attend_on_uncond_include_past = temp_attend_on_uncond_include_past

    def get_mask(
        self,
        device: torch.device,
        use_half: bool=False
    ) -> torch.Tensor:
        """
        Returns a mask to be used in the attention mechanism.
        """
        import torch
        if self.temporal_self_attention_only_on_conditioning:
            with torch.no_grad():
                attention_mask = torch.zeros(
                    (1, self.num_frames, self.num_frames),
                    dtype=torch.float16 if use_half else torch.float32,
                    device=device,
                )
                for frame in range(self.num_frames_conditioning, self.num_frames):
                    attention_mask[:, frame, self.num_frames_conditioning :] = float("-inf")
                    if self.temporal_self_attention_mask_included_itself:
                        attention_mask[:, frame, frame] = 0
                    if self.temp_attend_on_uncond_include_past:
                        attention_mask[:, frame, :frame] = 0
        else:
            attention_mask = None
        return attention_mask
