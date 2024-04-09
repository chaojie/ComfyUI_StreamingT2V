import torch
import torch.nn as nn
from typing import Union
from torch.nn.common_types import _size_2_t

__all__ = ["Conv2DSubChannels", "Conv2DExtendedChannels"]

class Conv2DSubChannels(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):

        if prefix + "weight" in state_dict and (
            (state_dict[prefix + "weight"].shape[0] > self.out_channels)
            or (state_dict[prefix + "weight"].shape[1] > self.in_channels)
        ):
            print(
                f"Model checkpoint has too many channels. Excluding channels of convolution {prefix}."
            )
            if self.bias is not None:
                bias = state_dict[prefix + "bias"][: self.out_channels]
                state_dict[prefix + "bias"] = bias
                del bias

            weight = state_dict[prefix + "weight"]
            state_dict[prefix + "weight"] = weight[
                : self.out_channels, : self.in_channels
            ]
            del weight

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class Conv2DExtendedChannels(nn.Conv2d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        in_channel_extension: int = 0,
        out_channel_extension: int = 0,
    ) -> None:
        super().__init__(
            in_channels + in_channel_extension,
            out_channels + out_channel_extension,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        print(f"Call extend channel loader with {prefix}")
        if prefix + "weight" in state_dict and (
            state_dict[prefix + "weight"].shape[0] < self.out_channels
            or state_dict[prefix + "weight"].shape[1] < self.in_channels
        ):
            print(
                f"Model checkpoint has insufficient channels. Extending channels of convolution {prefix} by adding zeros."
            )
            if self.bias is not None:
                bias = state_dict[prefix + "bias"]
                state_dict[prefix + "bias"] = torch.cat(
                    [
                        bias,
                        torch.zeros(
                            self.out_channels - len(bias),
                            dtype=bias.dtype,
                            layout=bias.layout,
                            device=bias.device,
                        ),
                    ]
                )
                del bias

            weight = state_dict[prefix + "weight"]
            extended_weight = torch.zeros(
                self.out_channels,
                self.in_channels,
                weight.shape[2],
                weight.shape[3],
                device=weight.device,
                dtype=weight.dtype,
                layout=weight.layout,
            )
            extended_weight[: weight.shape[0], : weight.shape[1]] = weight
            state_dict[prefix + "weight"] = extended_weight
            del extended_weight
            del weight

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
