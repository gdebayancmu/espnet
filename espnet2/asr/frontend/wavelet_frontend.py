import copy
import os

from typing import Optional
from typing import Tuple
from typing import Union
from ssqueezepy import cwt

import humanfriendly
import numpy as np
import torch
from torch_complex.tensor import ComplexTensor
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.frontends.frontend import Frontend
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.layers.log_mel import LogMel
from espnet2.layers.stft import Stft
from espnet2.utils.get_default_kwargs import get_default_kwargs


class WaveletFrontEnd(AbsFrontend):

    def __init__(
            self,
            fs: Union[int, str] = 16000,
            win_length: int = 320,
            hop_length: int = 160,
            **kwargs
    ):
        os.environ['SSQ_GPU'] = '1'
        assert check_argument_types()

        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        self.fs = fs
        self.win_length = win_length
        self.hop_length = hop_length
        self.op_size = 221  # Currently Hardcoded

    def output_size(self) -> int:
        return self.op_size

    def frame_with_overlap(self, signal: np.ndarray, Nw, Nsh) -> None:
        """
        Creates overlapping windows (frames) of the input signal
        input signal: Input audio signal
        param window_length: window length (in samples)
        param overlap_length: overlapping length (in samples)
        """
        signal = signal.cpu()
        signal_len = signal.shape[0]
        num_frames = np.ceil(signal_len / Nsh)
        framed_signal = np.zeros((int(num_frames), Nw))
        i = 0
        fr_id = 0
        while i < signal_len and fr_id < num_frames:
            pad_len = Nw - signal[i:i + Nw].shape[0]
            framed_signal[fr_id, :] = np.pad(signal[i:i + Nw], (0, pad_len), 'constant', constant_values=(0, 0))

            if i + Nw > signal_len:
                break

            fr_id += 1
            i += Nsh
        framed_signal = framed_signal[0:fr_id + 1, :]
        return framed_signal

    def forward(
            self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # print('----In Forward WAVELET------')
        # print('input', input.shape)
        # print('input_lengths', input_lengths)

        batch_size = input.shape[0]

        # 3. [Multi channel case]: Select a channel
        if input.dim() == 4:
            # h: (B, T, C, F) -> h: (B, T, F)
            if self.training:
                # Select 1ch randomly
                ch = np.random.randint(input.size(2))
                input = input[:, :, ch, :]
            else:
                # Use the first channel
                input = input[:, :, 0, :]

        wavelet_batch = []
        for batch in range(0, batch_size):
            torch.cuda.empty_cache()
            framed_signal = self.frame_with_overlap(input[batch], Nw=self.win_length, Nsh=self.hop_length)
            Wxo, dWx =  cwt(framed_signal, fs=self.fs, nv=32)
            # print('wxo size', Wxo.shape)
            wxo_abs = torch.mean(torch.abs(torch.tensor(Wxo)), dim=2)
            wxo_abs = torch.reshape(wxo_abs, (1, -1, self.op_size))
            wavelet_batch.append(wxo_abs)

        # print('wavelet_batch', len(wavelet_batch))
        input_feats = torch.cat(wavelet_batch, dim=0)
        feats_lens = torch.tensor([torch.div(x, self.hop_length, rounding_mode='floor') for x in input_lengths])

        # print('RETURNING WAVELET input_feats.shape', input_feats.shape, 'feat lens', feats_lens)

        return input_feats, feats_lens
