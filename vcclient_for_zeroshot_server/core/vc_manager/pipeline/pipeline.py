from abc import ABC, abstractmethod

import numpy as np


class Pipeline(ABC):

    def __init__(self, gpu_device_id=None):
        self.gpu_device_id = gpu_device_id

    @abstractmethod
    def force_stop(self):
        pass

    @abstractmethod
    def run(
        self,
        ref_wav_path: str,
        src_wav_path: str | None = None,
        source_audio: np.ndarray | None = None,
        length_adjust: float = 1.0,
        auto_f0_adjust: bool = False,
        semi_tone_shift: int = 0,
        diffusion_steps: int = 30,
        inference_cfg_rate: float = 0.7,
    ):
        pass
