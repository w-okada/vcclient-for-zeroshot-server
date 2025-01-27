from pydantic import BaseModel


class ConvertVoiceParam(BaseModel):
    voice_character_slot_index: int
    reference_voice_slot_index: int
    length_adjust: float
    auto_f0_adjust: bool
    semi_tone_shift: int
    diffusion_steps: int
    inference_cfg_rate: float
