from pathlib import Path
from pydantic import BaseModel

from vcclient_for_zeroshot_server.const import BasicVoiceType, VCType
from vcclient_for_zeroshot_server.const import LanguageType

class ModelImportParam(BaseModel):
    vc_type: VCType
    name: str
    terms_of_use_url: str = ""
    slot_index: int | None = None
    icon_file: Path | None = None


class SeedVCModelImportParam(ModelImportParam):
    repository_id: str | None = None
    seed_vc_model_file: Path
    seed_vc_config_file: Path
    f0_condition: bool = False


ModelImportParamMember = ModelImportParam | SeedVCModelImportParam


class SlotInfo(BaseModel):
    vc_type: VCType | None = None
    slot_index: int = -1
    name: str = ""
    description: str = ""
    credit: str = ""
    terms_of_use_url: str = ""
    icon_file: Path | None = None


class SeedVCSlotInfo(SlotInfo):
    vc_type: VCType = "seed-vc"
    repository_id: str | None = None
    seed_vc_model_file: Path
    seed_vc_config_file: Path

    f0_condition: bool = False
    fp16: bool = False

    length_adjust: float = 1.0
    auto_f0_adjust: bool = False
    semi_tone_shift: int = 0
    diffusion_steps: int = 30
    inference_cfg_rate: float = 0.7


SlotInfoMember = SlotInfo | SeedVCSlotInfo


class ReferenceVoiceImportParam(BaseModel):
    voice_type: BasicVoiceType | str
    wav_file: Path
    slot_index: int | None = None
    icon_file: Path | None = None
    text: str | None = None


class ReferenceVoice(BaseModel):
    voice_type: BasicVoiceType | str
    slot_index: int = -1
    wav_file: Path
    text: str
    language: LanguageType
    icon_file: Path | None = None


class VoiceCharacterImportParam(BaseModel):
    vc_type: VCType
    name: str
    terms_of_use_url: str = ""
    slot_index: int | None = None
    icon_file: Path | None = None
    zip_file: Path | None = None


class VoiceCharacter(BaseModel):
    vc_type: VCType | None = None
    slot_index: int = -1
    name: str = ""
    description: str = ""
    credit: str = ""
    terms_of_use_url: str = ""
    icon_file: Path | None = None
    reference_voices: list[ReferenceVoice] = []


class MoveModelParam(BaseModel):
    src: int
    dst: int


class MoveReferenceVoiceParam(BaseModel):
    src: int
    dst: int


class SetIconParam(BaseModel):
    icon_file: Path
