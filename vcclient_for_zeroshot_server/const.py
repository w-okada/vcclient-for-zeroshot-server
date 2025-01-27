from pathlib import Path
from typing import Literal, TypeAlias


# GLOBAL CONSTANTS
HERE = Path(__file__).parent.absolute()


def get_version():
    with open(HERE / "version.txt", "r", encoding="utf-8") as f:
        return f.read().strip()


APP_NAME = "VCClient"
VERSION = get_version()
LOGGER_NAME = "vcclient"
LOG_FILE = Path("vcclient.log")
ModuleDir = Path("modules")

UPLOAD_DIR_STR = "upload_dir"
UPLOAD_DIR = Path(UPLOAD_DIR_STR)
UPLOAD_DIR.mkdir(exist_ok=True)

SSL_KEY_DIR = Path("ssl_key")


def get_frontend_path():
    frontend_path = "web_front"
    return Path(frontend_path)


# seed-vc モジュール
VCType: TypeAlias = Literal["seed-vc"]
VCTypes: list[VCType] = [
    "seed-vc",
]

# SeedVCType: TypeAlias = Literal["seed-uvit-tat-xlsr-tiny-v1", "seed-uvit-whisper-small-wavenet-v1", "seed-uvit-whisper-base-v1", "custom"]
# SeedVCTypes: list[SeedVCType] = [
#     "seed-uvit-tat-xlsr-tiny-v1",
#     "seed-uvit-whisper-small-wavenet-v1",
#     "seed-uvit-whisper-base-v1",
#     "custom",
# ]


# Configu Manager
SettingsDir = Path("./settings")
SettingsDir.mkdir(parents=True, exist_ok=True)
ConfigFile = SettingsDir / "tts_conf.json"

# Slot Manager
ModelDir = Path("./models")
ModelDir.mkdir(parents=True, exist_ok=True)
MAX_SLOT_INDEX = 20
SLOT_PARAM_FILE = "params.json"

# Voice Character Slot Manager
VoiceCharacterDir = Path("./voice_characters")
VoiceCharacterDir.mkdir(parents=True, exist_ok=True)
MAX_VOICE_CHARACTER_SLOT_INDEX = 200
VOICE_CHARACTER_SLOT_PARAM_FILE = "params.json"
MAX_REFERENCE_VOICE_SLOT_INDEX = 100

BasicVoiceType: TypeAlias = Literal["anger", "disgust", "fear", "happy", "sad", "surprise"]
BasicVoiceTypes: list[BasicVoiceType] = [
    "anger",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
]

LanguageType: TypeAlias = Literal[
    "all_zh",  # 全部按中文识别
    "en",  # 全部按英文识别#######不变
    "all_ja",  # 全部按日文识别
    "all_yue",  # 全部按中文识别
    "all_ko",  # 全部按韩文识别
    "zh",  # 按中英混合识别####不变
    "ja",  # 按日英混合识别####不变
    "yue",  # 按粤英混合识别####不变
    "ko",  # 按韩英混合识别####不变
    "auto",  # 多语种启动切分识别语种
    "auto_yue",  # 多语种启动切分识别语种
]
LanguageTypes: list[LanguageType] = [
    "all_zh",
    "en",
    "all_ja",
    "all_yue",
    "all_ko",
    "zh",
    "ja",
    "yue",
    "ko",
    "auto",
    "auto_yue",
]
