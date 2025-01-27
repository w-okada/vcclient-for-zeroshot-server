import logging
from pathlib import Path
import shutil
from typing import cast

from vcclient_for_zeroshot_server.const import LOGGER_NAME, SLOT_PARAM_FILE
from vcclient_for_zeroshot_server.core.data_types.slot_manager_data_types import ModelImportParamMember, SeedVCModelImportParam, SeedVCSlotInfo


def import_model(model_dir: Path, model_importer_param: ModelImportParamMember, remove_src: bool = False):
    slot_dir = model_dir / f"{model_importer_param.slot_index}"
    slot_dir.mkdir(parents=True, exist_ok=True)
    try:
        if model_importer_param.vc_type == "seed-vc":
            assert isinstance(model_importer_param, SeedVCModelImportParam)
            if model_importer_param.repository_id is None:
                for src in cast(list[Path | None], [model_importer_param.icon_file, model_importer_param.seed_vc_config_file, model_importer_param.seed_vc_model_file]):
                    if src is not None:
                        dst = slot_dir / src.name
                        if len(str(src)) > 80 or len(str(dst)) > 80:
                            raise RuntimeError(f"filename is too long: {src} -> {dst}")
                        logging.getLogger(LOGGER_NAME).debug(f"copy {src} to {dst}")
                        shutil.copy(src, dst)
                        if remove_src is True:
                            src.unlink()

            # generate config file
            assert model_importer_param.slot_index is not None
            slot_info = SeedVCSlotInfo(
                slot_index=model_importer_param.slot_index,
                name=model_importer_param.name,
                icon_file=Path(model_importer_param.icon_file.name) if model_importer_param.icon_file is not None else None,
                repository_id=model_importer_param.repository_id,
                seed_vc_model_file=Path(model_importer_param.seed_vc_model_file.name),
                seed_vc_config_file=Path(model_importer_param.seed_vc_config_file.name),
                f0_condition=model_importer_param.f0_condition,
            )
            slot_info.terms_of_use_url = model_importer_param.terms_of_use_url

        else:
            logging.getLogger(LOGGER_NAME).error(f"Unknown tts type: {model_importer_param.tts_type}")
            raise RuntimeError(f"Unknown tts type: {model_importer_param.tts_type}")
    except RuntimeError as e:
        shutil.rmtree(slot_dir)
        raise e

    config_file = slot_dir / SLOT_PARAM_FILE
    with open(config_file, "w", encoding="utf-8") as f:
        f.write(slot_info.model_dump_json(indent=4))
