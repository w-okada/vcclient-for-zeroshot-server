import logging
import traceback
from fastapi import APIRouter, Depends, HTTPException, Request

from vcclient_for_zeroshot_server.const import LOGGER_NAME, UPLOAD_DIR
from vcclient_for_zeroshot_server.server.validation_error_logging_route import ValidationErrorLoggingRoute
from vcclient_for_zeroshot_server.core.data_types.slot_manager_data_types import ModelImportParam, ModelImportParamMember, MoveModelParam, SeedVCModelImportParam, SeedVCSlotInfo, SetIconParam, SlotInfo, SlotInfoMember
from vcclient_for_zeroshot_server.core.slot_manager.slot_manager import SlotManager


async def detect_model(request: Request) -> SeedVCModelImportParam:
    body = await request.body()
    import_param = ModelImportParam.model_validate_json(body)

    try:
        if import_param.vc_type == "seed-vc":
            return SeedVCModelImportParam.model_validate_json(body)
        else:
            logging.getLogger(LOGGER_NAME).error(f"unknown voice_changer_type:{import_param.vc_type}")
            raise HTTPException(status_code=400, detail=f"unknown voice_changer_type:{import_param.vc_type}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def detect_slot_info(request: Request) -> SeedVCSlotInfo:

    body = await request.body()
    slot_info = SlotInfo.model_validate_json(body)
    if slot_info.vc_type == "seed-vc":
        return SeedVCSlotInfo.model_validate_json(body)
    else:
        raise RuntimeError(f"body has slot_info with unknown voice_changer_type:{slot_info.vc_type}")


class RestAPISlotManager:
    def __init__(self):
        self.router = APIRouter()
        self.router.route_class = ValidationErrorLoggingRoute
        self.router.add_api_route("/api/slot-manager/slots", self.get_slots, methods=["GET"])
        self.router.add_api_route("/api/slot-manager/slots/{index}", self.get_slot, methods=["GET"])
        self.router.add_api_route("/api/slot-manager/slots", self.post_slot, methods=["POST"])
        self.router.add_api_route("/api/slot-manager/slots/{index}", self.put_slot_info, methods=["PUT"])
        self.router.add_api_route("/api/slot-manager/slots/{index}", self.delete_slot_info, methods=["DELETE"])
        self.router.add_api_route("/api/slot-manager/slots/operation/move_model", self.post_move_model, methods=["POST"])
        self.router.add_api_route("/api/slot-manager/slots/{index}/operation/set_icon_file", self.post_set_icon_file, methods=["POST"])
        self.router.add_api_route("/api/slot-manager/slots/{index}/operation/generate_onnx", self.post_generate_onnx, methods=["POST"])

        self.router.add_api_route("/api_slot-manager_slots", self.get_slots, methods=["GET"])
        self.router.add_api_route("/api_slot-manager_slots_{index}", self.get_slot, methods=["GET"])
        self.router.add_api_route("/api_slot-manager_slots", self.post_slot, methods=["POST"])
        self.router.add_api_route("/api_slot-manager_slots_{index}", self.put_slot_info, methods=["PUT"])
        self.router.add_api_route("/api_slot-manager_slots_{index}", self.delete_slot_info, methods=["DELETE"])
        self.router.add_api_route("/api_slot-manager_slots_operation_move_model", self.post_move_model, methods=["POST"])
        self.router.add_api_route("/api_slot-manager_slots_{index}_operation_set_icon_file", self.post_set_icon_file, methods=["POST"])
        self.router.add_api_route("/api_slot-manager_slots_{index}_operation_generate_onnx", self.post_generate_onnx, methods=["POST"])

    def get_slots(self, reload: bool = False):
        slot_manager = SlotManager.get_instance()
        if reload:
            slot_manager.reload()
        slots = slot_manager.get_slot_infos()
        return slots

    def get_slot(self, index: int, reload: bool = False):
        slot_manager = SlotManager.get_instance()
        if reload:
            slot_manager.reload()
        slot = slot_manager.get_slot_info(index)
        return slot

    def post_slot(self, import_param: ModelImportParamMember = Depends(detect_model)):
        try:
            slot_manager = SlotManager.get_instance()
            if import_param.vc_type == "seed-vc":
                assert isinstance(import_param, SeedVCModelImportParam)
                import_param.seed_vc_config_file = UPLOAD_DIR / import_param.seed_vc_config_file
                import_param.seed_vc_model_file = UPLOAD_DIR / import_param.seed_vc_model_file
            slot_manager.set_new_slot(import_param, remove_src=True)
        except Exception as e:
            logging.getLogger(LOGGER_NAME).error(f"Failed to set_new_slot: {e}")
            logging.getLogger(LOGGER_NAME).error(f"Failed to set_new_slot: {traceback.format_exc()}")
            raise HTTPException(status_code=400, detail=str(e))

    def post_set_icon_file(self, index: int, param: SetIconParam):
        param.icon_file = UPLOAD_DIR / param.icon_file.name
        slot_manager = SlotManager.get_instance()
        logging.getLogger(LOGGER_NAME).info(f"set new icon, param.icon_file {param.icon_file}")
        slot_manager.set_icon_file(index, param)

    def post_generate_onnx(self, index: int):
        slot_manager = SlotManager.get_instance()
        slot_manager.generate_onnx(index)

    def put_slot_info(self, index: int, slot_info: SlotInfoMember = Depends(detect_slot_info)):
        slot_manager = SlotManager.get_instance()
        slot_manager.update_slot_info(slot_info)

    def delete_slot_info(self, index: int):
        slot_manager = SlotManager.get_instance()
        slot_manager.delete_slot(index)

    def post_move_model(self, param: MoveModelParam):
        slot_manager = SlotManager.get_instance()
        slot_manager.move_model_slot(param)
