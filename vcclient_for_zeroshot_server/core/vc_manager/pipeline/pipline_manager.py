from vcclient_for_zeroshot_server.core.vc_manager.pipeline.pipeline import Pipeline
from vcclient_for_zeroshot_server.core.vc_manager.pipeline.seed_vc_pipeline import SeedVCPipeline
from ...data_types.slot_manager_data_types import SeedVCSlotInfo, SlotInfo


class PipelineManager:

    @classmethod
    def get_pipeline(cls, slot_info: SlotInfo) -> Pipeline:
        if slot_info.vc_type == "seed-vc":
            assert isinstance(slot_info, SeedVCSlotInfo)
            return SeedVCPipeline(slot_info)
        else:
            raise RuntimeError(f"Unknown vc_type:{slot_info.vc_type}")
