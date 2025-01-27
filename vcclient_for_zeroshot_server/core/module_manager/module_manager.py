import logging
from pathlib import Path
from typing import Callable
import math
import requests
import hashlib
from threading import Thread


from ...const import LOGGER_NAME, UPLOAD_DIR, ModuleDir
from ..data_types.data_types import ModuleInfo, ModuleStatus
from ..data_types.module_manager_data_types import ModuleDownloadStatus


REGISTERD_MODULES: list[ModuleInfo] = [
    ModuleInfo(
        id="JVNV_F1_VOICE",
        display_name="JVNV_F1_VOICE",
        url="https://huggingface.co/wok000/gpt-sovits-models/resolve/main/jvnv-voices/JVNV_F1.zip",
        save_to=UPLOAD_DIR / Path("JVNV_F1.zip"),
        hash="78715260c07a8e9dabc31f60e10ee0e7044155ecfca9a4bbb52d15fda3438077",
    ),
    ModuleInfo(
        id="JVNV_F2_VOICE",
        display_name="JVNV_F2_VOICE",
        url="https://huggingface.co/wok000/gpt-sovits-models/resolve/main/jvnv-voices/JVNV_F2.zip",
        save_to=UPLOAD_DIR / Path("JVNV_F2.zip"),
        hash="387e03c86e16607f8085da1ddf5857486244cbe19aff2dc03c7e5f67b2dc1221",
    ),
    ModuleInfo(
        id="JVNV_M1_VOICE",
        display_name="JVNV_M1_VOICE",
        url="https://huggingface.co/wok000/gpt-sovits-models/resolve/main/jvnv-voices/JVNV_M1.zip",
        save_to=UPLOAD_DIR / Path("JVNV_M1.zip"),
        hash="eddeb129a5318257e4290ebd2622b53143dd5eae6d9c10fed76daa69917fbd9a",
    ),
    ModuleInfo(
        id="JVNV_M2_VOICE",
        display_name="JVNV_M2_VOICE",
        url="https://huggingface.co/wok000/gpt-sovits-models/resolve/main/jvnv-voices/JVNV_M2.zip",
        save_to=UPLOAD_DIR / Path("JVNV_M2.zip"),
        hash="4bb68a46498dc80bfd425b4d4bd0397e23b65fd124db6516339d86704ed87620",
    ),
]
INITIAL_MODELS = [
    "JVNV_F1_VOICE",
    "JVNV_F2_VOICE",
    "JVNV_M1_VOICE",
    "JVNV_M2_VOICE",
]


class ModuleManager:
    _instance = None
    module_status_list: list[ModuleStatus] = []

    @classmethod
    def get_instance(cls):
        if cls._instance is None:

            cls._instance = cls()
            return cls._instance

        return cls._instance

    def __init__(self):
        self.module_status_list = []
        self.threads = {}
        self.reload()
        logging.getLogger(LOGGER_NAME).info(f"Initial module status: {self.module_status_list}")

    def reload(self):
        self.module_status_list = []
        for module_info in REGISTERD_MODULES:
            module_status = ModuleStatus(info=module_info, downloaded=False, valid=False)
            logging.getLogger(LOGGER_NAME).info(f"{module_info.id} : downloaded:{module_status.downloaded}, valid:{module_status.valid}")
            if module_info.save_to.exists():
                module_status.downloaded = True
                if self._check_hash(module_info.id):
                    module_status.valid = True
            self.module_status_list.append(module_status)

    def get_modules(self) -> list[ModuleStatus]:
        return self.module_status_list

    def _download(self, target_module: ModuleInfo, callback: Callable[[ModuleDownloadStatus], None]):
        # print(params)
        # (target_module, callback) = params
        target_module.save_to.parent.mkdir(parents=True, exist_ok=True)

        req = requests.get(target_module.url, stream=True, allow_redirects=True)
        content_length_header = req.headers.get("content-length")
        content_length = int(content_length_header) if content_length_header is not None else 1024 * 1024 * 1024
        chunk_size = 1024 * 1024
        chunk_num = math.ceil(content_length / chunk_size)
        with open(target_module.save_to, "wb") as f:
            for i, chunk in enumerate(req.iter_content(chunk_size=chunk_size)):
                f.write(chunk)
                callback(
                    ModuleDownloadStatus(
                        id=target_module.id,
                        status="processing",
                        progress=min(1.0, round((i + 1) / chunk_num, 2)),
                    )
                )

        callback(
            ModuleDownloadStatus(
                id=target_module.id,
                status="validating",
                progress=min(1.0, round((i + 1) / chunk_num, 2)),
            )
        )
        try:
            self._check_hash(target_module.id)
            logging.getLogger(LOGGER_NAME).info(f"Downloading completed: {target_module.id}")
            callback(
                ModuleDownloadStatus(
                    id=target_module.id,
                    status="done",
                    progress=1.0,
                )
            )
        except Exception as e:
            logging.getLogger(LOGGER_NAME).error(f"Downloading error: {target_module.id}, {e}")
            callback(
                ModuleDownloadStatus(
                    id=target_module.id,
                    status="error",
                    progress=1.0,
                    error_message=str(e),
                )
            )

    def _get_target_module(self, id: str):
        target_module = None
        for module in REGISTERD_MODULES:
            if module.id == id:
                target_module = module
                break
        return target_module

    def download(
        self,
        id: str,
        callback: Callable[[ModuleDownloadStatus], None],
    ):
        logging.getLogger(LOGGER_NAME).info(f"Downloading module: {id}")
        # check module exists
        target_module = self._get_target_module(id)
        if target_module is None:
            logging.getLogger(LOGGER_NAME).error(f"No such module: {id}")
            callback(
                ModuleDownloadStatus(
                    id=id,
                    status="error",
                    progress=1.0,
                    error_message=f"module not found {id}",
                )
            )
            return

        # release finished thread
        for exist_trehad_id, exist_thread in list(self.threads.items()):
            if exist_thread is None or exist_thread.is_alive():
                pass  # active thread
            else:
                # thread finished
                exist_thread.join()
                self.threads.pop(exist_trehad_id)

        # check already downloading
        if id in self.threads:
            logging.getLogger(LOGGER_NAME).error(f"Already downloading: {id}")
            callback(
                ModuleDownloadStatus(
                    id=id,
                    status="error",
                    progress=1.0,
                    error_message=f"module is already downloading {id}",
                )
            )
            return

        # start download
        self.threads[id] = None  # dummy, to avoid multiple download
        logging.getLogger(LOGGER_NAME).info(f"Start downloading: {id}")
        t = Thread(
            target=self._download,
            args=(
                target_module,
                callback,
            ),
        )
        t.start()
        self.threads[id] = t

    def _check_hash(self, id: str):
        target_module = self._get_target_module(id)
        if target_module is None:
            # raise VCClientError(ERROR_CODE_MODULE_NOT_FOUND, f"Module {id} not found")
            raise RuntimeError(f"Module {id} not found")

        with open(target_module.save_to, "rb") as f:
            data = f.read()
            hash = hashlib.sha256(data).hexdigest()
            if hash != target_module.hash:
                logging.getLogger(LOGGER_NAME).error(f"hash is not valid: valid:{target_module.hash}, incoming:{hash}")
                return False
            else:
                return True

    def get_module_filepath(self, id: str):
        target_module = self._get_target_module(id)
        if target_module is None:
            return None
        return target_module.save_to

    def download_initial_modules(self, callback: Callable[[list[ModuleDownloadStatus]], None]):
        modules = [x for x in self.get_modules() if x.info.id in REQUIRED_MODULES and x.valid is False]

        logging.getLogger(LOGGER_NAME).info("---- TEST MODULES ---- start")
        test_modules = [x for x in self.get_modules() if x.info.id in REQUIRED_MODULES]
        for i in test_modules:
            logging.getLogger(LOGGER_NAME).info(f"Module:{i.info.id} -> Download:{i.downloaded}, Status:{i.valid}")
        logging.getLogger(LOGGER_NAME).info("---- TEST MODULES ---- end")

        # x.info.idをキーにした辞書配列でstatusを管理。
        status_dict = {x.info.id: ModuleDownloadStatus(id=x.info.id, status="processing", progress=0.0) for x in modules}

        # status_dictをdownloadのコールバックで更新する
        def download_callback(status: ModuleDownloadStatus):
            status_dict[status.id] = status
            callback(list(status_dict.values()))

        for module in modules:
            self.download(module.info.id, download_callback)

        for threads in self.threads.values():
            threads.join()
        print("")
        print("module download fin!")

    def download_initial_models(self, callback: Callable[[list[ModuleDownloadStatus]], None]):
        modules = [x for x in self.get_modules() if x.info.id in INITIAL_MODELS and x.valid is False]
        # x.info.idをキーにした辞書配列でstatusを管理。
        status_dict = {x.info.id: ModuleDownloadStatus(id=x.info.id, status="processing", progress=0.0) for x in modules}

        # status_dictをdownloadのコールバックで更新する
        def download_callback(status: ModuleDownloadStatus):
            status_dict[status.id] = status
            callback(list(status_dict.values()))

        for module in modules:
            self.download(module.info.id, download_callback)

        for threads in self.threads.values():
            threads.join()
        print("")
        print("initial model download fin!")
