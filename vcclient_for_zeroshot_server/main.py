import logging
import platform
import signal
import time
import fire
from simple_performance_timer.Timer import Timer
from vcclient_for_zeroshot_server.core.configuration_manager.configuration_manager import ConfigurationManager
from vcclient_for_zeroshot_server.core.data_types.slot_manager_data_types import SeedVCModelImportParam, VoiceCharacterImportParam
from vcclient_for_zeroshot_server.core.slot_manager.slot_manager import SlotManager
from vcclient_for_zeroshot_server.core.vc_manager.vc_manager import VCManager
from vcclient_for_zeroshot_server.server.server import Server
from vcclient_for_zeroshot_server.app_status import AppStatus
from vcclient_for_zeroshot_server.const import LOG_FILE, LOGGER_NAME, VERSION
from vcclient_for_zeroshot_server.utils.download_callback import get_download_callback
from vcclient_for_zeroshot_server.utils.parseBoolArg import parse_bool_arg
from vcclient_for_zeroshot_server.utils.resolve_url import resolve_base_url
from vcclient_for_zeroshot_server.logger import setup_logger
from vcclient_for_zeroshot_server.core.gpu_device_manager.gpu_device_manager import GPUDeviceManager
from vcclient_for_zeroshot_server.proxy.ngrok_proxy_manager import NgrokProxyManager
from vcclient_for_zeroshot_server.core.voice_character_slot_manager.voice_character_slot_manager import VoiceCharacterSlotManager
from vcclient_for_zeroshot_server.core.module_manager.module_manager import ModuleManager


setup_logger(LOGGER_NAME, LOG_FILE)


default_model_params: list[SeedVCModelImportParam] = [
    # SeedVCModelImportParam(
    #     slot_index=None,
    #     name="xlsr-tiny-v1",
    #     vc_type="seed-vc",
    #     repository_id="Plachta/Seed-VC",
    #     seed_vc_model_file="DiT_uvit_tat_xlsr_ema.pth",
    #     seed_vc_config_file="config_dit_mel_seed_uvit_xlsr_tiny.yml",
    #     f0_condition=False,
    # ),
    # SeedVCModelImportParam(
    #     slot_index=None,
    #     name="whisper_small_wavenet_bigvgan",
    #     vc_type="seed-vc",
    #     repository_id="Plachta/Seed-VC",
    #     seed_vc_model_file="DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
    #     seed_vc_config_file="config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
    #     f0_condition=False,
    # ),
    SeedVCModelImportParam(
        slot_index=None,
        name="whisper_base_f0_44k_bigvgan",
        vc_type="seed-vc",
        repository_id="Plachta/Seed-VC",
        seed_vc_model_file="DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
        seed_vc_config_file="config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
        f0_condition=True,
    ),
]


def download_initial_models() -> None:
    # ModelDirの中身が空だった場合
    slot_manager = SlotManager.get_instance()
    slot_infos = slot_manager.get_slot_infos()
    slot_infos = [slot_info for slot_info in slot_infos if slot_info.vc_type is not None]
    if len(slot_infos) > 0:
        return

    # ファイルのダウンロード
    download_callback = get_download_callback()
    module_manager = ModuleManager.get_instance()
    module_manager.download_initial_models(download_callback)

    # JVNV Voicesの設定
    for v in ["JVNV_F1_VOICE", "JVNV_F2_VOICE", "JVNV_M1_VOICE", "JVNV_M2_VOICE"]:
        voice_character_slot_manager = VoiceCharacterSlotManager.get_instance()
        zipfile = module_manager.get_module_filepath(v)
        voice_character_import_param = VoiceCharacterImportParam(
            vc_type="seed-vc",
            name=v,
            terms_of_use_url="",
            zip_file=zipfile,
        )
        voice_character_slot_manager.set_new_slot(voice_character_import_param, remove_src=True)

        conf = ConfigurationManager.get_instance().get_server_configuration()
        conf.current_voice_charcter_slot_index = 0
        conf.current_reference_voice_indexes = [0]
        ConfigurationManager.get_instance().set_server_configuration(conf)

def start(
    host: str = "0.0.0.0",
    port: int = 18000,
    https: bool = False,
    launch_client: bool = True,
    allow_origins=None,
    ngrok_token: str | None = None,
    ngrok_proxy_url_file: str | None = None,
):
    timer_enabled = False
    https = parse_bool_arg(https)
    launch_client = parse_bool_arg(launch_client)

    with Timer("start", enalbe=timer_enabled) as t:
        logging.getLogger(LOGGER_NAME).info(f"Starting VCClient version:{VERSION}")

        if ngrok_token is not None and https is True:
            print("ngrok with https is not supported.")
            print("use http.")
            return

        GPUDeviceManager.get_instance()

        download_initial_models()

        slot_manager = SlotManager.get_instance()
        slot_infos = slot_manager.get_slot_infos()
        default_model_slots = [x for x in slot_infos if x.vc_type == "seed-vc"]
        if len(default_model_slots) == 0:
            for param in default_model_params:
                slot_manager.set_new_slot(param)
            conf = ConfigurationManager.get_instance().get_server_configuration()
            conf.current_slot_index = 0
            ConfigurationManager.get_instance().set_server_configuration(conf)
            

        # if slot_manager.get_slot_infos() is None:

        # 各種プロセス起動
        app_status = AppStatus.get_instance()
        VCManager.get_instance().start()
        VCManager.get_instance().warmup()
        
        # # (1) VCServer 起動
        allow_origins = "*"
        server = Server.get_instance(host=host, port=port, https=https, allow_origins=allow_origins)
        server_port = server.start()

        # # (2) NgrokProxy
        if ngrok_token is not None:
            try:
                proxy_manager = NgrokProxyManager.get_instance()
                proxy_url = proxy_manager.start(server_port, token=ngrok_token)
                # print(f"NgrokProxy:{proxy_url}")
                logging.getLogger(LOGGER_NAME).info(f"NgrokProxy: {proxy_url}")
                if ngrok_proxy_url_file is not None:
                    with open(ngrok_proxy_url_file, "w") as f:
                        f.write(proxy_url)

            except Exception as e:
                logging.getLogger(LOGGER_NAME).error(f"NgrokProxy Error:{e}")
                print("NgrokProxy Error:", e)
                print("")
                print("Ngrok proxy is not launched. Shutdown server... ")
                print("")
                server.stop()
                return
        else:
            proxy_manager = None
            proxy_url = None

        base_url = resolve_base_url(https, port)

        bold_green_start = "\033[1;32m"
        reset = "\033[0m"
        title = "    vcclient for seed-vc    "
        urls = [
            ["Application", base_url],
            ["Log(rich)", f"{base_url}/?app_mode=LogViewer"],
            ["Log(text)", f"{base_url}/vcclient.log"],
            ["API", f"{base_url}/docs"],
            ["License(js)", f"{base_url}/licenses-js.json"],
            ["License(py)", f"{base_url}/licenses-py.json"],
        ]

        if proxy_url is not None:
            urls.append(["Ngrok", proxy_url])

        key_max_length = max(len(url[0]) for url in urls)
        url_max_length = max(len(url[1]) for url in urls)

        padding = (key_max_length + url_max_length + 3 - len(title)) // 2

        if platform.system() != "Darwin":

            def gradient_text(text, start_color, end_color):
                text_color = (0, 255, 0)  # Green color for the text
                n = len(text)
                grad_text = ""
                for i, char in enumerate(text):
                    r = int(start_color[0] + (end_color[0] - start_color[0]) * i / n)
                    g = int(start_color[1] + (end_color[1] - start_color[1]) * i / n)
                    b = int(start_color[2] + (end_color[2] - start_color[2]) * i / n)
                    grad_text += f"\033[1m\033[38;2;{text_color[0]};{text_color[1]};{text_color[2]}m\033[48;2;{r};{g};{b}m{char}"
                return grad_text + reset

            start_color = (18, 121, 255)
            end_color = (0, 58, 158)
            print("")
            print(" " * padding + gradient_text(" " * len(title), start_color, end_color))
            print(" " * padding + gradient_text(title, start_color, end_color))
            print(" " * padding + gradient_text(" " * len(title), start_color, end_color))
        else:
            print("")
            print(f"{bold_green_start}{title}{reset}")
            print("")

        print("-" * (key_max_length + url_max_length + 5))
        for url in urls:
            print(f" {bold_green_start}{url[0].ljust(key_max_length)}{reset} | {url[1]} ")
        print("-" * (key_max_length + url_max_length + 5))

        logging.getLogger(LOGGER_NAME).info("--- VCClient READY ---")
        print(f"{bold_green_start}Please press Ctrl+C once to exit vcclient.{reset}")

        # # # (4)Native Client 起動
        # # if launch_client and platform.system() != "Darwin":
        # clinet_launcher = ClientLauncher(app_status.stop_app)
        # clinet_launcher.launch(port, https)

    try:
        while True:
            current_time = time.strftime("%Y/%m/%d %H:%M:%S")
            logging.getLogger(LOGGER_NAME).info(f"{current_time}: running...")
            if app_status.end_flag is True:
                break
            time.sleep(60)
    except KeyboardInterrupt:
        err_msg = "KeyboardInterrupt"

    print(f"{bold_green_start}terminate vcclient...{reset}")
    # 終了処理
    with Timer("end_cui", enalbe=timer_enabled) as t:  # noqa

        def ignore_ctrl_c(signum, frame):
            print(f"{bold_green_start}Ctrl+C is disabled during this process{reset}")

        original_handler = signal.getsignal(signal.SIGINT)

        try:
            signal.signal(signal.SIGINT, ignore_ctrl_c)
            # # (3)Native Client 終了(サーバとの通信途中でのサーバ停止を極力避けるため、クライアントから落とす。)
            # if launch_client:
            #     clinet_launcher.stop()
            VCManager.get_instance().stop()

            # # (1) VCServer 終了処理
            print(f"{bold_green_start}vcclient is terminating...{reset}")
            server.stop()
            print(f"{bold_green_start}vcclient is terminated.[{server_port}]{reset}")

            # TTSManager.get_instance().stop_tts()

            if len(err_msg) > 0:
                print("msg: ", err_msg)

            # ngrok
            if proxy_manager is not None:
                proxy_manager.stop()
        finally:
            print("")
            # signal.signal(signal.SIGINT, original_handler)

        signal.signal(signal.SIGINT, original_handler)


def main():
    fire.Fire(
        {
            "start": start,
        }
    )


if __name__ == "__main__":
    main()
