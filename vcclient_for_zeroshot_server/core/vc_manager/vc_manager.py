import logging
import os
from queue import Empty, Queue
import threading
import time
from typing import Optional
import numpy as np
from vcclient_for_zeroshot_server.const import LOGGER_NAME, VoiceCharacterDir
from vcclient_for_zeroshot_server.core.configuration_manager.configuration_manager import ConfigurationManager
from vcclient_for_zeroshot_server.core.slot_manager.slot_manager import SlotManager

from vcclient_for_zeroshot_server.core.vc_manager.pipeline.pipline_manager import PipelineManager
from vcclient_for_zeroshot_server.core.vc_manager.utils.audio_splitter import Frame, tagging
from vcclient_for_zeroshot_server.core.voice_character_slot_manager.voice_character_slot_manager import VoiceCharacterSlotManager
import wave
from scipy.io import wavfile
import samplerate as sr
import torch

from silero_vad import load_silero_vad, get_speech_timestamps
import tempfile

from vcclient_for_zeroshot_server.utils.color_print import color_text


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.

    Takes path, PCM audio data, and sample rate.
    """
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


global_index = 0


class VCManager:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:

            cls._instance = cls()
            return cls._instance

        return cls._instance
    
    def __init__(self):
        self.loaded_model_slot_id = -1
        self.pipeline = None

        self.start_convert = True
        self.convert_queue_in = Queue()
        self.convert_queue_out = Queue()

        self.rest_frames: list[Frame] = []
        self.rest_tags: list[bool] = []

        self.resampler_in = sr.Resampler("sinc_best", channels=1)
        self.resampler_in_48k_to_16k = sr.Resampler("sinc_best", channels=1)
        self.val_buffer_sec = 30
        self.vad_audio = np.array([0] * 16000 * self.val_buffer_sec, dtype=np.float32)  # 16Kのn秒分のデータ(for vad)
        self.audio = np.array([0] * 48000 * self.val_buffer_sec, dtype=np.float32)  # 48Kのn秒分のデータ(input)

        self.vad_model = load_silero_vad()

    def start(self):
        self.convert_thread = threading.Thread(target=self.convert_thread_func)
        self.convert_thread.start()

    def warmup(self):
        dummy_wave = np.random.rand(48000 * 1).astype(np.float32)
        try:
            self.run(dummy_wave, diffusion_steps=1)
        except Exception as e:
            # import traceback
            # traceback.print_exc()
            print("warmup error", e)
            


    def stop(self):
        self.start_convert = False
        self.convert_thread.join()
        print("[converter] loop thread finalized.")

    def _find_voice_start_end(self, tags: list[bool], n: int, flag: bool = True) -> int:
        start = -1
        consecutive_count = 0

        for i in range(len(tags)):
            if tags[i] is flag:
                consecutive_count += 1
            else:
                consecutive_count = 0

            if consecutive_count == n:
                start = i - n + 1  # 10個連続の最初の位置を計算
                break

        return start

    def convert_thread_func(self):
        global global_index
        convert_chunk_sec = 1
        while self.start_convert:
            # conf = ConfigurationManager.get_instance().get_configuration()
            # 書き起こし処理
            try:
                # (1) Queueから変換対象データを取得
                if self.convert_queue_in.empty():
                    time.sleep(0.5)
                    continue
                audio = np.array([])
                #  convert_chunk_sec単位で変換を実施
                while len(audio) < 48000 * convert_chunk_sec:
                    if self.convert_queue_in.empty():
                        if self.start_convert is False:
                            break
                        time.sleep(0.5)
                        continue
                    new_audio = self.convert_queue_in.get()
                    assert isinstance(new_audio, np.ndarray)
                    audio = np.concatenate([audio, new_audio])
                    audio = audio.astype(np.float32)  # float64->float32

                try:
                    # (2) VAD処理
                    # 入力は48K, float32で入ってくる想定。
                    global_index += 1

                    # print("vad_input:", audio.shape, audio.dtype, audio.max(),audio.min() )
                    # silero-vad向けに16Kに変換してから処理する
                    wav_16k = self.resampler_in_48k_to_16k.process(audio, 16000 / 48000, end_of_input=False)
                    self.vad_audio = np.concatenate([self.vad_audio, wav_16k])
                    self.vad_audio = self.vad_audio[-16000 * self.val_buffer_sec :]
                    self.audio = np.concatenate([self.audio, audio])
                    self.audio = self.audio[-48000 * self.val_buffer_sec :]

                    vad_result = get_speech_timestamps(
                        self.vad_audio,
                        self.vad_model,
                        sampling_rate=16000,
                        return_seconds=False,
                        min_speech_duration_ms=50,
                        min_silence_duration_ms=200,
                        threshold=0.2,
                    )
                    voice_end = 0
                    print(f"vad_result:{len(vad_result)}", vad_result)
                    for i, voice in enumerate(vad_result):
                        start = voice["start"]
                        end = voice["end"]
                        start = start - 1600  # 前後10msを多めにとっておく
                        end = end + 1600  # 前後10msを多めにとっておく
                        if start < 0:
                            start = 0
                        if end > len(self.vad_audio):
                            # endがvad_audioより長い場合は、発話が終了していないとみなして、次の入力を待つ（breakする）
                            # print(f"voice range: {start}->{end} no finish")
                            break

                        # # vadデータ(debug用)
                        # vad_no_voice_sample = self.vad_audio[voice_end:start]
                        # vad_voice_sample = self.vad_audio[start:end]

                        # vadに対応するオリジナルの入力データの抽出(アドレスを16k -> 48kに変換)
                        no_voice_sample_length = start - voice_end
                        if no_voice_sample_length < 0:
                            no_voice_sample_length = 0
                        no_voice_sample = np.zeros(no_voice_sample_length * 3, dtype=np.float32)
                        voice_sample = self.audio[start * 3 : end * 3]

                        # print(f"voice range: {start}->{end} length:{len(voice_sample)}, {end-start}")

                        # (3) 変換処理
                        # 対象をファイルに落としてから変換処理を実施。
                        # (pipeline内で読み込み時にサンプリングレートを適切に変換してくれるから。)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=".") as tmp_file:
                            pass
                        wavfile.write(tmp_file.name, 48000, voice_sample)
                        sr, converted_audio = self.run(wave_path=tmp_file.name)
                        os.remove(tmp_file.name)
                        voice_end = end  # 次のバッファの開始位置

                        # convert
                        converted_audio = np.clip(converted_audio, -1.0, 1.0)
                        converted_audio = converted_audio.reshape(-1)

                        converted_audio = self.resampler_in.process(converted_audio, 48000 / sr, end_of_input=False)

                        if len(no_voice_sample) > 0:
                            self.convert_queue_out.put((False, no_voice_sample))
                        self.convert_queue_out.put((True, converted_audio))
                        # file_name_converted = f"x_input_{global_index:03}_{i:03}_voice_{isVoice}_converted.wav"
                        # wavfile.write(file_name_converted, 48000 ,converted_audio)

                    self.vad_audio = self.vad_audio[voice_end:]
                    self.audio = self.audio[voice_end * 3 :]

                except Exception as e:
                    print("Error(2)", e)
                    import traceback

                    traceback.print_exc()
                continue

            except Empty:
                print("pass")
                pass
            except Exception as e:
                print("Error(1)", e)
                import traceback

                logging.getLogger(LOGGER_NAME).warn(f"Failed to convert_chunk_bulk_internal:{e}")
                logging.getLogger(LOGGER_NAME).warn(f"Failed to convert_chunk_bulk_internal:{traceback.format_exc()}")
        print("[loop] end")

    def put_chunk_to_convert_queue(self, chunk: np.ndarray):
        self.convert_queue_in.put(chunk)

    def get_chunk_from_converted_queue(self) -> np.ndarray:
        try:
            return self.convert_queue_out.get_nowait()
        except Empty:
            return None

    def load_model(self, slot_id: int):
        self.loaded_model_slot_id = slot_id
        slot_manager = SlotManager.get_instance()
        slot_info = slot_manager.get_slot_info(slot_id)
        self.pipeline = PipelineManager.get_pipeline(slot_info)

    def check_and_load_model(self):
        conf = ConfigurationManager.get_instance().get_server_configuration()
        # slot_info = SlotManager.get_instance().get_slot_info(conf.current_slot_index)

        # Check
        exec_load = False
        if self.pipeline is None:
            exec_load = True
        elif self.loaded_model_slot_id != conf.current_slot_index:
            exec_load = True
        elif self.pipeline.gpu_device_id != conf.gpu_device_id_int:
            exec_load = True
        # Load
        if exec_load is True:
            self.load_model(conf.current_slot_index)

    def run(self, waveform: np.ndarray | None = None, wave_path: str | None = None, diffusion_steps: Optional[int] = None):
        conf = ConfigurationManager.get_instance().get_server_configuration()

        slot_manager = SlotManager.get_instance()
        slot_info = slot_manager.get_slot_info(conf.current_slot_index)
        self.check_and_load_model()

        assert self.pipeline is not None, "Model is not loaded"

        voice_character_slot_manager = VoiceCharacterSlotManager.get_instance()
        print(color_text(f"voice character slot -> ",color="GREEN", format="BOLD") + str(conf.current_voice_charcter_slot_index))
        
        voice_character = voice_character_slot_manager.get_slot_info(conf.current_voice_charcter_slot_index)
        current_reference_voice_index = conf.current_reference_voice_indexes[conf.current_voice_charcter_slot_index]
        reference_voices = [v for v in voice_character.reference_voices if v.slot_index == current_reference_voice_index]
        assert len(reference_voices) == 1, f"reference voice not found. voice_character_slot_index:{conf.current_voice_charcter_slot_index}, reference_voice_slot_index:{current_reference_voice_index}"
        reference_voice = reference_voices[0]

        slot_dir = VoiceCharacterDir / f"{conf.current_voice_charcter_slot_index}"

        ref_wav_path = slot_dir / f"{reference_voice.wav_file}"

        synthesis_result = self.pipeline.run(
            ref_wav_path=str(ref_wav_path),
            src_wav_path=wave_path,
            source_audio=waveform,
            length_adjust=slot_info.length_adjust,
            auto_f0_adjust=slot_info.auto_f0_adjust,
            semi_tone_shift=slot_info.semi_tone_shift,
            diffusion_steps=slot_info.diffusion_steps if diffusion_steps is None else diffusion_steps,
            inference_cfg_rate=slot_info.inference_cfg_rate,
        )
        last_sampling_rate, last_audio_data = synthesis_result

        return last_sampling_rate, last_audio_data
