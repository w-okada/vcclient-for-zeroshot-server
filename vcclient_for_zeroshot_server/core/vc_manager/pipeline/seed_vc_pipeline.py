import logging
import os
from pathlib import Path
import time
import traceback
from simple_performance_timer.Timer import Timer
import librosa
import numpy as np
import torch
import torchaudio
from vcclient_for_zeroshot_server.const import LOGGER_NAME, ModelDir
from vcclient_for_zeroshot_server.core.configuration_manager.configuration_manager import ConfigurationManager
from vcclient_for_zeroshot_server.core.vc_manager.pipeline.pipeline import Pipeline
from vcclient_for_zeroshot_server.core.data_types.slot_manager_data_types import SlotInfoMember, SeedVCSlotInfo

import sys

import yaml

from vcclient_for_zeroshot_server.utils.color_print import color_text

sys.path.append("third_party/seed-vc")

from inference import load_models
from hf_utils import load_custom_model_from_hf
from modules.commons import recursive_munch, build_model, load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models(
    f0_condition: bool,
    repositry_id: str | None,
    checkpoint_path: str,
    config_path: str,
    fp16: bool,
):
    print(color_text(f"Load New Model", color="GREEN", format="BOLD"))
    print(color_text(f"  f0_condition -> ", color="CYAN", format="NORMAL"), f0_condition)
    print(color_text(f"  repositry_id -> ", color="CYAN", format="NORMAL"), repositry_id)
    print(color_text(f"  checkpoint_path -> ", color="CYAN", format="NORMAL"), checkpoint_path)
    print(color_text(f"  config_path -> ", color="CYAN", format="NORMAL"), config_path)
    print(color_text(f"  fp16 -> ", color="CYAN"), fp16)
    if not f0_condition:  # not singing mode
        print(color_text(f"Using non-singing mode", color="GREEN", format="BOLD"))
        if checkpoint_path is None:
            dit_checkpoint_path, dit_config_path = load_custom_model_from_hf("Plachta/Seed-VC", "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth", "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml")
        else:
            if repositry_id is not None:
                dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(str(repositry_id), str(checkpoint_path), str(config_path))
            else:
                dit_checkpoint_path = checkpoint_path
                dit_config_path = config_path
        f0_fn = None
    else:  # singing mode
        print(color_text(f"Using singing mode", color="GREEN", format="BOLD"))

        if checkpoint_path is None:
            dit_checkpoint_path, dit_config_path = load_custom_model_from_hf("Plachta/Seed-VC", "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth", "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml")
        else:
            if repositry_id is not None:
                dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(str(repositry_id), str(checkpoint_path), str(config_path))
            else:
                dit_checkpoint_path = checkpoint_path
                dit_config_path = config_path
        # f0 extractor
        from modules.rmvpe import RMVPE

        model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
        f0_extractor = RMVPE(model_path, is_half=False, device=device)
        f0_fn = f0_extractor.infer_from_audio


    import builtins
    original_print = builtins.print
    def custom_print(*args, **kwargs):
        text = "[module] " + " ".join(map(str, args))
        original_print(color_text(text, format="FAINT"), **kwargs)
    builtins.print = custom_print


    config = yaml.safe_load(open(dit_config_path, "r"))
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = "DiT"
    model = build_model(model_params, stage="DiT")
    hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
    sr = config["preprocess_params"]["sr"] # 44100, 22050 コンフィグで決まる

    # Load checkpoints
    model, _, _, _ = load_checkpoint(
        model,
        None,
        dit_checkpoint_path,
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
    )

    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    # Load additional modules
    from modules.campplus.DTDNN import CAMPPlus

    campplus_ckpt_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval()
    campplus_model.to(device)

    vocoder_type = model_params.vocoder.type

    print(f"Vocoder type: {vocoder_type}")
    if vocoder_type == "bigvgan":
        from modules.bigvgan import bigvgan

        bigvgan_name = model_params.vocoder.name
        bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
        # remove weight norm in the model and set to eval mode
        bigvgan_model.remove_weight_norm()
        bigvgan_model = bigvgan_model.eval().to(device)
        vocoder_fn = bigvgan_model
    elif vocoder_type == "hifigan":
        from modules.hifigan.generator import HiFTGenerator
        from modules.hifigan.f0_predictor import ConvRNNF0Predictor

        hift_config = yaml.safe_load(open("third_party/seed-vc/configs/hifigan.yml", "r"))
        hift_gen = HiFTGenerator(**hift_config["hift"], f0_predictor=ConvRNNF0Predictor(**hift_config["f0_predictor"]))
        hift_path = load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", "hift.pt", None)
        hift_gen.load_state_dict(torch.load(hift_path, map_location="cpu"))
        hift_gen.eval()
        hift_gen.to(device)
        vocoder_fn = hift_gen
    elif vocoder_type == "vocos":
        vocos_config = yaml.safe_load(open(model_params.vocoder.vocos.config, "r"))
        vocos_path = model_params.vocoder.vocos.path
        vocos_model_params = recursive_munch(vocos_config["model_params"])
        vocos = build_model(vocos_model_params, stage="mel_vocos")
        vocos_checkpoint_path = vocos_path
        vocos, _, _, _ = load_checkpoint(vocos, None, vocos_checkpoint_path, load_only_params=True, ignore_modules=[], is_distributed=False)
        _ = [vocos[key].eval().to(device) for key in vocos]
        _ = [vocos[key].to(device) for key in vocos]
        total_params = sum(sum(p.numel() for p in vocos[key].parameters() if p.requires_grad) for key in vocos.keys())
        print(f"Vocoder model total parameters: {total_params / 1_000_000:.2f}M")
        vocoder_fn = vocos.decoder
    else:
        raise ValueError(f"Unknown vocoder type: {vocoder_type}")

    speech_tokenizer_type = model_params.speech_tokenizer.type
    if speech_tokenizer_type == "whisper":
        # whisper
        from transformers import AutoFeatureExtractor, WhisperModel

        whisper_name = model_params.speech_tokenizer.name
        whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)
        del whisper_model.decoder
        whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

        def semantic_fn(waves_16k):
            ori_inputs = whisper_feature_extractor([waves_16k.squeeze(0).cpu().numpy()], return_tensors="pt", return_attention_mask=True, sampling_rate=16000)
            ori_input_features = whisper_model._mask_input_features(ori_inputs.input_features, attention_mask=ori_inputs.attention_mask).to(device)
            with torch.no_grad():
                ori_outputs = whisper_model.encoder(
                    ori_input_features.to(whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, : waves_16k.size(-1) // 320 + 1]
            return S_ori

    elif speech_tokenizer_type == "cnhubert":
        from transformers import (
            Wav2Vec2FeatureExtractor,
            HubertModel,
        )

        hubert_model_name = config["model_params"]["speech_tokenizer"]["name"]
        hubert_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model_name)
        hubert_model = HubertModel.from_pretrained(hubert_model_name)
        hubert_model = hubert_model.to(device)
        hubert_model = hubert_model.eval()
        hubert_model = hubert_model.half()

        def semantic_fn(waves_16k):
            ori_waves_16k_input_list = [waves_16k[bib].cpu().numpy() for bib in range(len(waves_16k))]
            ori_inputs = hubert_feature_extractor(ori_waves_16k_input_list, return_tensors="pt", return_attention_mask=True, padding=True, sampling_rate=16000).to(device)
            with torch.no_grad():
                ori_outputs = hubert_model(
                    ori_inputs.input_values.half(),
                )
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori

    elif speech_tokenizer_type == "xlsr":
        from transformers import (
            Wav2Vec2FeatureExtractor,
            Wav2Vec2Model,
        )

        model_name = config["model_params"]["speech_tokenizer"]["name"]
        output_layer = config["model_params"]["speech_tokenizer"]["output_layer"]
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
        wav2vec_model.encoder.layers = wav2vec_model.encoder.layers[:output_layer]
        wav2vec_model = wav2vec_model.to(device)
        wav2vec_model = wav2vec_model.eval()
        wav2vec_model = wav2vec_model.half()

        def semantic_fn(waves_16k):
            ori_waves_16k_input_list = [waves_16k[bib].cpu().numpy() for bib in range(len(waves_16k))]
            ori_inputs = wav2vec_feature_extractor(ori_waves_16k_input_list, return_tensors="pt", return_attention_mask=True, padding=True, sampling_rate=16000).to(device)
            with torch.no_grad():
                ori_outputs = wav2vec_model(
                    ori_inputs.input_values.half(),
                )
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori

    else:
        raise ValueError(f"Unknown speech tokenizer type: {speech_tokenizer_type}")
    # Generate mel spectrograms
    mel_fn_args = {
        "n_fft": config["preprocess_params"]["spect_params"]["n_fft"], 
        "win_size": config["preprocess_params"]["spect_params"]["win_length"], 
        "hop_size": config["preprocess_params"]["spect_params"]["hop_length"], 
        "num_mels": config["preprocess_params"]["spect_params"]["n_mels"], 
        "sampling_rate": sr, 
        "fmin": config["preprocess_params"]["spect_params"].get("fmin", 0), 
        "fmax": None if config["preprocess_params"]["spect_params"].get("fmax", "None") == "None" else 8000, 
        "center": False,
    }
    from modules.audio import mel_spectrogram

    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

    builtins.print = original_print
    return (
        model,
        semantic_fn,
        f0_fn,
        vocoder_fn,
        campplus_model,
        to_mel,
        mel_fn_args,
    )


def adjust_f0_semitones(f0_sequence, n_semitones):
    factor = 2 ** (n_semitones / 12)
    return f0_sequence * factor


def crossfade(chunk1, chunk2, overlap):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    if len(chunk2) < overlap:
        chunk2[:overlap] = chunk2[:overlap] * fade_in[: len(chunk2)] + (chunk1[-overlap:] * fade_out)[: len(chunk2)]
    else:
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


class SeedVCPipeline(Pipeline):
    def __init__(self, slot_info: SlotInfoMember):
        assert isinstance(slot_info, SeedVCSlotInfo)
        self.slot_info = slot_info
        self.slot_index = self.slot_info.slot_index

        conf = ConfigurationManager.get_instance().get_server_configuration()
        super().__init__(gpu_device_id=conf.gpu_device_id_int)
        logging.getLogger(LOGGER_NAME).info(f"construct new seed-vc pipeline: slot_index:{self.slot_index}, gpu_device_id:{self.gpu_device_id}")

        if self.slot_info.repository_id is None:
            checkpoint_path = Path(ModelDir, str(self.slot_index), self.slot_info.seed_vc_model_file)
            config_path = Path(ModelDir, str(self.slot_index), self.slot_info.seed_vc_config_file)
        else:
            checkpoint_path = self.slot_info.seed_vc_model_file
            config_path = self.slot_info.seed_vc_config_file
            

        models = load_models(
            repositry_id=self.slot_info.repository_id,
            f0_condition=self.slot_info.f0_condition,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            fp16=False,
        )
        self.model = models[0]
        self.semantic_fn = models[1]
        self.f0_fn = models[2]
        self.vocoder_fn = models[3]
        self.campplus_model = models[4]
        self.mel_fn = models[5]
        self.mel_fn_args = models[6]

        self.sr = self.mel_fn_args["sampling_rate"]

    def force_stop(self):
        pass

    @torch.no_grad()
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
        assert src_wav_path is not None or source_audio is not None
        assert src_wav_path is None or source_audio is None

        if src_wav_path is not None:
            source_audio = librosa.load(src_wav_path, sr=self.sr)[0]
        ref_audio = librosa.load(ref_wav_path, sr=self.sr)[0]

        sr = 22050 if not self.slot_info.f0_condition else 44100
        hop_length = 256 if not self.slot_info.f0_condition else 512
        max_context_window = sr // hop_length * 30
        overlap_frame_len = 16
        overlap_wave_len = overlap_frame_len * hop_length

        # Process audio
        source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
        ref_audio = torch.tensor(ref_audio[: sr * 25]).unsqueeze(0).float().to(device)

        time_vc_start = time.time()
        # Resample
        converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
        # if source audio less than 30 seconds, whisper can handle in one forward
        if converted_waves_16k.size(-1) <= 16000 * 30:
            S_alt = self.semantic_fn(converted_waves_16k)
        else:
            overlapping_time = 5  # 5 seconds
            S_alt_list = []
            buffer = None
            traversed_time = 0
            while traversed_time < converted_waves_16k.size(-1):
                if buffer is None:  # first chunk
                    chunk = converted_waves_16k[:, traversed_time : traversed_time + 16000 * 30]
                else:
                    chunk = torch.cat([buffer, converted_waves_16k[:, traversed_time : traversed_time + 16000 * (30 - overlapping_time)]], dim=-1)
                S_alt = self.semantic_fn(chunk)
                if traversed_time == 0:
                    S_alt_list.append(S_alt)
                else:
                    S_alt_list.append(S_alt[:, 50 * overlapping_time :])
                buffer = chunk[:, -16000 * overlapping_time :]
                traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
            S_alt = torch.cat(S_alt_list, dim=1)

        ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
        S_ori = self.semantic_fn(ori_waves_16k)

        mel = self.mel_fn(source_audio.to(device).float())
        mel2 = self.mel_fn(ref_audio.to(device).float())

        target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

        feat2 = torchaudio.compliance.kaldi.fbank(ori_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = self.campplus_model(feat2.unsqueeze(0))

        if self.slot_info.f0_condition:
            F0_ori = self.f0_fn(ori_waves_16k[0], thred=0.03) # ref audio
            F0_alt = self.f0_fn(converted_waves_16k[0], thred=0.03) # input audio
            # F0_ori_avrage = F0_ori[F0_ori > 1].mean()
            # F0_alt_avrage = F0_alt[F0_alt > 1].mean()
            # print(color_text(f"Ref voice f0 avr: {F0_ori_avrage}, Input voice f0 avr: {F0_alt_avrage}", color="CYAN"))

            F0_ori = torch.from_numpy(F0_ori).to(device)[None]
            F0_alt = torch.from_numpy(F0_alt).to(device)[None]

            voiced_F0_ori = F0_ori[F0_ori > 1]
            voiced_F0_alt = F0_alt[F0_alt > 1]

            log_f0_alt = torch.log(F0_alt + 1e-5)
            voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
            voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
            median_log_f0_ori = torch.median(voiced_log_f0_ori)
            median_log_f0_alt = torch.median(voiced_log_f0_alt)
            # print(color_text(f"Ref voice f0 median: {median_log_f0_ori}, Input voice f0 median: {median_log_f0_alt}", color="CYAN"))
            

            # shift alt log f0 level to ori log f0 level
            shifted_log_f0_alt = log_f0_alt.clone()
            if auto_f0_adjust:
                shifted_log_f0_alt[F0_alt > 1] = log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
            shifted_f0_alt = torch.exp(shifted_log_f0_alt)
            if semi_tone_shift != 0:
                # print(color_text(f"Shift f0 {semi_tone_shift} semitones", color="CYAN"))
                shifted_f0_alt[F0_alt > 1] = adjust_f0_semitones(shifted_f0_alt[F0_alt > 1], semi_tone_shift)
        else:
            F0_ori = None
            F0_alt = None
            shifted_f0_alt = None

        # Length regulation
        cond, _, codes, commitment_loss, codebook_loss = self.model.length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt)
        prompt_condition, _, codes, commitment_loss, codebook_loss = self.model.length_regulator(S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori)

        max_source_window = max_context_window - mel2.size(2)
        # split source condition (cond) into chunks
        processed_frames = 0
        generated_wave_chunks = []
        # generate chunk by chunk and stream the output
        while processed_frames < cond.size(1):
            chunk_cond = cond[:, processed_frames : processed_frames + max_source_window]
            is_last_chunk = processed_frames + max_source_window >= cond.size(1)
            cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
            with torch.autocast(device_type=device.type, dtype=torch.float16 if self.slot_info.fp16 else torch.float32):
                # Voice Conversion
                vc_target = self.model.cfm.inference(cat_condition, torch.LongTensor([cat_condition.size(1)]).to(mel2.device), mel2, style2, None, diffusion_steps, inference_cfg_rate=inference_cfg_rate)
                vc_target = vc_target[:, :, mel2.size(-1) :]
            vc_wave = self.vocoder_fn(vc_target.float()).squeeze()
            vc_wave = vc_wave[None, :]
            if processed_frames == 0:
                if is_last_chunk:
                    output_wave = vc_wave[0].cpu().numpy()
                    generated_wave_chunks.append(output_wave)
                    break
                output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                previous_chunk = vc_wave[0, -overlap_wave_len:]
                processed_frames += vc_target.size(2) - overlap_frame_len
            elif is_last_chunk:
                output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len)
                generated_wave_chunks.append(output_wave)
                processed_frames += vc_target.size(2) - overlap_frame_len
                break
            else:
                output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len)
                generated_wave_chunks.append(output_wave)
                previous_chunk = vc_wave[0, -overlap_wave_len:]
                processed_frames += vc_target.size(2) - overlap_frame_len
        vc_wave = torch.tensor(np.concatenate(generated_wave_chunks))[None, :].float()
        time_vc_end = time.time()
        print(f"RTF: {(time_vc_end - time_vc_start) / vc_wave.size(-1) * sr}")

        return sr, vc_wave.cpu().numpy()
