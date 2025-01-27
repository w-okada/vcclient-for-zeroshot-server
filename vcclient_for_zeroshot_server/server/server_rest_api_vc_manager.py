import io
import os
import tempfile
import uuid
import wave
from fastapi import APIRouter, Form
from fastapi.responses import StreamingResponse
from vcclient_for_zeroshot_server.server.validation_error_logging_route import ValidationErrorLoggingRoute
from vcclient_for_zeroshot_server.core.data_types.vc_manager_data_types import ConvertVoiceParam
from vcclient_for_zeroshot_server.core.vc_manager.vc_manager import VCManager
import numpy as np
from scipy.io import wavfile
from fastapi import File, UploadFile, Header
import base64

class RestAPIVCManager:
    def __init__(self):
        self.router = APIRouter()
        self.router.route_class = ValidationErrorLoggingRoute

        self.router.add_api_route("/api/vc-manager/operation/convertVoice", self.post_convert_voice, methods=["POST"])

        self.router.add_api_route("/api_vc-manager_operation_convertVoice", self.post_convert_voice, methods=["POST"])

    async def post_convert_voice(self, waveform: UploadFile = File(...), data_type: str = Form(default="bin"), x_timestamp: int | None = Header(default=0)):
        chunk = await waveform.read()
        # print(waveform)
        # print(data_type)
        if data_type == "file":
            audio_buffer = io.BytesIO()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(chunk)
                temp_path = temp_file.name
                try:
                    sr, data = VCManager.get_instance().run(waveform=None, wave_path=temp_path)
                    temp_file.close()
                    os.unlink(temp_path)
                    data_clipped = np.clip(data, -1.0, 1.0)
                    data_reshaped = data_clipped.reshape(-1)
                    wavfile.write(audio_buffer, sr, data_reshaped)
                except Exception as e:
                    # エラーが発生した場合も一時ファイルを確実に削除
                    print("Exception:", e)
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            return StreamingResponse(audio_buffer, media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=output.wav"})

        elif data_type == "bin":
            # ChromeのデフォルトのContextの設定(48kHz, 32bit float)で入ってくる想定
            try:
                chunk_np: np.ndarray = np.frombuffer(chunk, dtype=np.float32)
                VCManager.get_instance().put_chunk_to_convert_queue(chunk_np)
                converted_chunks:list[np.ndarray] = []

                while True:
                    converted = VCManager.get_instance().get_chunk_from_converted_queue()
                    if converted is None:
                        break
                    converted_chunks.append(converted)


                # マルチパートレスポンスを生成するジェネレータ
                boundary = uuid.uuid4().hex
                async def generate():
                    for idx, chunk in enumerate(converted_chunks):
                        isVoice, converted = chunk
                        return_bytes = converted.tobytes()
                        encoded_data = base64.b64encode(return_bytes).decode('ascii')
                        # print("return bytes length::", len(return_bytes))
                        # print("return bytes::", return_bytes)
                        
                        yield f"--{boundary}\r\n"
                        yield f"Content-Type: application/octet-stream\r\n"
                        yield f"Content-Disposition: attachment; filename=chunk_{idx}.bin\r\n"
                        yield f"X-Is-Voice: {isVoice}\r\n\r\n"
                        yield encoded_data
                        yield "\r\n"
                    yield f"--{boundary}--\r\n"

                # StreamingResponseでマルチパートデータを返す
                return StreamingResponse(generate(), media_type=f"multipart/form-data; boundary={boundary}")                    
                
            except Exception as e:
                print("Exception:", e)


        return StreamingResponse(audio_buffer, media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=output.wav"})
        # return Response(
        #     content=converted_np.tobytes(),
        #     media_type="application/octet-stream",
        #     headers={
        #         "x-timestamp": str(x_timestamp),
        #         "x-performance": performance_data.model_dump_json() if performance_data is not None else "{}",
        #     },
        # )

        # print(convert_voice_param)
        # sr, data = VCManager.get_instance().run(convert_voice_param)

        # data_clipped = np.clip(data, -1.0, 1.0)
        # data_reshaped = data_clipped.reshape(-1)
        # audio_buffer = io.BytesIO()
        # wavfile.write(audio_buffer, sr, data_reshaped)
        # return StreamingResponse(audio_buffer, media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=output.wav"})
