import logging
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
from mcelery.cos import download_cos_file, get_local_path, upload_cos_file
from mcelery.infer import celery_app, register_infer_tasks

logging.getLogger("matplotlib").setLevel(logging.WARNING)
sys.path.append(str(Path("third_party/Matcha-TTS").resolve()))

cosyvoice = CosyVoice("pretrained_models/CosyVoice-300M")
prompt_sr, target_sr = 16000, 22050


def postprocess(speech: torch.Tensor, top_db=60, hop_length=220, win_length=440, max_val=0.8) -> torch.Tensor:
    speech, _ = librosa.effects.trim(speech, top_db=top_db, frame_length=win_length, hop_length=hop_length)
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech


def set_all_random_seed():
    seed = random.randint(1, 100000000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@celery_app.task(lazy=False, name="cosy_infer", queue="cosy_infer", autoretry_for=(Exception,), default_retry_delay=10)
def cosy_infer_task(
    text: str,
    prompt_text_cos: str,
    prompt_wav_cos: str,
    output_cos: str,
    mode: int = 1,
) -> str:
    """
    COSY TTS 服务
    :param text: 音频文字内容
    :param prompt_text_cos: 参考文本 COS key, 据说是训练 rvc 的时候放入
    :param prompt_wav_cos: 参考音频 COS key
    :param output_cos: 合成的音频文件 COS key
    :param mode: 模式： 1 中文[同语言克隆] 2 中日英混合[跨语言克隆]
    :return: output_cos
    """
    if text == "":
        raise Exception("输出路径不能为空")
    if prompt_text_cos == "":
        raise Exception("参考文本不能为空")
    if prompt_wav_cos == "":
        raise Exception("参考音频不能为空")

    prompt_text_path = download_cos_file(prompt_text_cos)
    with prompt_text_path.open("r", encoding="utf-8") as f:
        prompt_text = f.read()
    if prompt_text == "":
        raise Exception("参考文本不能为空")

    prompt_wav_path = download_cos_file(prompt_wav_cos)
    if torchaudio.info(prompt_wav_path).sample_rate < prompt_sr:
        # TODO 自动转换采样率
        raise Exception("采样率低于 16000，请先转换采样率")

    prompt_speech_16k = postprocess(load_wav(prompt_wav_path, prompt_sr))
    set_all_random_seed()
    if mode == 1:
        logging.info("get zero_shot inference request")
        output = cosyvoice.inference_zero_shot(text, prompt_text, prompt_speech_16k)
    elif mode == 2:
        logging.info("get cross_lingual inference request")
        output = cosyvoice.inference_cross_lingual(text, prompt_speech_16k)
    else:
        raise Exception("模式只能是 1 或者 2")

    tts_speeches = torch.concat([o["tts_speech"] for o in output], dim=1)

    output_path = get_local_path(output_cos)
    torchaudio.save(output_path, tts_speeches, sample_rate=target_sr)
    upload_cos_file(output_cos)
    return output_cos


register_infer_tasks()
