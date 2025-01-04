
import os
import glob
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import logging
from tqdm import tqdm
import random
import numpy as np
import torch
import librosa
import ffmpeg
import folder_paths
from time import time as ttime

from functions.text_replacer import TextReplacer
import config

def get_device():
    """
    返回当前可用的设备（CUDA 或 CPU）。
    
    如果 CUDA 可用，则返回 CUDA 设备；否则，返回 CPU 设备。

    返回:
    torch.device: 当前可用的设备对象。
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_all_random_seed(seed):
    """
    设置所有随机数生成器的种子，以确保实验的可重复性。
    
    参数:
    seed (int): 用于随机数生成器的种子值。
    """
    
    # 设置 Python 内置的 random 模块的随机种子
    random.seed(seed)
    
    # 设置 NumPy 的随机种子
    np.random.seed(seed)
    
    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed)
    
    # 如果使用多个 GPU，设置所有 GPU 的随机种子
    torch.cuda.manual_seed_all(seed)


def generate_audio(output, t0, speed, target_sr=config.AUDIO_TARGET_SAMPLE_RATE):
    """
    生成音频，并调整播放速度。

    参数:
    output (list): 输出的音频数据列表，每个元素是一个包含 'tts_speech' 键的字典。
    t0 (float): 开始时间，用于计算生成音频的耗时。
    speed (float): 播放速度，可以是大于1.0或小于1.0的值。
    target_sr (int): 目标采样率，默认为22050。

    返回:
    dict: 包含生成的波形数据和采样率的字典。
    """
    output_list = []
    
    for out_dict in output:
        # 将 'tts_speech' 数据从张量转换为 NumPy 数组，并放大到 16-bit PCM 范围
        output_numpy = out_dict['tts_speech'].squeeze(0).numpy() * 32768
        output_numpy = output_numpy.astype(np.int16)
        
        # 如果播放速度不为1.0，则调整音频速度
        if speed > 1.0 or speed < 1.0:
            output_numpy = speed_change(output_numpy, speed, target_sr)
        
        # 将 NumPy 数组转换回张量，并标准化回原范围
        output_list.append(torch.Tensor(output_numpy / 32768).unsqueeze(0))
    
    # 计算音频生成的耗时
    t1 = ttime()
    print("cost time \t %.3f" % (t1 - t0))
    
    # 返回生成的波形数据和采样率
    return {"waveform": torch.cat(output_list, dim=1).unsqueeze(0), "sample_rate": target_sr}

def postprocess(speech, top_db=60, hop_length=220, win_length=440, max_val=0.8, target_sr=config.AUDIO_TARGET_SAMPLE_RATE):
    """
    对音频数据进行后处理，包括修剪静音部分、归一化和填充操作。

    参数:
    speech (torch.Tensor): 输入的音频数据。
    top_db (int): 修剪音频的dB阈值，默认为60。
    hop_length (int): 修剪操作的步长，默认为220。
    win_length (int): 修剪操作的窗口长度，默认为440。
    max_val (float): 最大振幅值，用于归一化，默认为0.8。
    target_sr (int): 目标采样率，默认为22050。

    返回:
    torch.Tensor: 处理后的音频数据。
    """
    
    # 修剪静音部分，使用librosa的trim方法
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    
    # 归一化操作，如果音频的最大振幅超过max_val，则进行归一化
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    
    # 填充操作，向音频末尾添加0.2秒的静音
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    
    return speech


def speed_change(input_audio, speed, sr):
    """
    调整音频的播放速度。

    参数:
    input_audio (numpy.ndarray): 输入的音频数据，必须是 np.int16 类型。
    speed (float): 播放速度，可以是大于1.0或小于1.0的值。
    sr (int): 音频的采样率。

    返回:
    numpy.ndarray: 处理后的音频数据，数据类型为 np.int16。
    """

    # 检查输入数据类型和声道数
    if input_audio.dtype != np.int16:
        raise ValueError("输入音频数据类型必须为 np.int16")
    
    # 转换为字节流
    raw_audio = input_audio.astype(np.int16).tobytes()

    # 设置 ffmpeg 输入流
    input_stream = ffmpeg.input('pipe:', format='s16le', acodec='pcm_s16le', ar=str(sr), ac=1)

    # 变速处理
    output_stream = input_stream.filter('atempo', speed)

    # 输出流到管道
    out, _ = (
        output_stream.output('pipe:', format='s16le', acodec='pcm_s16le').run(input=raw_audio, capture_stdout=True, capture_stderr=True)
    )

    # 将管道输出解码为 NumPy 数组
    processed_audio = np.frombuffer(out, np.int16)

    return processed_audio

def get_speaker_folders():
    """
    获取说话人模型文件夹列表。

    返回:
    list: 说话人模型文件夹列表。
    """
    return os.path.join(config.COSYVOICE_MODEL_DIR, "Speaker")


def list_model_files(directory, extension=".pt"):
    """
    列出指定目录中所有后缀为 .pt 的模型文件，并去掉文件后缀。

    参数:
    directory (str): 要扫描的目录路径。
    extension (str): 文件后缀，默认为 ".pt"。

    返回:
    list: 去掉后缀的模型文件列表。
    """
    pattern = f"{directory}/**/*{extension}"
    files = glob.glob(pattern, recursive=True)
    model_files = [os.path.splitext(os.path.basename(file))[0] for file in files]
    return model_files


def replace_tts_text(tts_text, file_name="多音字纠正配置.txt"):
    replacement_file= os.path.join(config.COSYVOICE_NODE_DIR, file_name)
    replacer = TextReplacer(tts_text, replacement_file)
    result_string = replacer.result_string
    return result_string