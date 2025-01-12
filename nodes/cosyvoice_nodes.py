import os
import torch
import torchaudio
import numpy as np
import ffmpeg
import folder_paths

import config

from time import time as ttime

from cosyvoice.utils.common import set_all_random_seed

from functions.download_models import download_cosyvoice_300m
from functions.cosyvoice_patches import CosyVoicePatches as CosyVoice
from functions.utils import postprocess, get_device, generate_audio, get_speaker_folders, list_model_files, replace_tts_text
from functions.text_replacer import TextReplacer

# NCE 预训练音色
class NCECosyVoiceSFT:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "speaker":(config.SFT_SPEAKER_LIST,{
                    "default":"中文女"
                }),
                "speed":("FLOAT",{
                    "default": 1.0
                }),
                "seed":("INT",{
                    "default": 42
                }),
                "stream": ("BOOLEAN",{
                    "default": False
                }),
                 "use_25hz":("BOOLEAN",{
                    "default": False
                }),
                "polyreplace":("BOOLEAN",{
                    "default": False
                }),
            },
            "optional":{
                "tts_text":("STRING",),
            }
        }
    
    CATEGORY = config.CATEGORY_NAME
    RETURN_TYPES = ("AUDIO",)
    FUNCTION="generate"

    def generate(self, tts_text, speed, speaker, seed, use_25hz, stream, polyreplace=False):
        t0 = ttime()
        _, model_dir = download_cosyvoice_300m(use_25hz)

        assert len(tts_text) > 0, "文本(tts_text)不能为空"
        if polyreplace:
            # 多音节替换
            print("You have enabled polyphonic word replacement.")
            tts_text = replace_tts_text(tts_text)
            
        cosyvoice = CosyVoice(model_dir)
        set_all_random_seed(seed)

        print('get inference_sft inference request')
        output = cosyvoice.inference_sft(tts_text=tts_text, spk_id=speaker, stream=stream)
        audio= generate_audio(output,t0,speed)

        return (audio,)

# NCE CosyVoice 跨语言克隆
class NCECosyVoiceCrossLingual:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "prompt_wav": ("AUDIO",),
                "speed":("FLOAT",{
                    "default": 1.0
                }),
                "seed":("INT",{
                    "default": 42
                }),
                "use_25hz":("BOOLEAN",{
                    "default": False
                }),
            },
            "optional":{
                "tts_text":("STRING",),
            }
        }
    
    CATEGORY = config.CATEGORY_NAME
    RETURN_TYPES = ("AUDIO",)
    FUNCTION="generate"

    def generate(self, tts_text, prompt_wav, speed, seed, use_25hz, polyreplace=False):
        t0 = ttime()
        _, model_dir = download_cosyvoice_300m(use_25hz)

        assert len(tts_text) > 0, "文本(tts_text)不能为空"
        assert prompt_wav is not None, " 参考音频(prompt_wav)不能为空"

        if polyreplace:
            # 多音节替换
            print("You have enabled polyphonic word replacement.")
            tts_text = replace_tts_text(tts_text)

        waveform = prompt_wav['waveform'].squeeze(0)
        source_sr = prompt_wav['sample_rate']
        speech = waveform.mean(dim=0,keepdim=True)
        if source_sr != config.AUDIO_PROMPT_SAMPLE_RATE:
            speech = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=config.AUDIO_PROMPT_SAMPLE_RATE)(speech)

        print('get inference_cross_lingual inference request')
        prompt_speech_16k = postprocess(speech)
        set_all_random_seed(seed)

        cosyvoice = CosyVoice(model_dir)
        output = cosyvoice.inference_cross_lingual(tts_text=tts_text, prompt_speech_16k=prompt_speech_16k, stream=False, speed=speed)
        audio= generate_audio(output,t0,speed)
        
        return (audio,)



# NCE 3秒音色克隆
class NCECosyVoiceZeroShot:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "tts_text":("STRING", {
                    "default": "",
                }),
                "speed":("FLOAT",{
                    "default": 1.0
                }),
                "seed":("INT",{
                    "default": 42
                }),
                "stream": ("BOOLEAN",{
                    "default": False
                }),
                 "use_25hz":("BOOLEAN",{
                    "default": False
                }),
                "polyreplace":("BOOLEAN",{
                    "default": False
                }),
            },
            "optional":{
                "prompt_text":("STRING",),
                "prompt_wav": ("AUDIO",),
                "speaker_model":("SPEAKER_MODEL",),
            }
        }
    
    CATEGORY = config.CATEGORY_NAME
    
    RETURN_TYPES = ("AUDIO", "SPEAKER_MODEL", )
    FUNCTION="generate"

    def generate(self, tts_text, speed, seed, stream, use_25hz, prompt_text=None, prompt_wav=None, speaker_model=None, polyreplace=False):
        t0 = ttime()
        _, model_dir = download_cosyvoice_300m(use_25hz)

        assert len(tts_text) > 0, "文本(tts_text)不能为空"
        if polyreplace:
            # 多音节替换
            print("You have enabled polyphonic word replacement.")
            tts_text = replace_tts_text(tts_text)

        cosyvoice = CosyVoice(model_dir)
        __spk_model = None        

        if speaker_model is None:
            assert len(prompt_text) > 0, "参考音频文本(prompt)不能为空"
            assert prompt_wav is not None, " 参考音频(prompt_wav)不能为空"

            waveform = prompt_wav['waveform'].squeeze(0)
            source_sr = prompt_wav['sample_rate']
            speech = waveform.mean(dim=0,keepdim=True)
            if source_sr != config.AUDIO_PROMPT_SAMPLE_RATE:
                speech = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=config.AUDIO_PROMPT_SAMPLE_RATE)(speech)

            print('get zero_shot inference request')
            prompt_speech_16k = postprocess(speech)
            set_all_random_seed(seed)

            output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream, speed)
            audio= generate_audio(output,t0,speed)
            __spk_model = cosyvoice.frontend.frontend_zero_shot(tts_text, prompt_text, prompt_speech_16k)
            del __spk_model['text']
            del __spk_model['text_len']

        else:
            print('get zero_shot_with_speaker_model inference request')
            set_all_random_seed(seed)
            output = cosyvoice.inference_sft_with_speaker_model(tts_text, speaker_model, stream, speed)
            audio= generate_audio(output,t0,speed)

            __spk_model = speaker_model
        return (audio, __spk_model, )


# NCE 保存说话人模型
class NCECosyVoiceSaveSpeakerModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "spk_model":("SPEAKER_MODEL",),
                "speaker_name":("STRING", {
                    "default": ""
                }),
            }
        }
    
    CATEGORY = config.CATEGORY_NAME
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION="generate"

    def generate(self, spk_model, speaker_name):
        speaker_model_dir = get_speaker_folders()

        # 判断目录是否存在，不存在则创建
        print(f"saving speaker model {speaker_name} to {speaker_model_dir}")
        if not os.path.exists(speaker_model_dir):
            os.makedirs(speaker_model_dir)
        
        # 保存模型
        speaker_full_name= speaker_name + ".pt"
        torch.save(spk_model, os.path.join(speaker_model_dir, speaker_full_name))
        return speaker_full_name

# NCE 加载说话人模型
class NCECosyVoiceLoadSpeakerModel:
    @classmethod
    def INPUT_TYPES(s):
        speaker_models = list_model_files(get_speaker_folders())
        return {
            "required":{
                "speaker_model":(speaker_models,{
                    "default": speaker_models[0]
                }),
            }
        }
    
    CATEGORY = config.CATEGORY_NAME

    RETURN_TYPES = ("SPEAKER_MODEL",)
    FUNCTION="generate"

    def generate(self, speaker_model):
        # print(f"loading speaker model {speaker_model}")
        full_speaker_model_path = os.path.join(get_speaker_folders(), speaker_model + ".pt")
        assert os.path.exists(full_speaker_model_path), "Speaker model is not exist"

        SPEAKER_MODEL = torch.load(full_speaker_model_path, map_location= get_device())

        return (SPEAKER_MODEL, )

