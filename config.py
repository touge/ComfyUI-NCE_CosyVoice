
import os, folder_paths

# config.py
'''插件模型目录'''
COSYVOICE_NODE_DIR        = os.path.dirname(os.path.abspath(__file__))

'''插件分类'''
CATEGORY_NAME             = "🐍 NCE/CosyVoice"

'''发言人文件，可能废除'''
SPK2INFO_FILE             = "spk2info.pt"

'''克隆音色模型目录'''
SPEAKER_MODEL_DIR         = "Speaker"

'''预训练音色列表'''
SFT_SPEAKER_LIST          = ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']

'''插件模型目录'''
COSYVOICE_MODEL_DIR       = os.path.join(folder_paths.models_dir, "CosyVoice")

'''输入音频采样率'''
AUDIO_PROMPT_SAMPLE_RATE  = 16000

'''目标音频采样率'''
AUDIO_TARGET_SAMPLE_RATE  = 22050