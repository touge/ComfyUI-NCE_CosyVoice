本仓库技术底层来自：https://github.com/FunAudioLLM/CosyVoice

# 系统安装

1. 首先克隆本仓库：
```
git clone https://github.com/touge/ComfyUI-NCE_CosyVoice
cd ComfyUI-NCE_CosyVoice
```

2. 安装第三方依赖库：
-> 参见 third_party 目录中的README.md 文档

4. 安装pynini，因为pynini在pip模式下容易出错，所以采用conda安装
```
conda install -y -c conda-forge pynini==2.1.6
```
4. 安装NCE_CosyVoice依赖包，在 ComfyUI-NCE_CosyVoice 中，安装依赖项目：
```
pip install -r requirements.txt
```

# ComfyUI-NCE_CosyVoice

**模型下载**

模型请下载到ComfyUI模型目录的CosyVoice子目录中。

使用git下载，请确保已安装git lfs。

mkdir -p /你的comfyui安装目录/models/CosyVoice

```
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git CosyVoice/CosyVoice2-0.5B
git clone https://www.modelscope.cn/iic/CosyVoice-300M.git CosyVoice/CosyVoice-300M
git clone https://www.modelscope.cn/iic/CosyVoice-300M-25Hz.git CosyVoice/CosyVoice-300M-25Hz
git clone https://www.modelscope.cn/iic/CosyVoice-300M-SFT.git CosyVoice/CosyVoice-300M-SFT
git clone https://www.modelscope.cn/iic/CosyVoice-300M-Instruct.git CosyVoice/CosyVoice-300M-Instruct
git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git CosyVoice/CosyVoice-ttsfrd
```

# 出错解决

**1. window平台下pypinyin通过pip install 失败**

删掉 requirements.txt中的pypinyin，然后使用conda安装

```
conda install -y -c conda-forge pynini==2.1.6
```
