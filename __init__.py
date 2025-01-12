
from .nodes.cosyvoice_nodes import *
# from .nodes.speaker_nodes import *

NODE_CONFIG = {
    "NCECosyVoiceSFT": {
        "class": NCECosyVoiceSFT,
        "name": "🎙️ CosyVoice 预训练音色"
    },
    "NCECosyVoiceZeroShot": {
        "class": NCECosyVoiceZeroShot,
        "name": "🎙️ CosyVoice 3秒音色克隆"
    },
    "NCECosyVoiceSaveSpeakerModel": {
        "class": NCECosyVoiceSaveSpeakerModel,
        "name": "🎙️ CosyVoice 保存说话人模型"
    },
    "NCECosyVoiceLoadSpeakerModel": {
        "class": NCECosyVoiceLoadSpeakerModel,
        "name": "🎙️ CosyVoice 加载说话人模型"
    },
    "NCECosyVoiceCrossLingual": {
        "class": NCECosyVoiceCrossLingual,
        "name": "🎙️ CosyVoice 跨语言克隆"
    },
}

def generate_node_mappings(node_config):
    node_class_mappings = {}
    node_display_name_mappings = {}

    for node_name, node_info in node_config.items():
        # print(f"node_name: {node_name}, node_info: {node_info}")
        node_class_mappings[node_name] = node_info["class"]
        node_display_name_mappings[node_name] = node_info.get("name", node_info["class"].__name__)

    return node_class_mappings, node_display_name_mappings

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_node_mappings(NODE_CONFIG)
WEB_DIRECTORY = "./web"


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print('🐍 NCE CosyVoice Loaded')
