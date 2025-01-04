import os
import time
from tqdm import tqdm

import config
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import logging

class CosyVoicePatches(CosyVoice):
    def __init__(self, model_dir):
        super().__init__(model_dir)

    def inference_sft_with_speaker_model(self, tts_text, speaker_model, stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True)):
            model_input = self.__frontend_sft(i, speaker_model)

            start_time = time.time()
            logging.info("\nsynthesis text {}".format(i))

            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / config.AUDIO_TARGET_SAMPLE_RATE
                logging.info("\nyield speech len {}, rtf {}".format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()
    
    def __frontend_sft(self, tts_text, speaker_model):
        tts_text_token, tts_text_token_len = self.frontend._extract_text_token(tts_text)
        model_input = {
            'text': tts_text_token,
            'text_len': tts_text_token_len,
            'llm_embedding': speaker_model["llm_embedding"],
            'flow_embedding': speaker_model["flow_embedding"]
            }
        return model_input