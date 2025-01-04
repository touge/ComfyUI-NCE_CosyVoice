
import config

# 多行文本输入
class NCECosyVoiceMultiText:
  @classmethod
  def INPUT_TYPES(s):
    return {"required": {"text": ("STRING", {
       "multiline": True, 
       "dynamicPrompts": True
      })}}
  RETURN_TYPES = ("TEXT",)
  FUNCTION = "generate"

  CATEGORY = config.CATEGORY_NAME

  def generate(self,text):
    return (text, )

