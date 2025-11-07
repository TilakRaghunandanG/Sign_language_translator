from gtts import gTTS
import tempfile, os
from googletrans import Translator

def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    tts.save(tmp.name)
    return tmp.name

def translate_text(text, dest_lang='en'):
    trans = Translator()
    out = trans.translate(text, dest=dest_lang)
    return out.text
