import re
import textwrap
from pathlib import Path

try:
    from transformers import pipeline
    _HAS_TRANSFORMERS = True
except:
    _HAS_TRANSFORMERS = False

SUMMARIZER_MODEL = "facebook/bart-large-cnn"
CAPTION_MODEL = "facebook/bart-base"

summarizer = pipeline("summarization", model=SUMMARIZER_MODEL) if _HAS_TRANSFORMERS else None
caption_gen = pipeline("text2text-generation", model=CAPTION_MODEL) if _HAS_TRANSFORMERS else None

def build_news_script(headline):
    short = headline
    if summarizer:
        try:
            res = summarizer(headline, max_length=120, min_length=20, do_sample=False)
            if res and isinstance(res, list):
                short = res[0].get("summary_text") or short
        except: pass

    expanded = None
    if caption_gen:
        try:
            prompt = f"Expand into 180-260 words: {short}"
            res = caption_gen(prompt, max_length=320, num_return_sequences=1)
            if res and isinstance(res, list):
                expanded = res[0].get("generated_text") or res[0].get("text")
        except: pass

    if expanded and len(expanded.split()) >= 120:
        return re.sub(r'\s+', ' ', expanded).strip()
    return f"{short}. In brief: {short}. For context: {short}. This is a brief report."

def generate_caption_and_hashtags(headline, script):
    clean = re.sub(r'\s+[-|]\s*[^-|\n]{1,80}$', '', headline).strip()
    caption = clean if len(clean)<120 else clean[:117]+"..."
    words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', clean)]
    uniq=[]
    for w in words:
        if w not in uniq: uniq.append(w)
        if len(uniq)>=3: break
    if not uniq: uniq = clean.split()[:3]
    tags = " ".join("#"+w for w in uniq[:3])
    return f"{caption}\n\n{tags}"
