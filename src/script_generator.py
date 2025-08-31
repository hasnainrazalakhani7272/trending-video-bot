import re, textwrap
from transformers import pipeline

_HAS_TRANSFORMERS = True
SUMMARIZER_MODEL = "facebook/bart-large-cnn"
CAPTION_MODEL = "facebook/bart-base"

summarizer = caption_gen = None
try:
    summarizer = pipeline("summarization", model=SUMMARIZER_MODEL)
    caption_gen = pipeline("text2text-generation", model=CAPTION_MODEL)
except: pass

def build_news_script(headline: str) -> str:
    short = headline
    try:
        if summarizer:
            res = summarizer(headline, max_length=120, min_length=20, do_sample=False)
            short = res[0].get("summary_text", short)
    except: pass
    expanded = None
    try:
        if caption_gen:
            prompt = f"Expand this into 180-260 words news report:\n\nSummary: {short}"
            res = caption_gen(prompt, max_length=320, num_return_sequences=1)
            expanded = res[0].get("generated_text") or res[0].get("text")
    except: pass
    if expanded and len(expanded.split()) >= 120: return re.sub(r'\s+', ' ', expanded).strip()
    return f"{short}. In brief: {short}. For context: {short}. This outlines main facts."

def generate_caption_and_hashtags(headline: str, script: str) -> str:
    clean = re.sub(r'\s+[-|]\s*[^-|\n]{1,80}$', '', headline).strip()
    try:
        if caption_gen:
            prompt = f"Write ONE short caption, then 3 hashtags:\nHeadline: {clean}\n\n{script}"
            res = caption_gen(prompt, max_length=80, num_return_sequences=1)
            txt = res[0].get("generated_text") or res[0].get("text")
            if txt:
                caps = txt.strip().split("\n")
                caption = caps[0]
                tags = re.findall(r'#\w+', txt)[:3] or ["#" + w.lower() for w in re.findall(r'\b\w{3,}\b', clean)[:3]]
                return f"{caption}\n\n{' '.join(tags[:3])}"
    except: pass
    words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', clean)]
    tags = ["#" + w for w in words[:3]]
    return f"{clean}\n\n{' '.join(tags)}"
