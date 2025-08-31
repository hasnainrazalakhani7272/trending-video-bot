import os, textwrap
from gtts import gTTS
import subprocess, math
from pathlib import Path

FFMPEG = os.getenv("FFMPEG_BIN","ffmpeg")

def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:180]

def tts_to_file(text, out_path):
    gTTS(text, lang="en").save(out_path)
    return out_path

def get_audio_duration(path):
    try:
        out = subprocess.check_output([ "ffprobe","-v","error","-show_entries","format=duration",
                                        "-of","default=noprint_wrappers=1:nokey=1", path ])
        return float(out.strip())
    except: return None

def split_into_sentences(text, width=80):
    parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', text.strip()) if p.strip()]
    return parts if parts else textwrap.wrap(text, width)

def build_timed_srt(script, srt_path, total_duration):
    sentences = split_into_sentences(script)
    words = [len(s.split()) for s in sentences]
    total_words = max(1,sum(words))
    times = [max(0.5,(w/total_words)*total_duration) for w in words]
    scale = total_duration/sum(times) if sum(times)>0 else 1
    times = [t*scale for t in times]
    cur=0.0
    def fmt(t): hh=int(t//3600); mm=int((t%3600)//60); ss=int(t%60); ms=int((t-math.floor(t))*1000); return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"
    with open(srt_path,"w",encoding="utf-8") as f:
        for i,(s,d) in enumerate(zip(sentences,times),start=1):
            start, end = cur, cur+d
            cur=end
            f.write(f"{i}\n{fmt(start)} --> {fmt(end)}\n")
            for line in textwrap.wrap(s,80): f.write(line+"\n")
            f.write("\n")
