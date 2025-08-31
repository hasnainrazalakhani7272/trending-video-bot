import os, random, shutil, textwrap, subprocess, requests
from pathlib import Path
from gtts import gTTS
import math

CONTENT_DIR = Path("content")
OUTPUT_DIR = Path("videos")
MUSIC_DIR = Path("music")
MIN_PER_IMG, MAX_PER_IMG, SRT_WRAP = 3.0, 12.0, 80
VOICE_LANG = "en"
FFMPEG = os.getenv("FFMPEG_BIN","ffmpeg")
FFPROBE = os.getenv("FFPROBE_BIN","ffprobe")

for d in (CONTENT_DIR, OUTPUT_DIR, MUSIC_DIR):
    d.mkdir(exist_ok=True)

def safe_filename(s): return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:180]
def run_cmd(cmd): subprocess.run(cmd, check=True)

def get_audio_duration(path):
    try:
        out = subprocess.check_output([FFPROBE,"-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",path])
        return float(out.strip())
    except: return None

def tts_to_file(text, out_path):
    gTTS(text, lang=VOICE_LANG).save(out_path)
    return out_path

def mix_audio(narration_path, music_path, out_path, music_volume=0.12):
    cmd = [FFMPEG,"-y","-i",narration_path,"-i",music_path,"-filter_complex",
           f"[0:a]volume=1[a0];[1:a]volume={music_volume}[a1];[a0][a1]amix=inputs=2:duration=first:dropout_transition=2[aout]",
           "-map","[aout]","-c:a","aac",out_path]
    run_cmd(cmd)
    return out_path

def split_into_sentences(text):
    parts = [p.strip() for p in text.strip().split(". ") if p.strip()]
    return parts or textwrap.wrap(text, SRT_WRAP)

def build_timed_srt(script, srt_path, total_duration):
    sentences = split_into_sentences(script)
    words_per_sentence = [len(s.split()) for s in sentences]
    total_words = max(1, sum(words_per_sentence))
    times = [max(0.5, (w/total_words)*total_duration) for w in words_per_sentence]
    scale = total_duration / sum(times) if sum(times)>0 else 1.0
    times = [t*scale for t in times]
    def fmt(t): hh=int(t//3600);mm=int((t%3600)//60);ss=int(t%60);ms=int((t-int(t))*1000); return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"
    cur=0.0
    with open(srt_path,"w",encoding="utf-8") as f:
        for i,(sent,dur) in enumerate(zip(sentences,times),1):
            start,end=cur,cur+dur;cur=end
            f.write(f"{i}\n{fmt(start)} --> {fmt(end)}\n")
            for line in textwrap.wrap(sent,SRT_WRAP): f.write(line+"\n")
            f.write("\n")
    return srt_path
