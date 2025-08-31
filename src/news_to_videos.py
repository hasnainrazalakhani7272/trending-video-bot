# src/news_to_videos.py
"""
Production-ready news -> video pipeline with:
- persistent dedupe (fuzzy), resumable workdirs
- free image sources: Wikimedia Commons -> source.unsplash -> ffmpeg placeholder
- improved audio/video sync (SRT timed to narration)
- robust caption + hashtag generation with sanitization and deterministic fallback
- transformers optional (falls back if not available)
"""

from __future__ import annotations
import os
import re
import json
import uuid
import shutil
import random
import textwrap
import subprocess
import requests
import time
import math
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Tuple

from gtts import gTTS

# transformers optional
try:
    from transformers import pipeline
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

# ---------------- CONFIG ----------------
CONTENT_DIR = Path("content")
OUTPUT_DIR = Path("videos")
MUSIC_DIR = Path("music")
STATE_FILE = CONTENT_DIR / "processed.json"   # stores list of processed normalized headlines
MAX_PROCESSED = 5000

NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # required
UNSPLASH_API_KEY = os.getenv("UNSPLASH_API_KEY", "")  # optional, not required for source.unsplash

# ffmpeg / ffprobe commands
FFMPEG = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE = os.getenv("FFPROBE_BIN", "ffprobe")

# models (optional)
SUMMARIZER_MODEL = "facebook/bart-large-cnn"
CAPTION_MODEL = "facebook/bart-base"

# audio/video tuning
MIN_PER_IMG = 3.0     # minimum seconds per image
MAX_PER_IMG = 12.0    # maximum seconds per image
SRT_WRAP = 80         # wrap cols for subtitle lines
VOICE_LANG = "en"

# fuzzy dedupe threshold (0-1). If normalized new title is > THRESH with any processed -> skip.
FUZZY_THRESHOLD = 0.82

# ensure dirs
for d in (CONTENT_DIR, OUTPUT_DIR, MUSIC_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------- UTILITIES ----------------
def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:180]

def run_cmd(cmd: List[str], check: bool = True) -> None:
    """Run and stream subprocess output (ffmpeg progress will appear in logs)."""
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=check)

def get_audio_duration(path: str) -> Optional[float]:
    try:
        out = subprocess.check_output([
            FFPROBE, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path
        ])
        return float(out.strip())
    except Exception:
        return None

# ---------------- TRANSFORMERS (optional) ----------------
summarizer = None
caption_gen = None
if _HAS_TRANSFORMERS:
    try:
        summarizer = pipeline("summarization", model=SUMMARIZER_MODEL)
    except Exception:
        summarizer = None
    try:
        caption_gen = pipeline("text2text-generation", model=CAPTION_MODEL)
    except Exception:
        caption_gen = None

# ---------------- NORMALIZE HEADLINE & DEDUPE ----------------
def normalize_headline(title: str) -> str:
    """
    Normalize headline to unify small variants:
    - Remove common ' - Source' suffixes, known site names
    - Convert smart quotes, weird punctuation
    - Lowercase and collapse spaces
    """
    t = title or ""
    # replace smart quotes
    t = t.replace("“", '"').replace("”", '"').replace("’", "'").replace("—", "-")
    # remove common source suffixes like " - Financial Times" or " | BBC News"
    t = re.sub(r"\s+[-|]\s*[^-|\n]{1,80}$", "", t)
    # remove leading site mentions like "The New York Times.Headline:" or "FT:"
    t = re.sub(r'^(the|a|an)\s+[\w\s]{1,40}\.?[:\-]\s*', '', t, flags=re.I)
    # remove repeated words/garbage sequences (e.g., multiple site names stuck together)
    t = re.sub(r'(\b[A-Za-z]{2,}\b)(\s+\1)+', r'\1', t)
    # collapse whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    return t.lower()

def load_processed() -> List[str]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def save_processed(items: List[str]) -> None:
    items = items[-MAX_PROCESSED:]
    STATE_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

def fuzzy_already_processed(norm_title: str, processed: List[str]) -> bool:
    # fast exact check
    if norm_title in processed:
        return True
    # fuzzy check
    for p in processed:
        if SequenceMatcher(None, norm_title, p).ratio() >= FUZZY_THRESHOLD:
            return True
    return False

# ---------------- NEWS FETCH ----------------
def fetch_trending_content(limit: int = 5) -> List[dict]:
    """
    Returns list of dicts: {"title": str, "source": str, "url": str}
    Requires NEWS_API_KEY.
    """
    if not NEWS_API_KEY:
        raise RuntimeError("NEWS_API_KEY not set in environment")
    url = "https://newsapi.org/v2/top-headlines"
    params = {"language": "en", "pageSize": limit, "apiKey": NEWS_API_KEY}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    out = []
    for a in data.get("articles", []) or []:
        out.append({
            "title": (a.get("title") or "").strip(),
            "source": (a.get("source", {}).get("name") or "").strip(),
            "url": (a.get("url") or "").strip()
        })
    # unique by title
    seen = set()
    uniq = []
    for x in out:
        t = x["title"]
        if t and t not in seen:
            seen.add(t)
            uniq.append(x)
    return uniq[:limit]

# ---------------- SCRIPT BUILDING ----------------
def build_news_script(headline: str) -> str:
    # step 1: short summary
    short = headline
    try:
        if summarizer:
            res = summarizer(headline, max_length=120, min_length=20, do_sample=False)
            if res and isinstance(res, list):
                short = res[0].get("summary_text") or short
    except Exception:
        short = headline
    # step 2: expand into ~180-260 words
    expanded = None
    try:
        if caption_gen:
            prompt = (
                f"Expand the following short news summary into a clear, engaging news report "
                f"of about 180-260 words. Keep it factual, neutral tone.\n\nSummary: {short}"
            )
            res = caption_gen(prompt, max_length=320, num_return_sequences=1)
            if res and isinstance(res, list):
                expanded = res[0].get("generated_text") or res[0].get("text")
    except Exception:
        expanded = None
    if expanded and len(expanded.split()) >= 120:
        return re.sub(r'\s+', ' ', expanded).strip()
    # deterministic fallback
    fallback = (
        f"{short}. In brief: {short}. For context: {short}. "
        "This brief report outlines the main facts and background for readers."
    )
    return fallback

# ---------------- CAPTION + HASHTAGS (robust) ----------------
def _parse_model_caption(raw: str) -> Optional[Tuple[str, List[str]]]:
    """
    Expect model to return: one sentence caption + hashtags separated by spaces or on new line.
    This parser tries to extract a single sentence and up to 3 hashtags.
    """
    if not raw:
        return None
    txt = raw.strip()
    # remove newlines except between caption/hashtags
    parts = [p.strip() for p in re.split(r'[\r\n]{1,}', txt) if p.strip()]
    # heuristic: if first part ends with period/exclamation/question and second part has hashtags -> good
    caption = parts[0]
    hashtags = []
    # find hashtags anywhere
    tags_found = re.findall(r'#\w[\w\-]+', txt)
    if tags_found:
        hashtags = tags_found[:3]
    else:
        # try to find words to make hashtags from second line or last words
        if len(parts) > 1:
            candidates = re.findall(r'\b\w+\b', parts[-1])
            hashtags = [f"#{safe_filename(w).lower()}" for w in candidates[:3]]
    # ensure caption is one sentence max (truncate after first sentence)
    sen = re.split(r'(?<=[.!?])\s+', caption.strip())[0]
    caption = sen.strip()
    if caption:
        if not hashtags:
            return (caption, [])
        return (caption, hashtags)
    return None

def generate_caption_and_hashtags(headline: str, script: str) -> str:
    # sanitize headline to remove appended source strings (we already normalize separately, but be safe)
    clean_headline = re.sub(r'\s+[-|]\s*[^-|\n]{1,80}$', '', headline).strip()
    # model attempt
    try:
        if caption_gen:
            prompt = (
                "Write ONE short engaging social media caption (single sentence) for the news, "
                "then on a new line provide exactly 3 relevant hashtags separated by spaces. "
                "Do not add extra explanation.\n\n"
                f"Headline: {clean_headline}\n\n{script}"
            )
            res = caption_gen(prompt, max_length=80, num_return_sequences=1)
            raw = None
            if res and isinstance(res, list):
                raw = res[0].get("generated_text") or res[0].get("text")
            parsed = _parse_model_caption(raw)
            if parsed:
                caption, tags = parsed
                # ensure exactly 3 hashtags
                if len(tags) < 3:
                    # fill with keywords from headline
                    kws = [w for w in re.findall(r'\b\w{4,}\b', clean_headline)][:3 - len(tags)]
                    tags += [f"#{safe_filename(w).lower()}" for w in kws]
                tags = tags[:3]
                return f"{caption}\n\n{' '.join(tags)}"
    except Exception:
        pass
    # deterministic fallback
    caption = clean_headline if len(clean_headline) < 120 else clean_headline[:117] + "..."
    words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', clean_headline)]
    uniq = []
    for w in words:
        if w not in uniq:
            uniq.append(w)
        if len(uniq) >= 3:
            break
    if not uniq:
        uniq = clean_headline.split()[:3]
    tags = " ".join("#" + safe_filename(w).lower() for w in uniq[:3])
    return f"{caption}\n\n{tags}"

# ---------------- IMAGES: Wikimedia -> source.unsplash -> placeholder ----------------
def download_stream_to(path: Path, url: str, timeout: int = 30) -> bool:
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception:
        return False

def generate_placeholder_image(text: str, out_path: Path) -> bool:
    fontfile = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    safe_text = text.replace("'", r"'\''")
    vf = (
        f"drawtext=fontfile={fontfile}:text='{safe_text}':fontcolor=white:fontsize=48:"
        "x=(w-text_w)/2:y=(h-text_h)/2:box=1:boxcolor=black@0.6"
    )
    cmd = [
        FFMPEG, "-y",
        "-f", "lavfi", "-i", "color=size=1920x1080:color=#0a0a0a",
        "-vf", vf,
        "-frames:v", "1", str(out_path)
    ]
    try:
        run_cmd(cmd)
        return True
    except Exception:
        return False

def fetch_wikimedia(query: str, count: int, save_dir: Path) -> List[Path]:
    imgs: List[Path] = []
    search_url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": count,
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": 1920
    }
    try:
        r = requests.get(search_url, params=params, timeout=15)
        r.raise_for_status()
        j = r.json()
        pages = j.get("query", {}).get("pages", {})
        for idx, (pid, page) in enumerate(pages.items()):
            ii = page.get("imageinfo", [])
            if not ii:
                continue
            url = ii[0].get("url")
            if not url:
                continue
            p = save_dir / f"{safe_filename(query)}_wm_{idx}.jpg"
            if download_stream_to(p, url):
                imgs.append(p)
            if len(imgs) >= count:
                break
    except Exception:
        pass
    return imgs

def fetch_source_unsplash(query: str, count: int, save_dir: Path) -> List[Path]:
    imgs: List[Path] = []
    for i in range(count):
        url = f"https://source.unsplash.com/1920x1080/?{requests.utils.quote(query)}"
        p = save_dir / f"{safe_filename(query)}_us_{i}.jpg"
        if download_stream_to(p, url):
            imgs.append(p)
    return imgs

def get_related_images(headline: str, count: int = 6, save_dir: Path = CONTENT_DIR) -> List[str]:
    save_dir.mkdir(parents=True, exist_ok=True)
    # cache-dir per normalized headline (so repeated runs reuse images)
    norm = safe_filename(headline)[:80]
    work_dir = save_dir / f"images_{norm}"
    work_dir.mkdir(parents=True, exist_ok=True)
    existing = list(work_dir.glob("*.jpg"))
    if existing:
        return [str(p) for p in existing[:count]]
    # 1) Wikimedia
    imgs = fetch_wikimedia(headline, count, work_dir)
    if imgs:
        return [str(p) for p in imgs[:count]]
    # 2) Source Unsplash fallback (no API key)
    imgs = fetch_source_unsplash(headline, count, work_dir)
    if imgs:
        return [str(p) for p in imgs[:count]]
    # 3) placeholder images
    imgs = []
    for i in range(count):
        p = work_dir / f"{safe_filename(headline)}_ph_{i}.jpg"
        if generate_placeholder_image(headline, p):
            imgs.append(p)
    return [str(p) for p in imgs[:count]]

# ---------------- SUBTITLES: timed to narration ----------------
def split_into_sentences(text: str) -> List[str]:
    # naive sentence splitter but OK for SRT: split on .!? followed by space and capital
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        # fallback to wrap
        return textwrap.wrap(text, SRT_WRAP)
    return parts

def build_timed_srt(script: str, srt_path: str, total_duration: float) -> str:
    """
    Split script into sentences, allocate duration proportional to word counts,
    and write SRT with precise start/end times aligned to total_duration.
    """
    sentences = split_into_sentences(script)
    words_per_sentence = [len(re.findall(r'\w+', s)) for s in sentences]
    total_words = max(1, sum(words_per_sentence))
    # compute times
    times = []
    for w in words_per_sentence:
        times.append(max(0.5, (w / total_words) * total_duration))
    # normalize to exactly total_duration (adjust rounding)
    scale = total_duration / sum(times) if sum(times) > 0 else 1.0
    times = [t * scale for t in times]
    # write srt
    def fmt(t: float) -> str:
        hh = int(t // 3600)
        mm = int((t % 3600) // 60)
        ss = int(t % 60)
        ms = int((t - math.floor(t)) * 1000)
        return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"
    cur = 0.0
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, (sent, dur) in enumerate(zip(sentences, times), start=1):
            start = cur
            end = cur + dur
            cur = end
            f.write(f"{i}\n")
            f.write(f"{fmt(start)} --> {fmt(end)}\n")
            # wrap lines for readability
            for line in textwrap.wrap(sent, SRT_WRAP):
                f.write(line + "\n")
            f.write("\n")
    return srt_path

# ---------------- AUDIO helpers ----------------
def tts_to_file(text: str, out_path: str) -> str:
    gTTS(text, lang=VOICE_LANG).save(out_path)
    return out_path

def mix_audio(narration_path: str, music_path: str, out_path: str, music_volume: float = 0.12) -> str:
    cmd = [
        FFMPEG, "-y",
        "-i", narration_path,
        "-i", music_path,
        "-filter_complex",
        f"[0:a]volume=1[a0];[1:a]volume={music_volume}[a1];[a0][a1]amix=inputs=2:duration=first:dropout_transition=2[aout]",
        "-map", "[aout]",
        "-c:a", "aac",
        out_path
    ]
    run_cmd(cmd)
    return out_path

# ---------------- CREATE SLIDESHOW & MUX ----------------
def create_slideshow(img_paths: List[str], per_img: float, temp_video: str) -> str:
    # Each image will be provided as "-loop 1 -t per_img -i path"
    inputs = []
    for p in img_paths:
        inputs.extend(["-loop", "1", "-t", str(per_img), "-i", p])
    filters = []
    for i in range(len(img_paths)):
        filters.append(
            f"[{i}:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]"
        )
    concat_inputs = "".join([f"[v{i}]" for i in range(len(img_paths))])
    filter_complex = ";".join(filters) + ";" + concat_inputs + f"concat=n={len(img_paths)}:v=1:a=0,format=yuv420p[vout]"
    cmd = [FFMPEG, "-y", *inputs, "-filter_complex", filter_complex, "-map", "[vout]", "-c:v", "libx264", "-pix_fmt", "yuv420p", temp_video]
    run_cmd(cmd)
    return temp_video

def mux_audio_video_subs(video_path: str, audio_path: str, srt_path: str, out_path: str) -> str:
    abs_srt = os.path.abspath(srt_path)
    vf = f"subtitles={abs_srt}:force_style='FontSize=24,PrimaryColour=&HFFFFFF&'"
    cmd = [
        FFMPEG, "-y",
        "-i", video_path,
        "-i", audio_path,
        "-vf", vf,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-c:a", "aac",
        "-shortest", out_path
    ]
    run_cmd(cmd)
    return out_path

# ---------------- PER-ITEM WORKFLOW ----------------
def create_video_for_headline(headline: str, script: str, index: int, output_dir: Path = OUTPUT_DIR, content_dir: Path = CONTENT_DIR) -> str:
    tid = uuid.uuid4().hex[:8]
    work = content_dir / f"work_{tid}"
    # if same headline run before and work dir exists, let it be (resumable) - do not overwrite
    work.mkdir(parents=True, exist_ok=True)
    try:
        # 1) narration (TTS)
        voice_path = str(work / f"voice_{index}.mp3")
        tts_to_file(script, voice_path)
        audio_len = get_audio_duration(voice_path) or max(10.0, len(script.split()) / 2.0)

        # 2) images
        images = get_related_images(headline, count=6, save_dir=content_dir)
        if not images:
            raise RuntimeError("No images available for headline")

        # 3) per-image duration matching narration
        per_img = audio_len / max(1, len(images))
        per_img = max(MIN_PER_IMG, min(MAX_PER_IMG, per_img))

        temp_slideshow = str(work / "slideshow.mp4")
        create_slideshow(images, per_img, temp_slideshow)

        # 4) subtitles timed to audio length
        srt_path = str(work / f"subs_{index}.srt")
        build_timed_srt(script, srt_path, total_duration=audio_len)

        # 5) mix music if available
        music_files = [str(p) for p in MUSIC_DIR.iterdir() if p.suffix.lower() in (".mp3", ".wav")] if MUSIC_DIR.exists() else []
        final_audio = str(work / "final_audio.mp3")
        if music_files:
            music = random.choice(music_files)
            mix_audio(voice_path, music, final_audio, music_volume=0.12)
        else:
            shutil.copy(voice_path, final_audio)

        # 6) mux video+audio+burn subs
        out_file = output_dir / f"news_{index}_{tid}.mp4"
        mux_audio_video_subs(temp_slideshow, final_audio, srt_path, str(out_file))

        return str(out_file)
    finally:
        # remove work dir to save space only on success; leave for inspection if something went wrong.
        # We'll remove only if out file exists
        try:
            if (output_dir / f"news_{index}_{tid}.mp4").exists():
                shutil.rmtree(work)
        except Exception:
            pass

# ---------------- PIPELINE ----------------
def create_videos_from_news(limit: int = 2) -> List[dict]:
    items = fetch_trending_content(limit=limit)
    processed = load_processed()
    results = []
    for i, doc in enumerate(items, start=1):
        raw_title = doc.get("title") or ""
        norm = normalize_headline(raw_title)
        if fuzzy_already_processed(norm, processed):
            print(f"SKIP (already processed): {raw_title}")
            continue

        print(f"START: {raw_title}")
        script = build_news_script(raw_title)
        caption = generate_caption_and_hashtags(raw_title, script)

        try:
            video_path = create_video_for_headline(raw_title, script, i)
            results.append({
                "headline": raw_title,
                "script": script,
                "caption": caption,
                "video_path": video_path,
                "source": doc.get("source"),
                "url": doc.get("url")
            })
            # checkpoint (store normalized form)
            processed.append(norm)
            save_processed(processed)
            print(f"COMPLETED: {video_path}")
        except Exception as e:
            print(f"ERROR processing '{raw_title}': {e}")
            # don't stop pipeline
            continue
    return results

# ------------ CLI -------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()
    out = create_videos_from_news(limit=args.limit)
    print(json.dumps(out, indent=2))

