# news_to_videos.py
import os
import requests
import subprocess
import random
import textwrap
import shutil
import uuid
from gtts import gTTS
from transformers import pipeline

# ---------------- CONFIG ----------------
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
UNSPLASH_API_KEY = os.getenv("UNSPLASH_API_KEY")

CONTENT_DIR = "content"
OUTPUT_DIR = "videos"
MUSIC_DIR = "music"   # put free mp3/wav files here

os.makedirs(CONTENT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MUSIC_DIR, exist_ok=True)

# ---------------- MODELS ----------------
# Keep your preferred models; make sure workflow pre-downloads them for stability.
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
caption_gen = pipeline("text2text-generation", model="facebook/bart-base")

# ---------------- Helpers ----------------
def safe_filename(s: str):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:180]

def run_cmd(cmd, check=True):
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=check)

# ---------------- News ----------------
def fetch_trending_content(limit=5):
    if not NEWS_API_KEY:
        raise RuntimeError("NEWS_API_KEY not set")
    url = f"https://newsapi.org/v2/top-headlines?language=en&pageSize={limit}&apiKey={NEWS_API_KEY}"
    resp = requests.get(url, timeout=15).json()
    articles = resp.get("articles", []) or []
    titles = [a.get("title") for a in articles if a.get("title")]
    return titles[:limit]

# ---------------- Script building ----------------
def build_news_script(headline):
    # Summarize headline (short) then expand into ~200-word script
    try:
        short = summarizer(headline, max_length=120, min_length=40, do_sample=False)[0]["summary_text"]
    except Exception as e:
        print("Summarizer failed:", e)
        short = headline

    try:
        prompt = f"Expand this into a clear and engaging news report of around 180-260 words:\n\n{short}"
        expanded = caption_gen(prompt, max_length=300, num_return_sequences=1)[0]["generated_text"]
        # Ensure reasonably long; fallback if too short
        if len(expanded.split()) < 100:
            return short + " " + expanded
        return expanded
    except Exception as e:
        print("Expansion failed:", e)
        return short

# ---------------- Caption + hashtags ----------------
def generate_caption(headline, script):
    try:
        prompt = (
            f"Write one short engaging Facebook caption (1 sentence) "
            f"and add 3 relevant hashtags (space separated) for this news.\n\nHeadline: {headline}\n\n{script}"
        )
        out = caption_gen(prompt, max_length=80, num_return_sequences=1)[0]["generated_text"]
        return out
    except Exception as e:
        print("Caption generation failed:", e)
        # fallback: headline + some hashtags derived
        tags = " ".join("#" + w.capitalize() for w in headline.split()[:3])
        return f"{headline} {tags}"

# ---------------- Unsplash image fetch (with placeholder fallback) ----------------
def get_related_images(query, count=6, save_dir=CONTENT_DIR):
    os.makedirs(save_dir, exist_ok=True)
    imgs = []
    for i in range(count):
        img_path = os.path.join(save_dir, f"{safe_filename(query)}_{i}.jpg")
        try:
            if UNSPLASH_API_KEY:
                url = f"https://api.unsplash.com/photos/random?query={requests.utils.quote(query)}&orientation=landscape&client_id={UNSPLASH_API_KEY}"
                res = requests.get(url, timeout=15).json()
                img_url = res.get("urls", {}).get("regular")
                if img_url:
                    r = requests.get(img_url, timeout=30)
                    with open(img_path, "wb") as f:
                        f.write(r.content)
                    imgs.append(img_path)
                    continue
            # fallback to source.unsplash (no API key) or generate placeholder
            fallback_url = f"https://source.unsplash.com/1920x1080/?{requests.utils.quote(query)}"
            r = requests.get(fallback_url, timeout=30)
            if r.status_code == 200 and r.content:
                with open(img_path, "wb") as f:
                    f.write(r.content)
                imgs.append(img_path)
                continue
        except Exception as e:
            print("Image fetch error:", e)
        # if all fails, generate a placeholder image using ffmpeg
        try:
            generate_placeholder_image(query, img_path)
            imgs.append(img_path)
        except Exception as ex:
            print("Placeholder generation failed:", ex)
    return imgs

def generate_placeholder_image(text, out_path):
    # Use ffmpeg to generate a single-frame image with text
    fontfile = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    # escape single quotes
    safe_text = text.replace("'", r"'\''")
    vf = f"drawtext=fontfile={fontfile}:text='{safe_text}':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=(h-text_h)/2:box=1:boxcolor=black@0.6"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=size=1920x1080:color=#0a0a0a",
        "-vf", vf,
        "-frames:v", "1", out_path
    ]
    run_cmd(cmd)

# ---------------- Subtitles (SRT) ----------------
def make_srt(script, srt_path, seg_seconds=6):
    # Break into wrapped lines and create srt with seg_seconds each
    lines = textwrap.wrap(script, 80)
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, line in enumerate(lines, start=1):
            start_s = (i - 1) * seg_seconds
            end_s = start_s + (seg_seconds - 0.5)
            def fmt(t):
                hh = int(t // 3600)
                mm = int((t % 3600) // 60)
                ss = int(t % 60)
                ms = int((t - int(t)) * 1000)
                return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"
            f.write(f"{i}\n")
            f.write(f"{fmt(start_s)} --> {fmt(end_s)}\n")
            f.write(line + "\n\n")
    return srt_path

# ---------------- Audio helpers ----------------
def tts_to_file(text, out_path):
    gTTS(text, lang="en").save(out_path)
    return out_path

def mix_audio(narration_path, music_path, out_path, music_volume=0.15):
    # create mixed audio so narration stays clear
    # command: ffmpeg -i narration -i music -filter_complex "[0:a]volume=1[a0];[1:a]volume=0.15[a1];[a0][a1]amix=inputs=2:duration=first:dropout_transition=2[aout]" -map "[aout]" out
    cmd = [
        "ffmpeg", "-y",
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

def get_audio_duration(path):
    # ffprobe to get duration in seconds
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path
        ])
        return float(out.strip())
    except Exception:
        return None

# ---------------- Create slideshow video (no audio) ----------------
def create_slideshow(img_paths, per_img_duration, temp_video_path):
    # Build ffmpeg input args
    inputs = []
    for p in img_paths:
        inputs.extend(["-loop", "1", "-t", str(per_img_duration), "-i", p])
    # filter: scale/pad each then concat
    filters = []
    for i in range(len(img_paths)):
        filters.append(f"[{i}:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]")
    concat_inputs = "".join([f"[v{i}]" for i in range(len(img_paths))])
    filter_complex = ";".join(filters) + ";" + concat_inputs + f"concat=n={len(img_paths)}:v=1:a=0,format=yuv420p[vout]"
    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", filter_complex, "-map", "[vout]", "-c:v", "libx264", "-pix_fmt", "yuv420p", temp_video_path]
    run_cmd(cmd)
    return temp_video_path

# ---------------- Combine video + audio + subtitles ----------------
def mux_video_audio_subs(video_path, audio_path, srt_path, out_path):
    # Burn subtitles using subtitles filter and mux audio
    # Use ffmpeg: ffmpeg -i video -i audio -vf "subtitles=..." -map 0:v -map 1:a -c:v libx264 -c:a aac -shortest out
    # Ensure SRT path is absolute and escape colons for windows? ffmpeg handles srt relative path OK.
    abs_srt = os.path.abspath(srt_path)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-vf", f"subtitles={abs_srt}:force_style='FontSize=24,PrimaryColour=&HFFFFFF&'",
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-c:a", "aac",
        "-shortest", out_path
    ]
    run_cmd(cmd)
    return out_path

# ---------------- Main create_video ----------------
def create_video(headline, script, index, output_dir=OUTPUT_DIR, content_dir=CONTENT_DIR):
    tid = uuid.uuid4().hex[:8]
    work_prefix = os.path.join(content_dir, f"work_{tid}")
    os.makedirs(work_prefix, exist_ok=True)

    try:
        # 1) TTS narration
        voice_path = os.path.join(work_prefix, f"voice_{index}.mp3")
        print("Generating TTS...")
        tts_to_file(script, voice_path)
        audio_len = get_audio_duration(voice_path) or 120.0
        print(f"Audio duration: {audio_len:.1f}s")

        # 2) Images
        print("Fetching images...")
        imgs = get_related_images(headline, count=6, save_dir=work_prefix)
        if not imgs:
            raise RuntimeError("No images fetched/created")

        # 3) Create slideshow video (no audio)
        per_img = max(6, audio_len / max(1, len(imgs)))  # ensure slideshow roughly matches narration
        temp_video = os.path.join(work_prefix, "slideshow.mp4")
        print(f"Creating slideshow ({len(imgs)} images, {per_img:.2f}s each)...")
        create_slideshow(imgs, per_img, temp_video)

        # 4) Subtitles (.srt)
        srt = os.path.join(work_prefix, f"subs_{index}.srt")
        make_srt(script, srt, seg_seconds=int(max(4, per_img//2)))  # chunk size relative to per image

        # 5) Background music (optional)
        music_files = [os.path.join(MUSIC_DIR, f) for f in os.listdir(MUSIC_DIR) if f.lower().endswith((".mp3", ".wav"))] if os.path.exists(MUSIC_DIR) else []
        final_audio = os.path.join(work_prefix, "final_audio.mp3")
        if music_files:
            music_path = random.choice(music_files)
            print("Mixing narration with music:", music_path)
            mix_audio(voice_path, music_path, final_audio, music_volume=0.12)
        else:
            print("No music files found; using narration only.")
            shutil.copy(voice_path, final_audio)

        # 6) Mux video + audio + burn subtitles
        out_video = os.path.join(output_dir, f"news_{index}.mp4")
        print("Muxing final video:", out_video)
        mux_video_audio_subs(temp_video, final_audio, srt, out_video)

        print("Done:", out_video)
        return out_video

    finally:
        # cleanup working dir to save space (keep final outputs)
        try:
            shutil.rmtree(work_prefix)
        except Exception as e:
            print("Cleanup failed:", e)

# ---------------- Pipeline ----------------
def create_videos_from_news(limit=2):
    titles = fetch_trending_content(limit=limit)
    results = []
    for i, title in enumerate(titles, start=1):
        print(f"\n===== ITEM {i}: {title} =====")
        script = build_news_script(title)
        caption = generate_caption(title, script)
        video_path = create_video(title, script, i)
        results.append({
            "headline": title,
            "script": script,
            "caption": caption,
            "video_path": video_path
        })
    return results

# ---------------- Quick run for debug ----------------
if __name__ == "__main__":
    print("Running debug generation (limit=1)...")
    out = create_videos_from_news(limit=1)
    print(out)
