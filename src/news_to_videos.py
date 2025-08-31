import os
import requests
import subprocess
from gtts import gTTS
from transformers import pipeline
import random
import textwrap

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
UNSPLASH_API_KEY = os.getenv("UNSPLASH_API_KEY")

# ------------------ INIT MODELS ------------------
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
caption_gen = pipeline("text2text-generation", model="facebook/bart-base")

# ------------------ SOURCES ------------------
COUNTRIES = ["us", "gb", "in", "au", "ca"]
CATEGORIES = ["general", "technology", "business", "science", "health", "sports"]

# ------------------ STEP 1: Fetch Headlines ------------------
def fetch_trending_content(limit=5):
    country = random.choice(COUNTRIES)
    category = random.choice(CATEGORIES)

    url = f"https://newsapi.org/v2/top-headlines?country={country}&category={category}&apiKey={NEWS_API_KEY}"
    resp = requests.get(url).json()
    articles = resp.get("articles", [])
    return [a.get("title", "No Title") for a in articles[:limit]]

# ------------------ STEP 2: Summarize Headline ------------------
def summarize_headline(headline):
    # longer summary for narration (2 min target => ~300 words)
    summary = summarizer(headline, max_length=150, min_length=60, do_sample=False)
    return summary[0]['summary_text']

# ------------------ STEP 3: Generate Caption ------------------
def generate_caption(headline, summary):
    prompt = f"{headline}. {summary}\nMake it sound catchy in one sentence."
    result = caption_gen(prompt, max_length=40, num_return_sequences=1)
    return result[0]['generated_text']

# ------------------ STEP 4: Fetch Related Images ------------------
def get_related_images(query, count=5, save_dir="content"):
    os.makedirs(save_dir, exist_ok=True)
    images = []
    for i in range(count):
        url = f"https://api.unsplash.com/photos/random?query={query}&client_id={UNSPLASH_API_KEY}"
        resp = requests.get(url).json()
        img_url = resp.get("urls", {}).get("regular")
        if img_url:
            img_path = os.path.join(save_dir, f"{query.replace(' ', '_')}_{i}.jpg")
            img_data = requests.get(img_url).content
            with open(img_path, "wb") as f:
                f.write(img_data)
            images.append(img_path)
    return images

# ------------------ STEP 5: Create Video ------------------
def create_video(headline, script, index, output_dir="videos", content_dir="content"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(content_dir, exist_ok=True)

    # Generate narration
    voice_path = os.path.join(content_dir, f"voice_{index}.mp3")
    gTTS(script, lang="en").save(voice_path)

    # Fetch background images
    img_paths = get_related_images(headline, count=6, save_dir=content_dir)

    video_path = os.path.join(output_dir, f"news_{index}.mp4")

    # Duration ~120s (2 min), divide across images
    per_img_time = max(15, 120 // max(1, len(img_paths)))

    # Build ffmpeg inputs
    input_images = []
    for img in img_paths:
        input_images.extend(["-loop", "1", "-t", str(per_img_time), "-i", img])

    # Build filter_complex with text overlay
    filter_complex = ""
    overlays = []
    for i, img in enumerate(img_paths):
        txt = headline if i == 0 else script[:120]  # first: headline, then summary snippet
        wrapped = textwrap.fill(txt, width=50)
        filter_complex += (
            f"[{i}:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,"
            f"drawtext=text='{wrapped}':x=(w-text_w)/2:y=h-120:fontsize=36:fontcolor=white:shadowx=2:shadowy=2[v{i}];"
        )
        overlays.append(f"[v{i}]")
    filter_complex += "".join(overlays)
    filter_complex += f"concat=n={len(img_paths)}:v=1:a=0,format=yuv420p[v]"

    cmd = [
        "ffmpeg", "-y",
        *input_images,
        "-i", voice_path,
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", f"{len(img_paths)}:a",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", video_path
    ]
    subprocess.run(cmd, check=True)
    return video_path

# ------------------ PIPELINE ------------------
def create_videos_from_news(limit=2):
    results = []
    headlines = fetch_trending_content(limit=limit)

    for i, headline in enumerate(headlines, start=1):
        script = summarize_headline(headline)
        caption = generate_caption(headline, script)
        video_path = create_video(headline, script, i)

        results.append({
            "headline": headline,
            "script": script,
            "caption": caption,
            "video_path": video_path
        })
    return results
