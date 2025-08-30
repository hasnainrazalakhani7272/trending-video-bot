import os
import requests
import subprocess
from gtts import gTTS
import google.generativeai as genai

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
UNSPLASH_API_KEY = os.getenv("UNSPLASH_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

# ------------------ STEP 1: Fetch Headlines ------------------
def fetch_trending_content(limit=5):
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"
    resp = requests.get(url).json()
    articles = resp.get("articles", [])
    return [a.get("title", "No Title") for a in articles[:limit]]

# ------------------ STEP 2: Summarize with Gemini ------------------
def summarize_headline(headline):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Summarize this news headline in 2-3 sentences for a video script:\n\n{headline}"
    response = model.generate_content(prompt)
    return response.text.strip()

# ------------------ STEP 3: Generate Social Caption ------------------
def generate_caption(headline, summary):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
    Create a short engaging Facebook post caption for this news video.
    Include a catchy hook, a brief informative summary, a call-to-action to follow,
    and 4-6 relevant hashtags.
    
    Headline: {headline}
    Summary: {summary}
    """
    response = model.generate_content(prompt)
    return response.text.strip()

# ------------------ STEP 4: Fetch Related Images ------------------
def get_related_images(query, count=3, save_dir="content"):
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
def create_video(headline, script, index, output_dir="videos"):
    os.makedirs(output_dir, exist_ok=True)

    # Generate TTS narration
    voice_path = f"content/voice_{index}.mp3"
    gTTS(script, lang="en").save(voice_path)

    # Fetch multiple background images
    img_paths = get_related_images(headline, count=3)

    video_path = os.path.join(output_dir, f"news_{index}.mp4")

    # Build ffmpeg inputs
    input_images = []
    for img in img_paths:
        input_images.extend(["-loop", "1", "-t", "10", "-i", img])

    # Filter chain
    filter_complex = ""
    for i in range(len(img_paths)):
        filter_complex += (
            f"[{i}:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,"
            f"setsar=1[v{i}];"
        )
    filter_complex += "".join([f"[v{i}]" for i in range(len(img_paths))])
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

