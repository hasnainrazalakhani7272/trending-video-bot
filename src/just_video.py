import os
import json
import requests
from gtts import gTTS
from transformers import pipeline
import subprocess
import random
import google.generativeai as genai

# -------------------- API Keys Setup --------------------
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
PEXELS_API_KEY = os.getenv('PEXELS_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# -------------------- Gemini Summary Generator --------------------
def generate_summary_gemini(headline, full_text):
    """Generate engaging Facebook-friendly summary using Gemini"""
    # Truncate text if too long to fit within token limits
    max_text_length = 4000  # Conservative limit for Gemini
    if len(full_text) > max_text_length:
        full_text = full_text[:max_text_length] + "..."
    
    prompt = f"""
    You are a social media expert creating engaging Facebook post summaries.
    
    Article Headline: {headline}
    Article Content: {full_text}
    
    Create a compelling 2-3 sentence summary that:
    - Hooks readers with an engaging opening
    - Highlights the most important/interesting points
    - Uses conversational tone perfect for Facebook engagement
    - Encourages comments, shares, and reactions
    - Stays between 100-150 words
    - Avoids clickbait but maintains intrigue
    
    Focus on what would make people stop scrolling and want to engage.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[Gemini Summary API error]: {e}")
        return None

# -------------------- Mistral Summary Generator --------------------
def generate_summary_mistral(headline, full_text):
    """Generate engaging summary using Mistral API"""
    # Truncate text if too long
    max_text_length = 3000  # Conservative limit for Mistral
    if len(full_text) > max_text_length:
        full_text = full_text[:max_text_length] + "..."
    
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}"
    }
    
    prompt = f"""Create an engaging Facebook post summary for this news article:

Headline: {headline}
Content: {full_text}

Requirements:
- 2-3 sentences, 100-150 words
- Engaging and conversational tone
- Highlight key points that drive engagement
- Perfect for Facebook audience
- Encourage interaction without being clickbait

Focus on creating content that stops the scroll and drives engagement."""

    data = {
        "model": "mistral-large-latest",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 200,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            print(f"[Mistral API error]: Status {response.status_code}")
            return None
    except Exception as e:
        print(f"[Mistral API error]: {e}")
        return None

# -------------------- Enhanced Summarization Function --------------------
def summarize_article(headline, full_text):
    """
    Enhanced summarization with multiple AI models as fallbacks
    Priority: Gemini -> Mistral -> Hugging Face BART
    """
    
    # First try Gemini (best for engagement)
    print("Trying Gemini for summary generation...")
    gemini_summary = generate_summary_gemini(headline, full_text)
    if gemini_summary:
        print("✓ Gemini summary generated successfully")
        return gemini_summary
    
    # Fallback to Mistral
    print("Gemini failed, trying Mistral...")
    mistral_summary = generate_summary_mistral(headline, full_text)
    if mistral_summary:
        print("✓ Mistral summary generated successfully")
        return mistral_summary
    
    # Final fallback to Hugging Face BART
    print("Both Gemini and Mistral failed, using Hugging Face BART...")
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Enhance the BART summary for Facebook engagement
        bart_summary = summarizer(full_text, max_length=150, min_length=50, do_sample=False)
        raw_summary = bart_summary[0]['summary_text']
        
        # Post-process BART summary to make it more engaging
        engaging_summary = make_summary_engaging(headline, raw_summary)
        print("✓ Hugging Face BART summary generated and enhanced")
        return engaging_summary
        
    except Exception as e:
        print(f"[BART summarization error]: {e}")
        # Ultimate fallback - create basic summary from headline and first part of text
        return create_basic_summary(headline, full_text)

def make_summary_engaging(headline, raw_summary):
    """Enhance BART summary to be more Facebook-friendly"""
    engaging_starters = [
        "🔥 Breaking: ",
        "📢 Important update: ",
        "🚨 Just in: ",
        "💡 Did you know? ",
        "⚡ Latest news: "
    ]
    
    starter = random.choice(engaging_starters)
    enhanced_summary = f"{starter}{raw_summary}"
    
    # Add engagement hook at the end
    engagement_hooks = [
        " What do you think about this?",
        " Share your thoughts below! 👇",
        " Let us know your opinion!",
        " What's your take on this?",
        " Your thoughts? 💭"
    ]
    
    hook = random.choice(engagement_hooks)
    return enhanced_summary + hook

def create_basic_summary(headline, full_text):
    """Create a basic engaging summary as ultimate fallback"""
    # Take first 200 characters of content
    content_snippet = full_text[:200] + "..." if len(full_text) > 200 else full_text
    
    return f"🔥 {headline}\n\n{content_snippet}\n\nStay informed with the latest updates! What are your thoughts? 💭"

# -------------------- Gemini Hashtag Generator --------------------
def generate_hashtags_gemini(headline, summary):
    prompt = f"""
    You are a social media expert generating hashtags for Facebook videos.

    Headline: {headline}
    Summary: {summary}

    Generate 8 to 10 relevant, trending hashtags (each starting with #, no spaces).
    Output hashtags only separated by spaces.
    """
    try:
        response = model.generate_content(prompt)
        hashtags = response.text.strip().split()
        return hashtags
    except Exception as e:
        print(f"[Gemini API error or limit reached]: {e}")
        return None
        
# -------------------- Text2Text Query Generator --------------------
query_generator = pipeline("text2text-generation", model="google/flan-t5-base")

def generate_image_search_query(headline):
    prompt = f"Generate keywords for image search: {headline}"
    result = query_generator(prompt, max_length=20, do_sample=False)
    query = result[0]['generated_text']
    return query

# Ensure necessary directories are created
def ensure_directories():
    directories = ['images', 'output_videos', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# ------------------ CORE FUNCTION 1: Fetch Trending Content ------------------
def fetch_trending_content(limit=5, fetch_more=20):
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"
    resp = requests.get(url).json()
    articles = resp.get("articles", [])

    # Take more articles than limit to increase variety
    articles = articles[:fetch_more]

    # Extract headline/full_text pairs
    headlines_full_texts = [{
        "headline": a.get("title", "No Title"),
        "full_text": a.get("content", "") or a.get("description", "")  # Fallback to description
    } for a in articles]

    # Randomly sample 'limit' articles to increase variation
    if len(headlines_full_texts) > limit:
        headlines_full_texts = random.sample(headlines_full_texts, limit)

    return headlines_full_texts

# ------------------ CORE FUNCTION 2: Fetch Related Images from Pexels ------------------
def get_related_images(query, count=3):
    images = []
    url = f'https://api.pexels.com/v1/search?query={query}&per_page={count}'
    headers = {'Authorization': PEXELS_API_KEY}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            for i, img in enumerate(data['photos']):
                img_url = img['src']['original']
                img_name = f"{query.replace(' ', '_')}_{i}.jpg"
                images.append((img_url, img_name))
        else:
            print(f"Error fetching images: {response.status_code}")
    except Exception as e:
        print(f"Error fetching images: {str(e)}")

    return images

# ------------------ AI Caption Generators ------------------
def generate_caption_gemini(headline, summary):
    """Generate complete Facebook caption using Gemini"""
    prompt = f"""
    Create a compelling Facebook video post caption that maximizes engagement.
    
    Headline: {headline}
    Summary: {summary}
    
    Create a complete Facebook post that includes:
    - Eye-catching opening hook
    - The headline integrated naturally
    - The summary content
    - 8-10 relevant trending hashtags
    - Call-to-action for engagement (comments/shares)
    - Emojis for visual appeal
    
    Make it feel authentic, not promotional. Optimize for Facebook's algorithm.
    Keep total length under 300 words.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[Gemini Caption API error]: {e}")
        return None

def generate_caption_mistral(headline, summary):
    """Generate complete Facebook caption using Mistral"""
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}"
    }
    
    prompt = f"""Create an engaging Facebook video post caption:

Headline: {headline}
Summary: {summary}

Requirements:
- Hook opening that stops the scroll
- Include headline and summary naturally
- Add 8-10 trending hashtags
- Include engaging emojis
- Call-to-action for comments/shares
- Under 300 words total
- Optimize for Facebook engagement

Make it authentic and shareable."""

    data = {
        "model": "mistral-large-latest",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 400,
        "temperature": 0.8
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            print(f"[Mistral Caption API error]: Status {response.status_code}")
            return None
    except Exception as e:
        print(f"[Mistral Caption API error]: {e}")
        return None

# ------------------ Enhanced Caption Generation ------------------
def generate_caption(headline, summary):
    """
    Enhanced caption generation with AI models and fallbacks
    Priority: Gemini -> Mistral -> Template-based with AI hashtags
    """
    
    print("Generating Facebook caption...")
    
    # Try Gemini first
    print("Trying Gemini for caption generation...")
    gemini_caption = generate_caption_gemini(headline, summary)
    if gemini_caption:
        print("✓ Gemini caption generated successfully")
        return gemini_caption
    
    # Fallback to Mistral
    print("Gemini failed, trying Mistral for caption...")
    mistral_caption = generate_caption_mistral(headline, summary)
    if mistral_caption:
        print("✓ Mistral caption generated successfully")
        return mistral_caption
    
    # Final fallback: Enhanced template with AI hashtags
    print("Both AI models failed, using enhanced template...")
    
    # Try to get AI-generated hashtags
    gemini_hashtags = generate_hashtags_gemini(headline, summary)
    
    if gemini_hashtags:
        hashtag_str = " ".join(gemini_hashtags)
    else:
        # Static hashtag fallback
        hashtags = ["#news", "#update", "#video", "#breakingnews", "#currentevents", "#newsfeed"]
        if "technology" in headline.lower():
            hashtags.extend(["#technews", "#technology", "#innovation"])
        elif "sports" in headline.lower():
            hashtags.extend(["#sports", "#sportsnews", "#athletics"])
        elif "finance" in headline.lower():
            hashtags.extend(["#finance", "#economy", "#stocks", "#cryptocurrency"])
        hashtag_str = " ".join(hashtags)

    # Enhanced template caption
    engagement_hooks = [
        "🔥 This just happened and everyone's talking about it!",
        "📰 Breaking news that's changing everything!",
        "🚨 Major update you need to see!",
        "💥 This story is everywhere right now!",
        "⚡ Latest development that's got everyone's attention!"
    ]
    
    hook = random.choice(engagement_hooks)
    
    caption = f"""{hook}

📖 {headline}

{summary}

What's your take on this? Drop your thoughts below! 👇

{hashtag_str}

#StayInformed #NewsUpdate"""

    print("✓ Enhanced template caption generated")
    return caption

# ------------------ CORE FUNCTION 5: Create Video ------------------
def create_video(images, script, index, output_dir, video_duration=60):
    os.makedirs(output_dir, exist_ok=True)
    voice_path = os.path.join(output_dir, f"voice_{index}.mp3")
    gTTS(script, lang="en").save(voice_path)

    image_display_time = video_duration / len(images)
    input_images = []
    for img_url, img_name in images:
        img_path = os.path.join(output_dir, img_name)
        if not os.path.exists(img_path):
            img_data = requests.get(img_url).content
            with open(img_path, "wb") as f:
                f.write(img_data)
        input_images.extend(["-loop", "1", "-t", str(image_display_time), "-i", img_path])

    video_path = os.path.join(output_dir, f"news_{index}.mp4")
    filter_complex = "".join([
        f"[{i}:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
        f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}];"
        for i in range(len(images))
    ])
    filter_complex += "".join([f"[v{i}]" for i in range(len(images))])
    filter_complex += f"concat=n={len(images)}:v=1:a=0,format=yuv420p[v]"

    cmd = [
        "ffmpeg", "-y",
        *input_images,
        "-i", voice_path,
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", f"{len(images)}:a",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", video_path
    ]

    subprocess.run(cmd, check=True)
    return video_path

# ------------------ Fetch and Save Headlines and Full Texts ------------------
def fetch_and_save_headlines_and_texts(limit=5, save_data=True, headlines_file="data/headlines.json"):
    headlines_full_texts = fetch_trending_content(limit)
    if save_data:
        with open(headlines_file, "w") as f:
            json.dump(headlines_full_texts, f, indent=4)
    return headlines_full_texts

# ------------------ Fetch Images and Save (with Fallback Strategy) ------------------
def fetch_images_and_save(headlines_file="data/headlines.json", images_dir="images", save_data=True, images_file="data/images.json"):
    with open(headlines_file, "r") as f:
        articles = json.load(f)

    images_data = []
    os.makedirs(images_dir, exist_ok=True)

    for article in articles:
        headline = article.get("headline")
        images = get_related_images(headline, count=3)

        # If no images, try generating an alternative query
        if not images:
            print(f"No images found for headline: '{headline}'")
            alt_query = generate_image_search_query(headline)
            print(f"Retrying with generated query: '{alt_query}'")
            images = get_related_images(alt_query, count=3)

        # Final fallback to generic "news" if still no images
        if not images:
            print("Using fallback query: 'news'")
            images = get_related_images("news", count=3)

        saved_images = []
        for img_url, img_name in images:
            img_path = os.path.join(images_dir, img_name)
            if not os.path.exists(img_path):
                try:
                    img_data = requests.get(img_url).content
                    with open(img_path, "wb") as f:
                        f.write(img_data)
                    saved_images.append((img_url, img_name))
                except Exception as e:
                    print(f"Error saving image {img_name}: {str(e)}")

        images_data.append({
            "headline": headline,
            "images": saved_images
        })

        if save_data:
            with open(images_file, "w") as f:
                json.dump(images_data, f, indent=4)

    return images_data

# ------------------ Generate Summaries and Save (UPDATED) ------------------
def generate_summaries_and_save(headlines_file="data/headlines.json", summaries_file="data/summaries.json", save_data=True):
    with open(headlines_file, "r") as f:
        articles = json.load(f)

    summaries = {}
    for article in articles:
        headline = article.get("headline")
        full_text = article.get("full_text", "")
        
        if full_text:
            print(f"\n📝 Generating summary for: {headline[:50]}...")
            summary = summarize_article(headline, full_text)
            summaries[headline] = summary
            print(f"✅ Summary generated: {summary[:100]}...")
        else:
            # If no full text, create a basic summary from headline
            summaries[headline] = f"📰 {headline}\n\nStay tuned for more updates on this developing story! What are your thoughts? 💭"

    if save_data:
        with open(summaries_file, "w") as f:
            json.dump(summaries, f, indent=4)

    return summaries

# ------------------ Create Videos and Save ------------------
def create_videos_and_save(images_file="data/images.json", summaries_file="data/summaries.json", output_dir="output_videos", save_data=True):
    os.makedirs(output_dir, exist_ok=True)

    with open(images_file, "r") as f:
        images_data = json.load(f)

    with open(summaries_file, "r") as f:
        summaries = json.load(f)

    video_results = []

    for data in images_data:
        headline = data["headline"]
        images = data["images"]
        summary = summaries.get(headline, "")

        caption = generate_caption(headline, summary)
        video_path = create_video(images, summary, len(video_results)+1, output_dir)

        video_results.append({
            "headline": headline,
            "video_path": video_path,
            "caption": caption
        })

        if save_data:
            with open("data/video_results.json", "w") as f:
                json.dump(video_results, f, indent=4)

    return video_results

# ------------------ Main Execution ------------------
if __name__ == "__main__":
    ensure_directories()
    fetch_and_save_headlines_and_texts(limit=1)
    fetch_images_and_save()
    generate_summaries_and_save()
    create_videos_and_save()