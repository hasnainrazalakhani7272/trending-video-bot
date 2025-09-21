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

# -------------------- Unified Content Generation --------------------
def generate_content_gemini(headline, full_text):
    """Generate engaging content using Gemini for both TTS and Facebook use"""
    # Truncate text if too long to fit within token limits
    max_text_length = 4000
    if len(full_text) > max_text_length:
        full_text = full_text[:max_text_length] + "..."
    
    prompt = f"""
    Create engaging content for a Facebook video post that will be used for both:
    1. Text-to-speech voice-over in the video
    2. Facebook post content
    
    Article Headline: {headline}
    Article Content: {full_text}
    
    Generate 150-200 words of engaging content that:
    - Expands on the headline with key information
    - Uses conversational, engaging tone
    - Works well when spoken (TTS) and when read
    - Hooks audience attention
    - Encourages engagement
    - No hashtags (those will be added separately)
    - No emojis in main content
    
    Focus on making it informative yet captivating for social media audience.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[Gemini Content API error]: {e}")
        return None

def generate_content_mistral(headline, full_text):
    """Generate engaging content using Mistral"""
    max_text_length = 3000
    if len(full_text) > max_text_length:
        full_text = full_text[:max_text_length] + "..."
    
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}"
    }
    
    prompt = f"""Create engaging 150-200 word content for a Facebook video:

Headline: {headline}
Article: {full_text}

Requirements:
- Expand on headline with key details
- Conversational and engaging tone
- Works for both voice-over and Facebook post
- Hook audience attention
- Encourage engagement
- No hashtags or emojis in main content
- Informative yet captivating

Perfect for social media audience."""

    data = {
        "model": "mistral-large-latest",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            print(f"[Mistral Content API error]: Status {response.status_code}")
            return None
    except Exception as e:
        print(f"[Mistral Content API error]: {e}")
        return None

def generate_unified_content(headline, full_text):
    """
    Generate unified content with fallback strategy
    Priority: Gemini -> Mistral -> Enhanced BART -> Basic template
    """
    
    print(f"\n📝 Generating content for: {headline[:50]}...")
    
    # First try Gemini
    print("Trying Gemini for content generation...")
    gemini_content = generate_content_gemini(headline, full_text)
    if gemini_content:
        print("✓ Gemini content generated successfully")
        return gemini_content
    
    # Fallback to Mistral
    print("Gemini failed, trying Mistral...")
    mistral_content = generate_content_mistral(headline, full_text)
    if mistral_content:
        print("✓ Mistral content generated successfully")
        return mistral_content
    
    # Fallback to Enhanced BART
    print("Both AI models failed, using Enhanced BART...")
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        bart_summary = summarizer(full_text, max_length=200, min_length=100, do_sample=False)
        raw_content = bart_summary[0]['summary_text']
        
        # Enhance BART content for engagement
        enhanced_content = f"Here's what's happening: {raw_content} This developing story continues to unfold, and we're keeping you updated with all the latest information. What are your thoughts on this situation?"
        print("✓ Enhanced BART content generated")
        return enhanced_content
        
    except Exception as e:
        print(f"[BART content error]: {e}")
        # Ultimate fallback
        return create_basic_content(headline, full_text)

def create_basic_content(headline, full_text):
    """Create basic engaging content as ultimate fallback"""
    content_snippet = full_text[:300] + "..." if len(full_text) > 300 else full_text
    
    basic_content = f"""Breaking news update: {headline}

{content_snippet}

This is a developing story, and we're monitoring the situation closely. We'll continue to bring you the latest updates as they become available. 

Stay informed and let us know what you think about this development in the comments below."""

    print("✓ Basic template content generated")
    return basic_content

# -------------------- Hashtag Generation (Separate) --------------------
def generate_hashtags_gemini(headline, content):
    """Generate hashtags using Gemini"""
    prompt = f"""
    Generate 8-10 trending, relevant hashtags for this Facebook video post:

    Headline: {headline}
    Content: {content}

    Requirements:
    - Each hashtag starts with #
    - No spaces in hashtags
    - Mix of specific and general hashtags
    - Trending and engaging hashtags
    - Relevant to news/current events
    
    Output only hashtags separated by spaces.
    """
    try:
        response = model.generate_content(prompt)
        hashtags = response.text.strip().split()
        return hashtags
    except Exception as e:
        print(f"[Gemini Hashtag API error]: {e}")
        return None

def generate_hashtags_mistral(headline, content):
    """Generate hashtags using Mistral"""
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}"
    }
    
    prompt = f"""Generate 8-10 trending hashtags for this Facebook post:

Headline: {headline}
Content preview: {content[:200]}...

Requirements:
- Start with #, no spaces
- Mix specific and general tags
- Trending and engaging
- News/current events focused

Output only hashtags separated by spaces."""

    data = {
        "model": "mistral-large-latest",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 100,
        "temperature": 0.8
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip().split()
        else:
            print(f"[Mistral Hashtag API error]: Status {response.status_code}")
            return None
    except Exception as e:
        print(f"[Mistral Hashtag API error]: {e}")
        return None

def generate_hashtags(headline, content):
    """
    Generate hashtags with fallback strategy
    Priority: Gemini -> Mistral -> Keyword-based -> Static
    """
    
    print("Generating hashtags...")
    
    # First try Gemini
    print("Trying Gemini for hashtags...")
    gemini_hashtags = generate_hashtags_gemini(headline, content)
    if gemini_hashtags:
        print("✓ Gemini hashtags generated successfully")
        return gemini_hashtags
    
    # Fallback to Mistral
    print("Gemini failed, trying Mistral for hashtags...")
    mistral_hashtags = generate_hashtags_mistral(headline, content)
    if mistral_hashtags:
        print("✓ Mistral hashtags generated successfully")
        return mistral_hashtags
    
    # Keyword-based fallback
    print("AI models failed, using keyword-based hashtags...")
    hashtags = ["#news", "#update", "#breakingnews", "#currentevents", "#newsfeed", "#trending"]
    
    # Add specific hashtags based on content
    text_lower = (headline + " " + content).lower()
    
    if any(word in text_lower for word in ["tech", "technology", "ai", "digital"]):
        hashtags.extend(["#technology", "#tech", "#innovation"])
    if any(word in text_lower for word in ["sports", "game", "team", "player"]):
        hashtags.extend(["#sports", "#sportsnews"])
    if any(word in text_lower for word in ["economy", "market", "finance", "stock"]):
        hashtags.extend(["#finance", "#economy", "#business"])
    if any(word in text_lower for word in ["health", "medical", "hospital"]):
        hashtags.extend(["#health", "#healthcare", "#medical"])
    if any(word in text_lower for word in ["politics", "government", "election"]):
        hashtags.extend(["#politics", "#government", "#policy"])
    
    # Add engagement hashtags
    hashtags.extend(["#stayinformed", "#newsupdate", "#socialmedia"])
    
    # Return first 10 hashtags
    final_hashtags = hashtags[:10]
    print(f"✓ Keyword-based hashtags generated: {len(final_hashtags)} hashtags")
    return final_hashtags

# -------------------- Image Search Query Generator --------------------
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
        "full_text": a.get("content", "") or a.get("description", "")
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

# ------------------ CORE FUNCTION 3: Create Video ------------------
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

# ------------------ NEW: Generate Facebook Post ------------------
def create_facebook_post(headline, content, hashtags):
    """Create final Facebook post combining content and hashtags"""
    
    # Add engaging hooks and emojis for Facebook
    post_hooks = [
        "🔥 Breaking News Alert!",
        "📰 Latest Update:",
        "🚨 Just In:",
        "💥 Major Development:",
        "⚡ News Flash:"
    ]
    
    hook = random.choice(post_hooks)
    hashtag_str = " ".join(hashtags) if isinstance(hashtags, list) else hashtags
    
    facebook_post = f"""{hook}

{content}

What's your take on this? Share your thoughts below! 👇

{hashtag_str}

#StayInformed #NewsUpdate"""

    return facebook_post

# ------------------ Fetch and Save Headlines and Full Texts ------------------
def fetch_and_save_headlines_and_texts(limit=5, save_data=True, headlines_file="data/headlines.json"):
    headlines_full_texts = fetch_trending_content(limit)
    if save_data:
        with open(headlines_file, "w") as f:
            json.dump(headlines_full_texts, f, indent=4)
    return headlines_full_texts

# ------------------ Fetch Images and Save ------------------
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

# ------------------ NEW: Generate Content and Hashtags ------------------
def generate_content_and_hashtags(headlines_file="data/headlines.json", content_file="data/content.json", save_data=True):
    """Generate unified content and hashtags separately"""
    
    with open(headlines_file, "r") as f:
        articles = json.load(f)

    content_data = {}
    
    for article in articles:
        headline = article.get("headline")
        full_text = article.get("full_text", "")
        
        print(f"\n🎯 Processing: {headline}")
        
        # Generate unified content
        if full_text:
            content = generate_unified_content(headline, full_text)
        else:
            content = f"Breaking news: {headline}. We're following this developing story and will provide more details as they become available. Stay tuned for updates on this important story."
        
        # Generate hashtags separately
        hashtags = generate_hashtags(headline, content)
        
        content_data[headline] = {
            "content": content,
            "hashtags": hashtags
        }
        
        print(f"✅ Content and hashtags generated for: {headline[:50]}...")

    if save_data:
        with open(content_file, "w") as f:
            json.dump(content_data, f, indent=4)

    return content_data

# ------------------ Create Videos and Facebook Posts ------------------
def create_videos_and_posts(images_file="data/images.json", content_file="data/content.json", output_dir="output_videos", save_data=True):
    """Create videos and Facebook posts using unified content"""
    
    os.makedirs(output_dir, exist_ok=True)

    with open(images_file, "r") as f:
        images_data = json.load(f)

    with open(content_file, "r") as f:
        content_data = json.load(f)

    video_results = []

    for data in images_data:
        headline = data["headline"]
        images = data["images"]
        
        # Get content and hashtags
        article_data = content_data.get(headline, {})
        content = article_data.get("content", f"Breaking: {headline}")
        hashtags = article_data.get("hashtags", ["#news", "#update"])
        
        # Create video using content for TTS
        video_path = create_video(images, content, len(video_results)+1, output_dir)
        
        # Create Facebook post
        facebook_post = create_facebook_post(headline, content, hashtags)

        video_results.append({
            "headline": headline,
            "content": content,
            "hashtags": hashtags,
            "video_path": video_path,
            "facebook_post": facebook_post
        })

        print(f"✅ Video and post created for: {headline[:50]}...")

    if save_data:
        with open("data/video_results.json", "w") as f:
            json.dump(video_results, f, indent=4)

    return video_results

# ------------------ Main Execution ------------------
if __name__ == "__main__":
    ensure_directories()
    
    print("🚀 Starting automated video and social media generation...")
    
    # Step 1: Fetch headlines and content
    print("\n📰 Step 1: Fetching trending news...")
    fetch_and_save_headlines_and_texts(limit=1)
    
    # Step 2: Generate unified content and hashtags
    print("\n✍️ Step 2: Generating AI content and hashtags...")
    generate_content_and_hashtags()
    
    # Step 3: Fetch related images
    print("\n🖼️ Step 3: Fetching related images...")
    fetch_images_and_save()
    
    # Step 4: Create videos and Facebook posts
    print("\n🎬 Step 4: Creating videos and Facebook posts...")
    results = create_videos_and_posts()
    
    print(f"\n🎉 Complete! Generated {len(results)} videos and Facebook posts.")
    print("📁 Check 'output_videos' folder for videos and 'data/video_results.json' for Facebook posts.")
    