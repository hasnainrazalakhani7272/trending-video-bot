
import os
from transformers import pipeline
from just_video import fetch_and_save_headlines_and_texts, generate_summaries_and_save

MODEL_NAME = os.getenv('LLM_MODEL_NAME', 'distilgpt2')  # Use open model for CI
# Note: device_map='auto' requires 'accelerate' package. Make sure to install it.
generator = pipeline('text-generation', model=MODEL_NAME, device_map='auto')

def generate_hashtags_llm(headline, summary):
    prompt = f"""
    You are a social media expert generating hashtags for Facebook videos.

    Headline: {headline}
    Summary: {summary}

    Generate 8 to 10 relevant, trending hashtags (each starting with #, no spaces).
    Output hashtags only separated by spaces.
    """
    result = generator(prompt, max_length=64, do_sample=True)
    hashtags = result[0]['generated_text'].strip().split()
    return hashtags

if __name__ == "__main__":
    # Fetch headlines and full texts
    headlines_full_texts = fetch_and_save_headlines_and_texts(limit=2, save_data=False)
    # Generate summaries
    summaries = generate_summaries_and_save(save_data=False)

    for article in headlines_full_texts:
        headline = article.get("headline")
        summary = summaries.get(headline, "")
        hashtags = generate_hashtags_llm(headline, summary)
        print(f"Headline: {headline}")
        print(f"Summary: {summary}")
        print(f"Generated Hashtags: {hashtags}\n")
