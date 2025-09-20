import os
import requests
import logging
import sys

# ---------------------- Configuration ----------------------
GRAPH_URL = "https://graph.facebook.com/v20.0"
PAGE_ACCESS_TOKEN = os.getenv("FACEBOOK_PAGE_ACCESS_TOKEN")
PAGE_ID = os.getenv("FACEBOOK_PAGE_ID")

# ---------------------- Logging Setup ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    stream=sys.stdout
)

# ---------------------- Facebook Poster Class ----------------------
class FacebookPoster:
    def __init__(self, access_token, page_id):
        self.base_url = GRAPH_URL
        self.page_access_token = access_token
        self.page_id = page_id

    def upload_video(self, video_path, caption):
        """Upload a video to the Facebook page with a caption."""
        if not os.path.exists(video_path):
            logging.warning(f"⚠️ File not found: {video_path}")
            return None

        url = f"{self.base_url}/{self.page_id}/videos"

        try:
            with open(video_path, "rb") as video_file:
                files = {"source": video_file}
                data = {
                    "description": caption,
                    "access_token": self.page_access_token
                }
                response = requests.post(url, files=files, data=data).json()

            if "id" in response:
                logging.info(f"✅ Uploaded video: {video_path} (Video ID: {response['id']})")
                return response["id"]
            else:
                logging.error(f"❌ Upload failed: {response}")
                return None

        except Exception as e:
            logging.exception("❌ Exception during video upload")
            return None

        def post_text(self, message):
            """Post a text-only status update to the Facebook page."""
            url = f"{self.base_url}/{self.page_id}/feed"
            data = {
                "message": message,
                "access_token": self.page_access_token
            }
            try:
                response = requests.post(url, data=data).json()
                if "id" in response:
                    logging.info(f"✅ Posted text to page {self.page_id} (Post ID: {response['id']})")
                    return response["id"]
                else:
                    logging.error(f"❌ Text post failed: {response}")
                    return None
            except Exception as e:
                logging.exception("❌ Exception during text post")
                return None

        def post_image(self, image_path, caption=None):
            """Upload an image with optional caption to the Facebook page."""
            if not os.path.exists(image_path):
                logging.warning(f"⚠️ File not found: {image_path}")
                return None
            url = f"{self.base_url}/{self.page_id}/photos"
            try:
                with open(image_path, "rb") as image_file:
                    files = {"source": image_file}
                    data = {"access_token": self.page_access_token}
                    if caption:
                        data["caption"] = caption
                    response = requests.post(url, files=files, data=data).json()
                if "id" in response:
                    logging.info(f"✅ Uploaded image: {image_path} (Photo ID: {response['id']})")
                    return response["id"]
                else:
                    logging.error(f"❌ Image post failed: {response}")
                    return None
            except Exception as e:
                logging.exception("❌ Exception during image upload")
                return None

# ---------------------- Convenience Function ----------------------

def upload_video(video_path, caption):
    """Convenience function to upload a video using env credentials."""
    if not PAGE_ACCESS_TOKEN or not PAGE_ID:
        logging.error("❌ FACEBOOK_PAGE_ACCESS_TOKEN or FACEBOOK_PAGE_ID not set in environment.")
        return None
    poster = FacebookPoster(PAGE_ACCESS_TOKEN, PAGE_ID)
    return poster.upload_video(video_path, caption)

def post_text(message):
    """Convenience function to post text using env credentials."""
    if not PAGE_ACCESS_TOKEN or not PAGE_ID:
        logging.error("❌ FACEBOOK_PAGE_ACCESS_TOKEN or FACEBOOK_PAGE_ID not set in environment.")
        return None
    poster = FacebookPoster(PAGE_ACCESS_TOKEN, PAGE_ID)
    return poster.post_text(message)

def post_image(image_path, caption=None):
    """Convenience function to post image using env credentials."""
    if not PAGE_ACCESS_TOKEN or not PAGE_ID:
        logging.error("❌ FACEBOOK_PAGE_ACCESS_TOKEN or FACEBOOK_PAGE_ID not set in environment.")
        return None
    poster = FacebookPoster(PAGE_ACCESS_TOKEN, PAGE_ID)
    return poster.post_image(image_path, caption)
