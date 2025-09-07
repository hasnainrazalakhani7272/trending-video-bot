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

# ---------------------- Convenience Function ----------------------
def upload_video(video_path, caption):
    """Convenience function to upload a video using env credentials."""
    if not PAGE_ACCESS_TOKEN or not PAGE_ID:
        logging.error("❌ FACEBOOK_PAGE_ACCESS_TOKEN or FACEBOOK_PAGE_ID not set in environment.")
        return None

    poster = FacebookPoster(PAGE_ACCESS_TOKEN, PAGE_ID)
    return poster.upload_video(video_path, caption)
