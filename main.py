import os
import random
from instabot import Bot
from transformers import BlipProcessor, BlipForConditionalGeneration
from textblob import TextBlob

# Import templates from templates.py
from templates import SPECIAL_TEMPLATES, DEFAULT_CAPTION_TEMPLATES, DEFAULT_HASHTAGS, THEME_HASHTAGS

# Paths
IMAGE_FOLDER = "images"
INSTAGRAM_USERNAME = "prompteng123"
INSTAGRAM_PASSWORD = "promptproject"

def select_random_image(image_folder):
    """Select a random image from the specified folder."""
    images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith((".jpg", ".png"))]
    if not images:
        raise FileNotFoundError("No images found in the folder!")
    return random.choice(images)

def generate_caption_blip(image_path):
    """Generate a caption for the image using BLIP."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Open and process the image
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    # Generate caption
    caption_ids = model.generate(**inputs)
    caption = processor.decode(caption_ids[0], skip_special_tokens=True)
    print("Caption: ", caption)
    return caption

def select_special_template(caption):
    """Select a special template if the caption matches a theme."""
    for theme, templates in SPECIAL_TEMPLATES.items():
        if theme in caption.lower():
            template = random.choice(templates)
            return template.format(labels=theme)
    return None

def generate_custom_hashtags(caption):
    """Generate custom hashtags based on the caption or default hashtags."""
    blob = TextBlob(caption)
    keywords = blob.noun_phrases

    hashtags = []
    for keyword in keywords:
        for theme, theme_tags in THEME_HASHTAGS.items():
            if theme in keyword.lower():
                hashtags.extend(theme_tags)

    # Use default hashtags if no relevant ones are found
    if not hashtags:
        hashtags = DEFAULT_HASHTAGS

    # Remove duplicates and limit to 10 hashtags
    hashtags = list(set(hashtags))
    return " ".join(hashtags[:10])

def generate_caption_and_hashtags(image_path):
    """Generate a caption and hashtags with fallback options."""
    try:
        # Generate a caption using BLIP
        caption = generate_caption_blip(image_path)
    except Exception:
        caption = ""

    # Attempt to use a special template
    special_caption = select_special_template(caption)

    # If no special template matches, use a default caption
    if not special_caption:
        special_caption = random.choice(DEFAULT_CAPTION_TEMPLATES)

    hashtags = generate_custom_hashtags(special_caption)
    return special_caption, hashtags

def post_to_instagram(image_path, caption, hashtags):
    """Post the image to Instagram with the caption and hashtags."""
    bot = Bot()
    try:
        bot.login(username=INSTAGRAM_USERNAME, password=INSTAGRAM_PASSWORD)
        full_caption = f"{caption}\n\n{hashtags}"
        bot.upload_photo(image_path, caption=full_caption)
    finally:
        bot.logout()

# Main script
if __name__ == "__main__":
    try:
        # Select an image
        image_path = select_random_image(IMAGE_FOLDER)

        # Generate caption and hashtags
        caption, hashtags = generate_caption_and_hashtags(image_path)

        print("Generated Caption:", caption)
        print("Generated Hashtags:", hashtags)

        # Post to Instagram
        post_to_instagram(image_path, caption, hashtags)
    except Exception as e:
        print("Error:", str(e))
