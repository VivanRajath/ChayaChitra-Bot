import os
import logging
import torch
from io import BytesIO
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from diffusers import StableDiffusionPipeline
from PIL import Image

# Read tokens from environment variables
BOT_TOKEN = os.environ.get("BOT_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not BOT_TOKEN or not HF_TOKEN:
    raise EnvironmentError("Please set BOT_TOKEN and HF_TOKEN as environment variables.")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"📌 Using device: {device}")

# Load model
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16" if device == "cuda" else None,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    use_auth_token=HF_TOKEN
).to(device)

# /start command
def start(update, context):
    user = update.message.from_user
    update.message.reply_text(
        f"👋 Hi {user.first_name}! Send me a prompt and I’ll generate an image for you."
    )
    logger.info(f"🚀 Started by {user.first_name} (@{user.username or 'no username'})")

# Prompt handler
def generate_image(update, context):
    prompt = update.message.text.strip()
    user = update.message.from_user
    logger.info(f"📝 Prompt from {user.first_name}: {prompt}")

    update.message.reply_text("⏳ Generating image... please wait!")

    try:
        image = pipe(prompt).images[0]

        # Save using Pillow
        bio = BytesIO()
        bio.name = "image.png"
        image.save(bio, format="PNG")
        bio.seek(0)

        update.message.reply_photo(photo=bio, caption=f"🖼️ Prompt: {prompt}\n✅ Done!")
        logger.info("✅ Image sent.")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        update.message.reply_text("⚠️ Something went wrong while generating the image.")

# Main function
def main():
    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, generate_image))

    logger.info("✅ Bot is running...")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
