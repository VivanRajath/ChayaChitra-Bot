import os
import logging
import asyncio
import nest_asyncio
nest_asyncio.apply()

from flask import Flask
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
    Application,
)
from huggingface_hub import InferenceClient
from PIL import Image
import io
import threading
import sys

# === Version Check ===
if sys.version_info >= (3, 13):
    print("⚠️ Python 3.13+ may have compatibility issues. Use Python 3.11 or 3.12.")

# === Tokens from Environment ===
BOT_TOKEN = os.getenv("BOT_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN is missing. Set it in Render's environment variables.")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing. Set it in Render's environment variables.")

# === Logging ===
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# === Hugging Face Client ===
client = InferenceClient(model="runwayml/stable-diffusion-v1-5", token=HF_TOKEN)

# === Flask App ===
flask_app = Flask(__name__)

@flask_app.route("/", methods=["GET"])
def index():
    return "🤖 Telegram Image Generator Bot is live on Render!"

@flask_app.route("/health", methods=["GET"])
def health():
    return {"status": "healthy", "bot": "running"}

def start_flask():
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"🌐 Starting Flask server on port {port}")
    flask_app.run(host="0.0.0.0", port=port, debug=False)

# === Telegram Handlers ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hi! Send me a text prompt and I'll generate an image using Stable Diffusion!"
    )

async def generate_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = update.message.text.strip()
    if not prompt:
        await update.message.reply_text("❗ Please provide a text prompt.")
        return

    logger.info(f"📝 Prompt received: {prompt}")
    status_msg = await update.message.reply_text("⏳ Generating image...")

    try:
        response = client.text_to_image(prompt)
        image = Image.open(io.BytesIO(response))

        bio = io.BytesIO()
        bio.name = "image.png"
        image.save(bio, "PNG")
        bio.seek(0)

        await status_msg.delete()
        await update.message.reply_photo(photo=bio, caption=f"🖼️ Prompt: \"{prompt}\"")
        logger.info("✅ Image sent successfully")
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        await status_msg.edit_text("⚠️ Failed to generate image. Try again later.")

# === Telegram Application ===
application: Application = None

def create_application():
    global application
    application = ApplicationBuilder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, generate_image))
    logger.info("✅ Telegram bot initialized")

async def start_bot():
    global application
    if application is None:
        create_application()

    await application.initialize()
    await application.start()
    logger.info("🤖 Telegram bot started")
    await application.run_polling()

# === Entrypoint ===
if __name__ == "__main__":
    try:
        flask_thread = threading.Thread(target=start_flask, daemon=True)
        flask_thread.start()

        loop = asyncio.get_event_loop()
        loop.create_task(start_bot())
        loop.run_forever()

    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)
