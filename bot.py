import os
import logging
import asyncio
from flask import Flask, request
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
    AIORateLimiter,
    Application,
)
from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO
import threading

# === Environment Variables ===
BOT_TOKEN = os.environ.get("BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_HUGGINGFACE_TOKEN")

# === Logging ===
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

# === Setup Device ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📌 Using device: {device}")

# === Load Stable Diffusion Model ===
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        revision="fp16" if device == "cuda" else "main",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_auth_token=HF_TOKEN,
    ).to(device)
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# === Flask App ===
flask_app = Flask(__name__)

# === Telegram Bot Application ===
application: Application = ApplicationBuilder().token(BOT_TOKEN).rate_limiter(AIORateLimiter()).build()

# === Bot Handlers ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hi! Send me a prompt and I’ll generate an image for you.")
    print(f"🚀 Bot started by {update.effective_user.first_name}")

async def generate_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = update.message.text
    print(f"📝 Prompt received: {prompt}")
    await update.message.reply_text("⏳ Generating image... please wait!")

    try:
        image = pipe(prompt).images[0]
        bio = BytesIO()
        bio.name = "image.png"
        image.save(bio, "PNG")
        bio.seek(0)
        await update.message.reply_photo(photo=bio, caption=f"🖼️ Prompt: {prompt}\n✅ Here's your image!")
    except Exception as e:
        print(f"❌ Error generating image: {e}")
        await update.message.reply_text("⚠️ Sorry, something went wrong while generating the image.")

# === Register Bot Handlers ===
application.add_handler(CommandHandler("start", start))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, generate_image))

# === Flask Webhook Endpoint ===
@flask_app.route(f"/{BOT_TOKEN}", methods=["POST"])
async def webhook():
    update = Update.de_json(request.get_json(force=True), application.bot)
    await application.process_update(update)
    return "ok"

@flask_app.route("/", methods=["GET"])
def index():
    return "🤖 Bot is running!"

# === Main Entrypoint ===
def start_flask():
    port = int(os.environ.get("PORT", 5000))
    flask_app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    # Start Flask server in a separate thread
    threading.Thread(target=start_flask).start()

    async def start_bot():
        await application.initialize()
        await application.start()
        print("✅ Telegram bot started.")
        await application.updater.start_polling()
        await application.wait_until_shutdown()

    asyncio.run(start_bot())
