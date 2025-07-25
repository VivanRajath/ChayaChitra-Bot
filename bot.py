
import os
import logging
import asyncio
from flask import Flask, request
from telegram import Update, Bot
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    AIORateLimiter,
    Application,
    filters
)
from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO

# === Environment Variables ===
BOT_TOKEN = '7849179622:AAEnCXeNnJikeFJiirg8vZXLF6AOIf3oU-U'
HF_TOKEN = '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'

# === Logging ===
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# === Setup Device ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📌 Using device: {device}")

# === Load Stable Diffusion Model ===
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    use_auth_token=HF_TOKEN
).to(device)

# === Flask App ===
flask_app = Flask(__name__)

# === Telegram Bot Application ===
application: Application = ApplicationBuilder().token(BOT_TOKEN).rate_limiter(AIORateLimiter()).build()

# === Handlers ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    username = f"@{user.username}" if user.username else "no username"
    await update.message.reply_text(f"👋 Hi {user.first_name}! Send me a prompt and I’ll generate an image for you.")
    print(f"🚀 Bot started by {user.first_name} ({username})")

async def generate_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = update.message.text
    user = update.effective_user
    username = f"@{user.username}" if user.username else "no username"
    print(f"📝 Prompt received from {user.first_name} ({username}): {prompt}")

    await update.message.reply_text("⏳ Generating image... please wait!")
    image = pipe(prompt).images[0]

    bio = BytesIO()
    bio.name = "image.png"
    image.save(bio, "PNG")
    bio.seek(0)

    await update.message.reply_photo(photo=bio, caption=f"🖼️ Prompt: {prompt}\n✅ Here's your image!")

# Register handlers
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
if __name__ == "__main__":
    import threading

    # Start Flask app in a separate thread
    def run_flask():
        port = int(os.environ.get("PORT", 5000))
        flask_app.run(host="0.0.0.0", port=port)

    threading.Thread(target=run_flask).start()

    # Run the Telegram bot (polling fallback, can be removed if webhook-only)
    asyncio.run(application.initialize())
    asyncio.run(application.start())
    print("✅ Bot is running...")
    asyncio.get_event_loop().run_forever()
