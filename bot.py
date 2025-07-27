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
    Application,
)
from huggingface_hub import InferenceClient
from PIL import Image
import io
import threading




BOT_TOKEN = os.getenv("BOT_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

# === Logging ===
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

# === Hugging Face Inference Client ===
client = InferenceClient(model="runwayml/stable-diffusion-v1-5", token=HF_TOKEN)

# === Flask App ===
flask_app = Flask(__name__)

# === Telegram Bot Application ===
application: Application = ApplicationBuilder().token(BOT_TOKEN).build()

# === Bot Handlers ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hi! Send me a prompt and I’ll generate an image using Stable Diffusion!")

async def generate_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = update.message.text
    print(f"📝 Prompt received: {prompt}")
    await update.message.reply_text("⏳ Generating image... please wait!")

    try:
        response = client.text_to_image(prompt)
        image = Image.open(io.BytesIO(response))

        bio = io.BytesIO()
        bio.name = "image.png"
        image.save(bio, "PNG")
        bio.seek(0)

        await update.message.reply_photo(photo=bio, caption=f"🖼️ Prompt: {prompt}\n✅ Here's your image!")
    except Exception as e:
        print(f"❌ Error generating image: {e}")
        await update.message.reply_text("⚠️ Sorry, something went wrong while generating the image.")

# === Register Handlers ===
application.add_handler(CommandHandler("start", start))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, generate_image))

# === Flask Web Server ===
@flask_app.route("/", methods=["GET"])
def index():
    return "🤖 Bot is live!"

def start_flask():
    port = int(os.environ.get("PORT", 10000))
    flask_app.run(host="0.0.0.0", port=port)

# === Main Entrypoint ===
if __name__ == "__main__":
    threading.Thread(target=start_flask).start()

    async def start_bot():
        await application.initialize()
        await application.start()
        print("✅ Telegram bot started.")
        await application.run_polling()  # ✅ CORRECT WAY
        await application.shutdown()

    asyncio.run(start_bot())
