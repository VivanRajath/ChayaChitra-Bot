import os
import logging
from flask import Flask, request
from telegram import Update, Bot
from telegram.ext import Dispatcher, CommandHandler, MessageHandler, Filters
from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO

# Tokens from environment variables
BOT_TOKEN = os.environ.get("BOT_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN")

# Logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📌 Using device: {device}")

# Load Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    use_auth_token=HF_TOKEN
).to(device)

# Flask app
app = Flask(__name__)
bot = Bot(token=BOT_TOKEN)
dispatcher = Dispatcher(bot, None, workers=0, use_context=True)

# Handlers
def start(update, context):
    user = update.message.from_user
    username = f"@{user.username}" if user.username else "no username"
    update.message.reply_text(f"👋 Hi {user.first_name}! Send me a prompt and I’ll generate an image for you.")
    print(f"🚀 Bot started by {user.first_name} ({username})")

def generate_image(update, context):
    prompt = update.message.text
    user = update.message.from_user
    username = f"@{user.username}" if user.username else "no username"
    print(f"📝 Prompt received from {user.first_name} ({username}): {prompt}")

    update.message.reply_text("⏳ Generating image... please wait!")
    image = pipe(prompt).images[0]

    bio = BytesIO()
    bio.name = "image.png"
    image.save(bio, "PNG")
    bio.seek(0)

    update.message.reply_photo(photo=bio, caption=f"🖼️ Prompt: {prompt}\n✅ Here's your image!")

dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, generate_image))

# Webhook endpoint
@app.route(f"/{BOT_TOKEN}", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), bot)
    dispatcher.process_update(update)
    return "ok"

@app.route("/", methods=["GET"])
def index():
    return "🤖 Bot is running!"

# Main
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
