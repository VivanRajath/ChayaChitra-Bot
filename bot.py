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
import sys

# Check Python version compatibility
if sys.version_info >= (3, 13):
    print("⚠️ Warning: Python 3.13+ may have compatibility issues. Consider using Python 3.11 or 3.12")

BOT_TOKEN = os.getenv("BOT_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

# Validate environment variables
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN environment variable is required")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

# === Logging ===
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# === Hugging Face Inference Client ===
client = InferenceClient(model="runwayml/stable-diffusion-v1-5", token=HF_TOKEN)

# === Flask App ===
flask_app = Flask(__name__)

# === Global Application Variable ===
application: Application = None

def create_application():
    """Create and configure the Telegram application"""
    global application
    try:
        application = ApplicationBuilder().token(BOT_TOKEN).build()
        logger.info("✅ Telegram application created successfully")
        return application
    except Exception as e:
        logger.error(f"❌ Failed to create Telegram application: {e}")
        raise

# === Bot Handlers ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    await update.message.reply_text(
        "👋 Hi! Send me a text prompt and I'll generate an image using Stable Diffusion!\n\n"
        "Just type your description and I'll create an image for you!"
    )

async def generate_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate image from text prompt"""
    prompt = update.message.text.strip()
    
    if not prompt:
        await update.message.reply_text("Please provide a text prompt for image generation.")
        return
    
    logger.info(f"📝 Prompt received: {prompt}")
    
    # Send initial response
    status_message = await update.message.reply_text("⏳ Generating image... please wait!")

    try:
        # Generate image
        response = client.text_to_image(prompt)
        image = Image.open(io.BytesIO(response))

        # Prepare image for sending
        bio = io.BytesIO()
        bio.name = "generated_image.png"
        image.save(bio, "PNG")
        bio.seek(0)

        # Delete status message and send image
        await status_message.delete()
        await update.message.reply_photo(
            photo=bio, 
            caption=f"🖼️ Generated from: \"{prompt}\""
        )
        
        logger.info(f"✅ Image generated successfully for prompt: {prompt}")
        
    except Exception as e:
        logger.error(f"❌ Error generating image: {e}")
        await status_message.edit_text(
            "⚠️ Sorry, something went wrong while generating the image. Please try again with a different prompt."
        )

# === Flask Web Server ===
@flask_app.route("/", methods=["GET"])
def index():
    return "🤖 Telegram Image Generator Bot is live!"

@flask_app.route("/health", methods=["GET"])
def health():
    return {"status": "healthy", "bot": "running"}

def start_flask():
    """Start Flask web server"""
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"🌐 Starting Flask server on port {port}")
    flask_app.run(host="0.0.0.0", port=port, debug=False)

# === Main Bot Function ===
async def start_bot():
    """Initialize and start the Telegram bot"""
    global application
    
    try:
        # Create application if not exists
        if application is None:
            create_application()
        
        # Add handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, generate_image))
        
        # Initialize and start
        await application.initialize()
        await application.start()
        
        logger.info("✅ Telegram bot started successfully")
        
        # Run polling
        await application.run_polling(
            drop_pending_updates=True,
            close_loop=False
        )
        
    except Exception as e:
        logger.error(f"❌ Error starting bot: {e}")
        raise
    finally:
        if application:
            try:
                await application.stop()
                await application.shutdown()
                logger.info("🛑 Bot stopped gracefully")
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")

# === Main Entrypoint ===
if __name__ == "__main__":
    try:
        # Start Flask server in background thread
        flask_thread = threading.Thread(target=start_flask, daemon=True)
        flask_thread.start()
        
        # Start bot
        asyncio.run(start_bot())
        
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)