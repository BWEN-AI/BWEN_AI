import chainlit as cl
from workflow.agent import setup_workflow
from utils.message_handler import handle_message
from config.prompts import WELCOME_MESSAGE
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@cl.on_chat_start
async def start():
    app = setup_workflow()
    cl.user_session.set("app", app)
    cl.user_session.set("message_history", [])
    await cl.Message(content=WELCOME_MESSAGE).send()

@cl.on_message
async def on_message(msg: cl.Message):
    app = cl.user_session.get("app")
    message_history = cl.user_session.get("message_history")
    
    response_content, updated_history = await handle_message(msg, app, message_history)
    
    cl.user_session.set("message_history", updated_history)
    await cl.Message(content=response_content).send()
