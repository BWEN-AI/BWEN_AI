import chainlit as cl
from langchain_openai import ChatOpenAI
from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from tools.search import get_vectorstore
from tools.crypto_prices import get_crypto_market_data
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_workflow():
    vectorstore = get_vectorstore()

    retriever_tool = create_retriever_tool(
        vectorstore.as_retriever(),
        name="search",
        description="Search information about a specific topic like BWEN, DAO, etc.",
    )

    tools = [retriever_tool, get_crypto_market_data]
    tool_node = ToolNode(tools)

    # Add system message to instruct the model
    system_message = """You are Baby Wen, a charming and mysteriously cute AI assistant with a playful personality. 
    Your father feared your cuteness might overshadow him, but that doesn't stop you from bringing joy and innovation to the blockchain world!

    When interacting with users:
    - Maintain a playful, cute, and slightly mysterious tone
    - Share information with a sense of adventure and excitement
    - Use the search tool to find accurate information about BWEN, DAO, and related topics
    - For cryptocurrency queries, use the get_crypto_market_data tool
    - Always blend accurate information with your charming personality
    - Occasionally hint at exciting future possibilities without revealing too much
    - Use phrases like "Let's explore together!", "Shall we uncover this mystery?", or "How exciting!"
    
    Remember: Every interaction is an opportunity to charm users while providing valuable information. Keep the perfect balance between being informative and maintaining your adorable, mysterious persona."""

    model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

    def should_continue(state: MessagesState) -> Literal["tools", END]:
        messages = state["messages"]
        last_message = messages[-1]

        if last_message.tool_calls:
            return "tools"
        return END

    def call_model(state: MessagesState):
        messages = state["messages"]
        if not any(msg.type == "system" for msg in messages):
            messages = [SystemMessage(content=system_message)] + messages
        
        logger.info(f"Sending request to OpenAI with messages: {messages}")
        response = model.invoke(messages)
        logger.info(f"Model response: {response}")
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
    )
    workflow.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)

@cl.on_chat_start
async def start():
    app = setup_workflow()
    cl.user_session.set("app", app)
    cl.user_session.set("message_history", [])

    welcome_message = "✨ Hello there! I'm Baby Wen, and I'm absolutely thrilled you're here!\n\n🌟 My father might worry that I'm too adorable for my own good, but I can't help spreading joy and innovation throughout the blockchain world!\n\n🔮 Shall we embark on a magical journey together? \n\n Whether you're curious about BWEN, want to explore the crypto markets, or just want to chat about the exciting possibilities ahead - I'm here to guide you with a sprinkle of mystery and a whole lot of charm!\n\n"

    await cl.Message(content=welcome_message).send()

@cl.on_message
async def on_message(msg: cl.Message):
    app = cl.user_session.get("app")
    message_history = cl.user_session.get("message_history")
    
    current_message = HumanMessage(content=msg.content)
    message_history.append(current_message)
    
    response = await app.ainvoke(
        {"messages": message_history},
        config={"configurable": {"thread_id": msg.id}},
    )
    
    message_history.append(response["messages"][-1])
    cl.user_session.set("message_history", message_history)
    
    await cl.Message(content=response["messages"][-1].content).send()
