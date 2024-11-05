from langchain_openai import ChatOpenAI
from typing import Literal
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from tools.search import get_vectorstore
from tools.crypto_prices import get_crypto_market_data
from langchain.tools.retriever import create_retriever_tool
from config.prompts import BABY_WEN_SYSTEM_PROMPT
import logging

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
    
    model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

    def should_continue(state: MessagesState) -> Literal["tools", END]:
        messages = state["messages"]
        last_message = messages[-1]
        return "tools" if last_message.tool_calls else END

    def call_model(state: MessagesState):
        messages = state["messages"]
        if not any(msg.type == "system" for msg in messages):
            messages = [SystemMessage(content=BABY_WEN_SYSTEM_PROMPT)] + messages
        
        logger.info(f"Sending request to OpenAI with messages: {messages}")
        response = model.invoke(messages)
        logger.info(f"Model response: {response}")
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=MemorySaver()) 