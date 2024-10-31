import chainlit as cl
from langchain_core.messages import HumanMessage

async def handle_message(msg: cl.Message, app, message_history):
    current_message = HumanMessage(content=msg.content)
    message_history.append(current_message)
    
    response = await app.ainvoke(
        {"messages": message_history},
        config={"configurable": {"thread_id": msg.id}},
    )
    
    message_history.append(response["messages"][-1])
    return response["messages"][-1].content, message_history 