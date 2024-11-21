import chainlit as cl
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import AsyncCallbackHandler

async def handle_message(msg: cl.Message, app, message_history):
    current_message = HumanMessage(content=msg.content)
    message_history.append(current_message)
    
    stream_message = cl.Message(content="")
    await stream_message.send()

    class ChainlitStreamingHandler(AsyncCallbackHandler):
        async def on_llm_new_token(self, token: str, **kwargs) -> None:
            await stream_message.stream_token(token)

    response = await app.ainvoke(
        {"messages": message_history},
        config={
            "configurable": {"thread_id": msg.id},
            "callbacks": [ChainlitStreamingHandler()]
        },
    )
    
    await stream_message.update()
    
    message_history.append(response["messages"][-1])
    return response["messages"][-1].content, message_history 