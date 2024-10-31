from typing import Annotated, Literal, TypedDict


from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.tools.retriever import create_retriever_tool
from tools.search import get_vectorstore


load_dotenv()

vectorstore = get_vectorstore()
print("Vectorstore initialized:", vectorstore)

retriever_tool = create_retriever_tool(
    vectorstore.as_retriever(search_kwargs={"k": 10}),  # Specify number of results to return
    name="search",
    description="Search information about a specific topic like BWEN, DAO, etc.",
)

tools = [retriever_tool]
tool_node = ToolNode(tools)

model = ChatOpenAI(model="gpt-4o").bind_tools(tools)
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tools"

    return END

def call_model(state: MessagesState):
    messages = state["messages"]
    print("Input messages:", messages)  # Debug input
    response = model.invoke(messages)
    print("Model response:", response)  # Debug output
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

app = workflow.compile()

# Modify test section with error handling
try:
    # Test retriever tool output
    test_output = retriever_tool.invoke("What is Baby Wen?")
    print("Retriever Tool Output:", test_output)
    print("Output type:", type(test_output))
    print("Output length:", len(str(test_output)))
    
    # Get number of documents in the retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    collection_data = vectorstore.get(include=["documents", "metadatas"])
    print(f"\nNumber of documents in vectorstore: {collection_data}")
    

except Exception as e:
    print("Error during retrieval:", e)
