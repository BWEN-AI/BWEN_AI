from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

if __name__ == "__main__":
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "bwen-kb"
    vector_store = PineconeVectorStore(index_name=index_name, embedding=OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY")))
    query = "What is BWEN?"

    retriever = vector_store.as_retriever(search_kwargs = {"k":3})
    print(retriever.get_relevant_documents(query))
    
    
