import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ObsidianLoader
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

def create_vectorstore():
    """Create a new vectorstore from documents."""
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "bwen-kb"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))

    print("Creating new vectorstore...")
    loader = ObsidianLoader(os.path.join(project_root, "bwen_kb"))
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    PineconeVectorStore.from_documents(
        splits, embeddings, index_name=index_name
    )

def main():
    """Create and persist the vectorstore."""
    print("Starting vectorstore creation...")
    create_vectorstore()
    print("Vectorstore created in Pinecone")

if __name__ == "__main__":
    main()
