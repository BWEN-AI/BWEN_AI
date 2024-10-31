from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ObsidianLoader
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def create_vectorstore(persist_directory: str) -> Chroma:
    """Create a new vectorstore from documents."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    print("Creating new vectorstore...")
    loader = ObsidianLoader(os.path.join(project_root, "bwen_kb"))
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    return Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_directory,
    )


def get_vectorstore() -> Chroma:
    """Get the vectorstore."""
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "bwen-kb"
    
    embedding = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
    
    print("Connecting to Pinecone vectorstore...")
    return PineconeVectorStore(index_name=index_name, embedding=embedding)

vectorstore = get_vectorstore()
