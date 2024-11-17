import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import ObsidianLoader
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain.docstore.document import Document

load_dotenv()
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
index_name = "bwen-kb"

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

embeddings = OpenAIEmbeddings(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="text-embedding-3-large",
)
loader = ObsidianLoader(os.path.join(project_root, "bwen_kb"))
docs = loader.load()
index = pc.Index(index_name)
vector_store = PineconeVectorStore(
    index=index,  # Pass the index object, not the name
    embedding=embeddings,
    text_key="text",
    namespace="",
)

def add_documents():
    """Create a new vectorstore from documents."""
    for doc in docs:
        doc.metadata["path"] = ""
        if doc.metadata["source"] != "BWen FAQ.md":
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents([doc])

        vector_store.from_documents(splits, embeddings, index_name=index_name)

def add_bwen_faq():
    target_doc = next(doc for doc in docs if doc.metadata["source"] == "BWen FAQ.md")

    splits = target_doc.page_content.split("---")

    formatted_docs = [
        Document(
            page_content=split.strip(),
            metadata={
                "source": target_doc.metadata["source"],
                "path": "",
            },
        )
        for split in splits
    ]
    vector_store.from_documents(
        formatted_docs, embeddings, index_name=index_name
    )


def get_documents_by_source(source_file):
    """Retrieve all documents that match the specified source file using similarity search."""

    # Create a dummy query embedding to search with
    results = vector_store.similarity_search(
        "dummy query", k=10000, filter={"source": source_file}
    )
    return [match.id for match in results]


def main():
    """Create and persist the vectorstore."""
    # all documents
    # add_documents()

    # Delete files in BWen FAQ
    ids = get_documents_by_source("BWen FAQ.md")
    vector_store.delete(ids=ids)

    # Readd updated docs
    add_bwen_faq()

if __name__ == "__main__":
    main()
