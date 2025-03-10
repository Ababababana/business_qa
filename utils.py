# Doc texts split
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import uuid
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.tools.tavily_search import TavilySearchResults




def setup_qdrant(collection_name,embedding_model,embed_size):
    client = QdrantClient(":memory:")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embed_size, distance=Distance.COSINE),
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_model,
    )
    return vector_store


def setup_text_splitter(chunk_size=2000,chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter


def predict_from_crag(inputs, graph):
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state_dict = graph.invoke(
        {"question": inputs, "steps": []}, config
    )
    return {"response": state_dict["generation"], "steps": state_dict["steps"]}


def load_example_docs(url_list):
    docs=[]
    for url in url_list:
        loader = RecursiveUrlLoader(
            url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text
        )
        docs.append(*loader.load())
    return docs

