from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain.storage import InMemoryStore
from langchain_community.retrievers import BM25Retriever


def create_splitters():
    parent = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=2000, chunk_overlap=100)
    child = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=200, chunk_overlap=30)
    return parent, child


def initialize_vectorstore(docs, embeddings):
    return Chroma.from_documents(docs, embeddings)


def build_parent_document_retriever(vectorstore, parent_splitter, child_splitter):
    store = InMemoryStore()
    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )


def build_keyword_retriever(docs):
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = 3
    return retriever


def build_ensemble_retriever(parent, keyword):
    return EnsembleRetriever(retrievers=[parent, keyword], weights=[0.6, 0.4])