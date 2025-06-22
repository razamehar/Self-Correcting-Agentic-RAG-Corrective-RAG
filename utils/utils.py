import re
from langchain_community.document_loaders import PyPDFLoader


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.encode("ascii", errors="ignore").decode()
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\n', ' ').strip()
    return text


def load_and_preprocess_documents(path):
    loader = PyPDFLoader(path)
    docs = loader.load()
    for doc in docs:
        doc.page_content = preprocess_text(doc.page_content)
    return docs