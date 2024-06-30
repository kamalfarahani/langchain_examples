from pathlib import Path

from langchain_core.documents.base import Document
from langchain_community.document_loaders import PyPDFLoader


type DocumentPages = list[Document]


def load_pdf_documents(documents_path: Path) -> list[DocumentPages]:
    """
    Loads the documents from the documents path.

    Args:
        documents_path: The path to the documents.

    Returns:
        (list[DocumentPages]): The list of documents pages.
    """
    docs: list[list[Document]] = []
    for file_path in documents_path.glob("*.pdf"):
        loader = PyPDFLoader(str(file_path))
        docs.append(loader.load())

    return docs
