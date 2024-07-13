from pathlib import Path
from typing import NamedTuple

from langchain_core.documents.base import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Paper(NamedTuple):
    title: str
    authors: list[str]
    year: int
    abstract: str
    url: str
    pages: list[Document]


def make_paper_metadata(paper: Paper) -> dict:
    """
    Makes the paper metadata.

    Args:
        paper: The paper.

    Returns:
        (dict): The paper metadata.
    """
    return {
        "title": paper.title,
        "authors": str(paper.authors),
        "year": paper.year,
        "url": paper.url,
    }


def load_paper(
    paper_path: Path,
    title: str,
    authors: list[str],
    year: int,
    abstract: str,
    url: str,
) -> Paper:
    """
    Loads the paper from the paper path.

    Args:
        paper_path: The path to the paper.
        title: The title of the paper.
        authors: The authors of the paper.
        year: The year of the paper.
        abstract: The abstract of the paper.
        url: The url of the paper.

    Returns:
        (Paper): The paper.
    """
    loader = PyPDFLoader(str(paper_path))
    docs = loader.load()
    paper = Paper(
        title=title,
        authors=authors,
        year=year,
        abstract=abstract,
        url=url,
        pages=docs,
    )

    paper_metadata = make_paper_metadata(paper)
    for page in paper.pages:
        page.metadata.update(paper_metadata)

    return paper


def split_paper(
    paper: Paper,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Splits the paper into pages.

    Args:
        paper: The paper to split.

    Returns:
        (list[Document]): The list of spited contents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    splits = text_splitter.split_documents(paper.pages)

    return splits
