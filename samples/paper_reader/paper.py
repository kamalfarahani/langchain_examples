import json

from pathlib import Path
from typing import NamedTuple

from langchain_core.documents.base import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from paper_reader.prompts import extract_paper_info_prompt


class Paper(NamedTuple):
    title: str
    authors: list[str]
    year: int
    abstract: str
    url: str
    pages: list[Document]


def extract_paper_info(paper_path: Path, llm: BaseChatModel) -> dict:
    """
    Extracts the paper info from the paper path.

    Args:
        paper_path: The path to the paper.

    Returns:
        (dict): The paper info containing the title, authors, year and abstract.
    """
    loader = PyPDFLoader(str(paper_path))
    docs = loader.load()
    frist_page = docs[0]

    # Setup the extract info chain
    chain = extract_paper_info_prompt | llm | StrOutputParser()

    while True:
        # Extract the info from the paper
        json_data = chain.invoke(
            {"page": frist_page.page_content},
        )
        try:
            # Convert the json data to a dictionary
            data = json.loads(json_data)
        except Exception as e:
            print("Failed to extract the info from the paper.")
            print(e)
            print(json_data)
            print("Retrying...")
        else:
            break

    return data


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
