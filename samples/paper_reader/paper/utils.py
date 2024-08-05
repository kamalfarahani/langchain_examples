from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.document_loaders import PyPDFLoader

from paper_reader.paper import Paper, load_paper
from paper_reader.paper.info_exractor import PaperInfoExtractor


def load_paper_from_path(llm: BaseChatModel, path: Path) -> Paper:
    """
    Loads the paper from the path.

    Args:
        llm: The language model.
        path: The path to the paper.

    Returns:
        (Paper): The paper.
    """
    paper_info_extractor = PaperInfoExtractor(llm=llm)
    pdf_loader = PyPDFLoader(str(path))
    pages = pdf_loader.load()

    info = paper_info_extractor.extract_info(pages)
    paper = load_paper(
        paper_path=path,
        url=str(path),
        **info,
    )

    return paper
