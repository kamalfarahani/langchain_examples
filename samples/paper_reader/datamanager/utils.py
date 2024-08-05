from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from paper_reader.paper import Paper
from paper_reader.datamanager import DBManager
from paper_reader.keywords import KeywordsExtractor


def add_papers_to_db(
    llm: BaseChatModel,
    db_manager: DBManager,
    papers: list[Paper],
    category_name: str,
) -> None:
    """
    Adds the papers to the database.

    Args:
        llm: The language model.
        db_manager: The database manager.
        papers: The papers to add.
        category_name: The name of the category of the papers.

    Returns:
        None
    """
    keywords_extractor = KeywordsExtractor(llm=llm)
    for paper in papers:
        keywords = keywords_extractor(paper)
        db_manager.add_paper(
            paper=paper,
            keywords=keywords,
        )

        if db_manager.get_category_by_name(category_name) is None:
            db_manager.add_category_by_name(category_name)

        db_manager.add_paper_category(
            paper.url,
            category_name,
        )
