import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents.base import Document

from paper_reader.paper import Paper
from paper_reader.prompts import (
    extract_paper_info_prompt,
    extarct_abstract_prompt,
    is_there_abstract_prompt,
)


class PaperInfoExtractor:
    def __init__(self, llm: BaseChatModel) -> None:
        """
        Initializes the paper info extractor.

        Args:
            llm: The LLM to use.

        Returns:
            None
        """
        self.llm = llm
        self.is_there_abstract_chain = (
            is_there_abstract_prompt | self.llm | StrOutputParser()
        )
        self.extract_paper_info_chain = (
            extract_paper_info_prompt | self.llm | StrOutputParser()
        )
        self.extract_abstract_chain = (
            extarct_abstract_prompt | self.llm | StrOutputParser()
        )

    def extract_info(self, pages: list[Document]) -> dict:
        """
        Extracts the information from the paper.

        Args:
            paper: The paper to extract the information from.

        Returns:
            (dict[str, str]): The information.
        """
        first_page = pages[0]
        info_str = self.extract_paper_info_chain.invoke(
            {
                "page": first_page,
            }
        )

        try:
            info = json.loads(info_str)
        except json.JSONDecodeError:
            info = {
                "title": "",
                "authors": [],
                "year": 0,
            }

        if info["year"] is None:
            info["year"] = 0
        if info["authors"] is None:
            info["authors"] = []
        if info["title"] is None:
            info["title"] = ""

        abstract = self.extarct_abstract(pages)
        info["abstract"] = abstract

        return info

    def extarct_abstract(self, pages: list[Document]) -> str:
        """
        Extracts the abstract from the paper.

        Args:
            paper: The paper to extract the abstract from.

        Returns:
            (str): The abstract.
        """
        for page in pages:
            is_there_abstract: str = self.is_there_abstract_chain.invoke(
                {
                    "text": page.page_content,
                }
            )

            if "true" in is_there_abstract.casefold():
                return self.extract_abstract_chain.invoke(
                    {
                        "page": page.page_content,
                    }
                )

        return ""
