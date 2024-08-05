import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers.string import StrOutputParser

from paper_reader.paper import Paper
from paper_reader.summarize import MapReduceSummarizer
from paper_reader.prompts import extract_keywords_prompt


class KeywordsExtractor:
    def __init__(self, llm: BaseChatModel, chunk_size: int = 2500) -> None:
        self.llm = llm
        self.chunk_size = chunk_size
        self.summarizer = MapReduceSummarizer(self.llm)

        self.extract_keywords_chain = (
            extract_keywords_prompt | self.llm | StrOutputParser()
        )

    def __call__(self, paper: Paper) -> list[str]:
        """
        Extracts the keywords from the paper.

        Args:
            paper: The paper to extract the keywords from.

        Returns:
            list[str]: The keywords.
        """
        abstract = paper.abstract
        summary = self.summarizer(paper)
        keywords = self.extract_keywords(abstract, summary)

        return keywords

    def extract_keywords(self, abstract: str, summary: str) -> list[str]:
        """
        Extracts the keywords from the paper.

        Args:
            abstract: The abstract of the paper.
            summary: The summary of the paper.

        Returns:
            list[str]: The keywords.
        """
        keywords_json = self.extract_keywords_chain.invoke(
            {
                "abstract": abstract,
                "summary": summary,
            }
        )

        try:
            keywords = json.loads(keywords_json)["keywords"]
        except json.JSONDecodeError:
            keywords = []

        return keywords
