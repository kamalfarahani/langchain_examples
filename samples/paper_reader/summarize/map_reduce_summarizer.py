from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.documents.base import Document

from paper_reader.paper import Paper
from paper_reader.prompts.summarize import map_summarize_prompt, reduce_summarize_prompt


class MapReduceSummarizer:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

        self.map_chain = map_summarize_prompt | self.llm | StrOutputParser()
        self.reduce_chain = reduce_summarize_prompt | self.llm | StrOutputParser()

    def __call__(
        self,
        paper: Paper,
        chunk_size: int = 6000,
    ) -> str:
        """
        Summarizes the paper.

        Args:
            paper: The paper to summarize.
            chunk_size: The size of the chunk.

        Returns:
            (str): The summary.
        """
        summaries = [
            self.summarize_chunk(chunk) for chunk in paper.split(chunk_size=chunk_size)
        ]

        return self.reduce_chain.invoke(
            {
                "summaries": summaries,
            }
        )

    def summarize_chunk(self, chunk: Document) -> str:
        """
        Summarizes the chunk.

        Args:
            chunk: The chunk to summarize.

        Returns:
            (str): The summary.
        """
        return self.map_chain.invoke(
            {
                "chunk": chunk.page_content,
            }
        )
