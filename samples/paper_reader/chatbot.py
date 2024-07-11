from pathlib import Path
from collections.abc import Iterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.documents.base import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma

from paper_reader.paper import Paper, split_paper
from paper_reader.prompts import chat_prompt


class Chatbot:
    def __init__(
        self,
        llm: BaseChatModel,
        embeddings: Embeddings,
        paper: Paper,
        max_retrives_for_search=10,
    ) -> None:
        """
        Initializes the chatbot.

        Args:
            llm: The LLM to use.
            embeddings: The embeddings to use.
            documents_path: The path to the documents.
            max_retrives_for_search: The maximum number of retrives for search.

        Returns:
            None
        """
        self.llm = llm
        self.embeddings = embeddings
        self.paper = paper
        self.max_retrives_for_search = max_retrives_for_search

        self.setup_retriever()
        self.setup_chain()

    def setup_retriever(self) -> None:
        """
        Sets up the retriever.

        Args:
            None

        Returns:
            None
        """
        docs = split_paper(self.paper)
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
        )
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.max_retrives_for_search},
        )

    def setup_chain(self) -> None:
        """
        Sets up the chain.

        Args:
            None

        Returns:
            None
        """
        question_answer_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=chat_prompt,
        )

        self.chain = create_retrieval_chain(
            self.retriever,
            question_answer_chain,
        )

    def ask(self, question: str) -> str:
        """
        Asks the question.

        Args:
            question: The question to ask.

        Returns:
            (str): The answer.
        """
        result = self.chain.invoke(
            {
                "input": question,
                "abstract": self.paper.abstract,
            }
        )

        return result["answer"]
