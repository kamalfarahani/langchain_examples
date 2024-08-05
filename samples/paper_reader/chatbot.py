from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings.embeddings import Embeddings
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_chroma import Chroma

from paper_reader.paper import Paper
from paper_reader.summarize import MapReduceSummarize
from paper_reader.keywords import KeywordsExtractor
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
        self.setup_summarizer()
        self.setup_keywords_extractor()

    def setup_retriever(self) -> None:
        """
        Sets up the retriever.

        Args:
            None

        Returns:
            None
        """
        docs = self.paper.split()
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

    def setup_summarizer(self) -> None:
        """
        Sets up the summarizer.

        Args:
            None

        Returns:
            None
        """
        self.summarizer = MapReduceSummarize(self.llm)

    def setup_keywords_extractor(self) -> None:
        """
        Sets up the keywords extractor.

        Args:
            None

        Returns:
            None
        """
        self.keywords_extractor = KeywordsExtractor(self.llm)

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

    def summarize(self) -> str:
        """
        Summarizes the paper.

        Args:
            None

        Returns:
            (str): The summary.
        """
        return self.summarizer(self.paper)

    def extract_keywords(self) -> list[str]:
        """
        Extracts the keywords from the paper.

        Args:
            None

        Returns:
            (list[str]): The keywords.
        """
        return self.keywords_extractor(self.paper)
