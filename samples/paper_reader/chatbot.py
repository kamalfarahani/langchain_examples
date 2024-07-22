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

from paper_reader.paper import Paper, split_paper
from paper_reader.prompts import chat_prompt, extract_keywords_prompt


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

    def setup_summarizer(self) -> None:
        """
        Sets up the summarizer.

        Args:
            None

        Returns:
            None
        """
        map_prompt = hub.pull("rlm/map-prompt")
        reduce_prompt = hub.pull("rlm/reduce-prompt")

        map_chain = LLMChain(
            llm=self.llm,
            prompt=map_prompt,
        )

        reduce_chain = LLMChain(
            llm=self.llm,
            prompt=reduce_prompt,
        )

        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain,
            document_variable_name="doc_summaries",
        )

        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=4000,
        )

        self.summarizer = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="docs",
            return_intermediate_steps=False,
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

    def summarize(self) -> str:
        """
        Summarizes the paper.

        Args:
            None

        Returns:
            (str): The summary.
        """
        docs = split_paper(self.paper)
        result = self.summarizer.invoke(docs)

        return result["output_text"]

    def extract_keywords(self) -> str:
        """
        Extracts the keywords from the paper.

        Args:
            None

        Returns:
            (list[str]): The keywords.
        """
        chain = extract_keywords_prompt | self.llm | StrOutputParser()
        summary = self.summarize()
        result = chain.invoke(
            {
                "summary": summary,
                "abstract": self.paper.abstract,
            }
        )

        return result
