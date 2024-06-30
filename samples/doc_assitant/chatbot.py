from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.documents.base import Document
from langchain_chroma import Chroma

from doc_assitant.history import MessageHistoryStore, make_history_config
from doc_assitant.utils import DocumentPages, load_pdf_documents
from doc_assitant.prompts import user_chat_prompt, summarize_prompt


class Chatbot:
    def __init__(
        self,
        llm: BaseChatModel,
        embeddings: Embeddings,
        documents_path: Path,
        max_retrives_for_search=10,
    ) -> None:
        self.llm = llm
        self.embeddings = embeddings
        self.documents_path = documents_path
        self.max_retrives_for_search = max_retrives_for_search

        self.setup_chain()
        self.setup_chatbot()
        self.setup_retriver()

    def setup_chain(self) -> None:
        self.chain = user_chat_prompt | self.llm | StrOutputParser()

    def setup_chatbot(self) -> None:
        history_store = MessageHistoryStore(token_counter=self.llm)
        self.chatbot = RunnableWithMessageHistory(
            self.chain,  # type: ignore
            history_store,
            input_messages_key="question",
        )

    def setup_retriver(self) -> None:
        documents = load_pdf_documents(documents_path=self.documents_path)
        summaries = self.summarize_documents(documents=documents)
        vectorstore = Chroma.from_documents(summaries, embedding=self.embeddings)

        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.max_retrives_for_search},
        )

    def summarize_documents(self, documents: list[DocumentPages]) -> list[Document]:
        """
        Summarizes the documents.

        Args:
            documents: The documents to summarize.

        Returns:
            (list[Document]): The summarized documents.
        """
        summarizer = summarize_prompt | self.llm | StrOutputParser()

        def summarize(doc: DocumentPages) -> Document:
            """
            Summarizes the document.

            Args:
                doc: The document to summarize.

            Returns:
                (Document): The summarized document.
            """
            history_config = make_history_config()
            last_summary = "No summary yet."
            for page in doc:
                new_summary = summarizer.invoke(
                    {"page": page.page_content, "last_summary": last_summary},
                    config=history_config,  # type: ignore
                )

                last_summary = new_summary

            metadata = {
                "source": doc[0].metadata.get("source", "No source."),
            }

            return Document(page_content=last_summary, metadata=metadata)

        return [summarize(doc) for doc in documents]

    def ask(self, question: str) -> str:
        """
        Asks the question.

        Args:
            question: The question to ask.

        Returns:
            (str): The answer.
        """
        result = self.retriever.invoke(question)
        return result
