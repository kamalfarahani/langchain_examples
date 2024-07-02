from pathlib import Path
from collections.abc import Iterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.documents.base import Document
from langchain_chroma import Chroma

from doc_assitant.history import MessageHistoryStore, make_history_config
from doc_assitant.utils import DocumentPages, load_pdf_documents
from doc_assitant.prompts import user_chat_prompt, summarize_prompt, answer_prompt


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
        self.history_config = make_history_config()
        self.history_store = MessageHistoryStore(token_counter=self.llm)
        self.session_id = self.history_config["configurable"]["session_id"]
        self.chatbot = RunnableWithMessageHistory(
            self.chain,  # type: ignore
            self.history_store,
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
            last_summary = "No summary yet."
            for page in doc:
                new_summary = summarizer.invoke(
                    {"page": page.page_content, "last_summary": last_summary},
                )

                last_summary = new_summary

            metadata = {
                "source": doc[0].metadata.get("source", "No source."),
            }

            return Document(page_content=last_summary, metadata=metadata)

        return [summarize(doc) for doc in documents]

    def find_relevant_documents(self, question: str) -> list[Document]:
        """
        Asks the question.

        Args:
            question: The question to ask.

        Returns:
            (str): The answer.
        """
        return self.retriever.invoke(question)

    def ask_from_document(self, question: str, document_path: Path) -> str:
        """
        Asks the question from the document.

        Args:
            question: The question to ask.
            document_path: The document to ask from.

        Returns:
            (str): The answer.
        """
        chain = answer_prompt | self.llm | StrOutputParser()
        pages = PyPDFLoader(str(document_path)).load()
        current_information = "No information yet."
        for page in pages:
            current_information = chain.invoke(
                {
                    "question": question,
                    "new_page_content": page.page_content,
                    "current_information": current_information,
                }
            )

        self.history_store.add_user_message_to_history(self.session_id, question)

        self.history_store.add_ai_message_to_history(
            self.session_id, current_information
        )

        return current_information

    def chat(self, question: str) -> Iterator[str]:
        """
        Chats with the question.

        Args:
            question: The question to chat with.

        Returns:
            (str): The answer.
        """
        rag = {
            "question": RunnablePassthrough(),
            "context": self.retriever,
        } | self.chatbot

        result = rag.stream(question, config=self.history_config)

        return result
