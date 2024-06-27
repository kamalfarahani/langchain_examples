from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.embeddings.embeddings import Embeddings
from langchain_chroma import Chroma

from doc_assitant.history import MessageHistoryStore
from doc_assitant.constants import SYSTEM_PRIME_PROMPT, MESSAGE_PROMPT_TEMPLATE


class Chatbot:
    def __init__(
        self, llm: BaseChatModel, embeddings: Embeddings, documents_path: Path
    ) -> None:
        self.llm = llm
        self.embeddings = embeddings
        self.setup_chain()

    def setup_chain(self) -> None:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    SYSTEM_PRIME_PROMPT,
                ),
                (
                    "user",
                    MESSAGE_PROMPT_TEMPLATE,
                ),
            ]
        )

        self.chain = prompt | self.llm | StrOutputParser()

    def setup_chatbot(self) -> None:
        history_store = MessageHistoryStore()
        self.chatbot = RunnableWithMessageHistory(
            self.chain,
            history_store,
            input_messages_key="question",
        )

    def setup_retriver(self) -> None:
        documents = load_
        vectorstore = Chroma.from_documents(documents, embedding=self.embeddings)
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 1},
        )
