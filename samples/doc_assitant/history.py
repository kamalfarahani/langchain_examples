import uuid

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.language_models.chat_models import BaseChatModel


class MessageHistoryStore:
    def __init__(
        self,
        token_counter: BaseChatModel,
        store: dict[str, BaseChatMessageHistory] | None = None,
        history_token_length: int = 1000,
    ) -> None:
        """
        Initializes the message history store.

        Args:
            store: The message history store.
            history_token_length: The maximum number of tokens in a message history.

        Returns:
            None
        """
        self.trimmer = trim_messages(
            max_tokens=history_token_length,
            strategy="last",
            token_counter=token_counter,
            include_system=True,
            allow_partial=False,
            start_on="human",
        )

        if store is None:
            store = {}
        self.store = store

    def __call__(self, session_id: str) -> BaseChatMessageHistory:
        """
        Returns the message history for the given session id.

        Args:
            session_id: The session id.

        Returns:
            (BaseChatMessageHistory): The message history.
        """
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()

        if len(self.store[session_id].messages) > 0:
            history = ChatMessageHistory(
                messages=self.trimmer.invoke(self.store[session_id].messages)
            )
        else:
            history = self.store[session_id]

        return history


def make_session_id() -> str:
    """
    Returns a session id.

    Returns:
        (str): The session id.
    """
    return str(uuid.uuid4())


def make_history_config() -> dict:
    """
    Returns a config.

    Returns:
        (dict): The config dictionary.
    """
    return {"configurable": {"session_id": make_session_id()}}
