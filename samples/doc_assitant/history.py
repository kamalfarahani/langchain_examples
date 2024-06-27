from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import trim_messages


class MessageHistoryStore:
    def __init__(
        self,
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
            token_counter=None,
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

        return self.trimmer.invoke(self.store[session_id])
