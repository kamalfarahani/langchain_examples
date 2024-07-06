import uuid

from operator import itemgetter

from colorama import Fore
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory


HISTORY_TOKEN_LENGTH = 1000


class MessageHistoryStore:
    def __init__(self, store: dict[str, BaseChatMessageHistory] | None = None) -> None:
        """
        Initializes the message history store.

        Args:
            store: The message history store.
        """
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

        return self.store[session_id]


def make_session_id() -> str:
    """
    Returns a session id.

    Returns:
        (str): The session id.
    """
    return str(uuid.uuid4())


def make_config() -> dict:
    """
    Returns a config.

    Returns:
        (dict): The config dictionary.
    """
    return {"configurable": {"session_id": make_session_id()}}


def start_chat(llm: BaseChatModel) -> None:
    """
    Starts the chat.

    Args:
        llm: The LLM to use.

    Returns:
        None
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant named Doodoo. Answer all questions to the best of your ability in {language}.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    trimmer = trim_messages(
        max_tokens=HISTORY_TOKEN_LENGTH,
        strategy="last",
        token_counter=llm,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )

    chain = (
        RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
        | prompt
        | llm
        | StrOutputParser()
    )

    message_store = MessageHistoryStore(store={})
    model = RunnableWithMessageHistory(
        chain,
        message_store,
        input_messages_key="messages",  # This is needed because we sat `messages` the `variable_name` in the prompt template
    )

    config = make_config()
    language = input(f"{Fore.BLUE}>> Enter language: {Fore.RESET}")
    while True:
        text = input(f"{Fore.BLUE}>> Enter text: {Fore.RESET}")
        stream = model.stream(
            {"messages": [HumanMessage(content=text)], "language": language},
            config=config,
        )

        for token in stream:
            print(f"{Fore.CYAN}{token}", end="", flush=True)

        print(Fore.RESET)


def main():
    model_name = input("Enter model name: ")
    llm = ChatOllama(model=model_name)
    start_chat(llm)


if __name__ == "__main__":
    main()
