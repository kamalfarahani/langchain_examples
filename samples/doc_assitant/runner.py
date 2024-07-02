import time

import colorama

from pathlib import Path
from collections.abc import Iterator
from pathlib import Path

from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from doc_assitant.chatbot import Chatbot


WELCOME_MESSAGE = """\
Welcome to Doc Assitant!

This is a simple bot that can do the following:
- Find relevant documents based on the query
- Ask questions about the documents
- Summarize the documents
- Ask questions based on the summarized documents

Write one of these commands:

question: Ask questions about the documents.
find: Find relevant documents based on the query and then ask from entire document.
exit: Exit the program.
"""


class Runner:
    def __init__(
        self,
        llm: BaseChatModel,
        embeddings: Embeddings,
        documents_path: Path,
    ) -> None:
        """
        Initializes the runner.

        Args:
            llm: The LLM to use.
            embeddings: The embeddings to use.
            documents_path: The path to the documents.

        Returns:
            None
        """
        self.llm = llm
        self.embeddings = embeddings
        self.documents_path = documents_path

    def run(self) -> None:
        """
        Runs the runner.

        Args:
            None

        Returns:
            None
        """
        self.chatbot = Chatbot(
            llm=self.llm,
            embeddings=self.embeddings,
            documents_path=self.documents_path,
        )

        while True:
            print_mangenta(WELCOME_MESSAGE)
            mode = input_cyan("Enter mode: ")
            if mode == "exit":
                break
            elif mode == "find":
                self.find()
            elif mode == "question":
                self.ask()
            else:
                print_red("Invalid mode.")

    def find(self) -> None:
        """
        Finds the relevant documents.
        Then calls the ask_from_document method.

        Args:
            None

        Returns:
            None
        """
        question = input(f"{colorama.Fore.CYAN}Query: {colorama.Fore.RESET}")
        answer = self.chatbot.find_relevant_documents(question)
        for idx, doc in enumerate(answer):
            print_cyan(f"Document {idx + 1}:")
            print_yellow(f"Path: {doc.metadata['source']}")
            print_mangenta(doc.page_content)
            print_line()

        document_index = input_cyan("Enter document index: ")
        document_path = Path(answer[int(document_index) - 1].metadata["source"])
        self.ask_from_document(document_path)

    def ask_from_document(self, document_path: Path) -> None:
        """
        Asks the question from the document.

        Args:
            document_path: The document to ask from.

        Returns:
            None
        """
        while True:
            question = input_cyan("Enter question from document: ")
            if question == "exit":
                break
            answer = self.chatbot.ask_from_document(question, document_path)
            print_mangenta(answer)
            print_line()

    def ask(self) -> None:
        """
        Asks the question from chatbot with summarized documents.

        Args:
            None

        Returns:
            None
        """
        question = input_cyan("Enter question: ")
        answer = self.chatbot.chat(question)
        print_stream(answer)
        print_line()


def input_cyan(text: str) -> str:
    return input(f"{colorama.Fore.CYAN}{text}{colorama.Fore.RESET}")


def print_line() -> None:
    print("_" * 40)


def print_red(text: str) -> None:
    print(f"{colorama.Fore.RED}{text}{colorama.Fore.RESET}")


def print_mangenta(text: str) -> None:
    print(f"{colorama.Fore.MAGENTA}{text}{colorama.Fore.RESET}")


def print_yellow(text: str) -> None:
    print(f"{colorama.Fore.YELLOW}{text}{colorama.Fore.RESET}")


def print_cyan(text: str) -> None:
    print(f"{colorama.Fore.CYAN}{text}{colorama.Fore.RESET}")


def print_stream(stream: Iterator[str]) -> None:
    for word in stream:
        print(f"{colorama.Fore.MAGENTA}{word}{colorama.Fore.RESET}", end="", flush=True)

    print()
