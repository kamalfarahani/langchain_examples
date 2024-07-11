from pathlib import Path

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from doc_assitant import Runner


def main():
    """
    The main function for the CLI.
    """
    model = input("Enter ollama model name: ")
    llm = ChatOllama(model=model)
    embeddings = OllamaEmbeddings(model=model)

    runner = Runner(
        llm=llm,
        embeddings=embeddings,
        documents_path=Path("/home/kamal/Downloads"),
    )

    runner.run()
