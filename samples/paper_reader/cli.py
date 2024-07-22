import colorama

from pathlib import Path

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

from paper_reader.paper import extract_paper_info, load_paper
from paper_reader.chatbot import Chatbot


def main():
    model = input("Enter ollama model name: ")
    llm = ChatOllama(model=model, temperature=0)
    embeddings = OllamaEmbeddings(model=model)

    paper_path = Path(input("Enter paper path: "))
    paper_info = extract_paper_info(paper_path=paper_path, llm=llm)

    print("The paper abstract is:")
    print_blue(paper_info["abstract"])

    for key, value in paper_info.items():
        if value is None:
            paper_info[key] = "N/A"

    paper = load_paper(
        paper_path=paper_path,
        title=paper_info["title"],
        authors=paper_info["authors"],
        year=paper_info["year"],
        abstract=paper_info["abstract"],
        url="https://example.com/paper",
    )

    chatbot = Chatbot(
        llm=llm,
        embeddings=embeddings,
        paper=paper,
    )

    while True:
        question = input_cyan("Enter question: ")

        if question == "exit":
            break
        elif question == "clear":
            print("\033[2J\033[;H")
            continue
        elif question == "summarize":
            summary = chatbot.summarize()
            print_mangenta(summary)
            continue
        elif question == "keywords":
            keywords = chatbot.extract_keywords()
            print_mangenta(keywords)
            continue

        answer = chatbot.ask(question)
        print_mangenta(answer)


def input_cyan(text: str) -> str:
    return input(f"{colorama.Fore.CYAN}{text}{colorama.Fore.RESET}")


def print_mangenta(text: str) -> None:
    print(f"{colorama.Fore.MAGENTA}{text}{colorama.Fore.RESET}")


def print_blue(text: str) -> None:
    print(f"{colorama.Fore.BLUE}{text}{colorama.Fore.RESET}")
