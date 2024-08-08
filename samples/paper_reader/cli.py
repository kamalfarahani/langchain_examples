import colorama

from paper_reader.chatbot import Chatbot


def main(chatbot: Chatbot) -> None:
    """
    The main function for the CLI.

    Args:
        chatbot: The chatbot.

    Returns:
        None
    """
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
