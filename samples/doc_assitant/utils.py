from pathlib import Path


def load_documents(documents_path: Path) -> list[str]:
    """
    Loads the documents from the documents path.

    Args:
        documents_path: The path to the documents.

    Returns:
        (list[str]): The list of documents.
    """
    documents = []
    for file in documents_path.glob("*.md"):
        with open(file, "r") as f: