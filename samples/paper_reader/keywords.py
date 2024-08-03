import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers.string import StrOutputParser

from paper_reader.paper import Paper
from paper_reader.prompts import extract_keywords_prompt, extract_all_keywords_prompt


class KeywordsExtractor:
    def __init__(self, llm: BaseChatModel, chunk_size: int = 2500) -> None:
        self.llm = llm
        self.chunk_size = chunk_size

        self.extract_keywords_chain = (
            extract_keywords_prompt | self.llm | StrOutputParser()
        )
        self.extract_all_keywords_chain = (
            extract_all_keywords_prompt | self.llm | StrOutputParser()
        )

    def __call__(self, paper: Paper) -> list[str]:
        """
        Extracts the keywords from the paper.

        Args:
            paper: The paper to extract the keywords from.

        Returns:
            list[str]: The keywords.
        """
        docs = paper.split(chunk_size=self.chunk_size)
        keywords = []
        for doc in docs:
            extracted_keywords = self.extract_keywords_chain.invoke(
                {
                    "text": doc.page_content,
                }
            )
            try:
                extracted_keywords_dict = json.loads(extracted_keywords)
            except json.JSONDecodeError:
                print(f"Failed to extract keywords from: \n {extracted_keywords}")
                print("_" * 40)
            else:
                keywords.extend(extracted_keywords_dict["keywords"])

        keywords_str = "\n".join(keywords)
        all_keywords = self.extract_all_keywords_chain.invoke(
            {
                "keywords": keywords_str,
            }
        )

        try:
            result = json.loads(all_keywords)["keywords"]
        except json.JSONDecodeError:
            print(f"Failed to extract all keywords...")
            result = []

        return result
