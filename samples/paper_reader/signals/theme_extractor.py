from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers.string import StrOutputParser

from paper_reader.prompts import extract_theme_prompt


class ThemeExtractor:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm
        self.extarct_theme_chain = extract_theme_prompt | self.llm | StrOutputParser()

    def __call__(self, keywords: list[str]) -> str:
        """
        Extracts the theme from the keywords.

        Args:
            keywords: The keywords to extract the theme from.

        Returns:
            str: The theme.
        """
        theme = self.extarct_theme_chain.invoke(
            {
                "keywords": keywords,
            }
        )

        return theme
