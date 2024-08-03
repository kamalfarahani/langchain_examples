from langchain import hub
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers.string import StrOutputParser

from paper_reader.paper import Paper


class MapReduceSummarize:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm
        map_prompt = hub.pull("rlm/map-prompt")
        reduce_prompt = hub.pull("rlm/reduce-prompt")

        self.map_chain = map_prompt | self.llm | StrOutputParser()
        self.reduce_chain = reduce_prompt | self.llm | StrOutputParser()

    def __call__(self, paper: Paper) -> str:
        """
        Summarizes the paper.

        Args:
            paper: The paper to summarize.

        Returns:
            (str): The summary.
        """
        docs = paper.split()
        summaries = [self.map_chain.invoke({"docs": doc}) for doc in docs]
        summaries_str = "\n".join(summaries)

        return self.reduce_chain.invoke({"doc_summaries": summaries_str})
