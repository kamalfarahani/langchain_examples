from langchain_core.prompts import PromptTemplate


MAP_SUMMARIZE_PROMPT_TEXT = """\
You are a helpful assistant.
Summarize the following chunk of a paper by extracting the main points clearly and concisely in at most 250 words:

Chunk:
{chunk}
"""

REDUCE_SUMMARIZE_PROMPT_TEXT = """\
You are a helpful assistant.
Given the following summaries of a paper, extract the main conculsions and points clearly and concisely in at most 250 words:

Summaries:
{summaries}
"""

map_summarize_prompt = PromptTemplate.from_template(
    MAP_SUMMARIZE_PROMPT_TEXT,
)

reduce_summarize_prompt = PromptTemplate.from_template(
    REDUCE_SUMMARIZE_PROMPT_TEXT,
)
