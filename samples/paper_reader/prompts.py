from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


SYSTEM_PROMPT_TEXT = """\
You are an assistant for question-answering based on an aricle with the following abstract:

Abstract:
{abstract}

Use the abstract and the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.

Context:
{context}
"""

EXTRACT_PAPER_INFO_PROMPT_TEXT = """\
Given the first page of a paper extract the information in json format in the following schema:
{{
    "title": str,
    "authors": list[str],
    "year": int,
    "abstract": str
}}

if any of the information is missing just put `null` in the json.

Paper First Page:
{page}

only output the json data in pure string format.
DO NOT output any other text like "Here is the json data".
"""


chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT_TEXT),
        ("human", "{input}"),
    ]
)

extract_paper_info_prompt = PromptTemplate.from_template(EXTRACT_PAPER_INFO_PROMPT_TEXT)
