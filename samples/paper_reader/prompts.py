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
}}

If any of the information is missing just put `null` in the json.

Paper First Page:
{page}

Only output the json data in pure string format.
DO NOT output any other text like "Here is the json data".
"""

EXTRACT_ABSTRACT_PROMPT_TEXT = """\
Given the first page extract the abstract in at least 200 and at most 500 words.

Paper First Page:
{page}

Only output the abstarct in pure string format.
DO NOT output any other text like "Here is the abstract".
"""

EXTRACT_KEYWORDS_PROMPT_TEXT = """\
Given the abstract and the summary of the paper, extract the scientific and techinical from the paper.
The output should be in json format in the following schema:

{{
    "keywords": list[str]
}}

Only output the json data in pure string format.
DO NOT output any other text like "Here is the json data".

Abstract:
{abstract}

Summary:
{summary}
"""

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT_TEXT),
        ("human", "{input}"),
    ]
)

extract_paper_info_prompt = PromptTemplate.from_template(
    EXTRACT_PAPER_INFO_PROMPT_TEXT,
)

extarct_abstract_prompt = PromptTemplate.from_template(
    EXTRACT_ABSTRACT_PROMPT_TEXT,
)

extract_keywords_prompt = PromptTemplate.from_template(
    EXTRACT_KEYWORDS_PROMPT_TEXT,
)

extract_keywords_from_summary_prompt = PromptTemplate.from_template(
    EXTRACT_KEYWORDS_FROM_SUMMARY_PROMPT_TEXT,
)
