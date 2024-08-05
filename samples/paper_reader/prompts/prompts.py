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

EXTRACT_KEY_WORDS_PROMPT_TEXT = """\
Given the following text extract at most 5 scientific and techinical keywords from it.
Do not extract keywords that are abbreviations, such as "AI", "ML", "VR", "AR", etc. instead extract the full form of the word.
Give output in the following schema:
{{
    "keywords": list[str]
}}

Only output the json data in pure string format.
DO NOT output any other text like "Here is the json data".

Text:
{text}
"""

EXTRACT_ALL_KEYWORDS_PROMPT_TEXT = """\
Given the following list of keywords extract the unique keywords in the following schema:

{{
    "keywords": list[str]
}}

Only output the json data in pure string format.
DO NOT output any other text like "Here is the json data".

Keywords:
{keywords}
"""

EXTRACT_GIST_KEYWORDS_PROMPT_TEXT = """\
Given the following keywords from a paper extacrt at most 30 unique keywords that capture the main theme of the keywords.
Do not include general keywords in the output such as "Science", "Technology", "Computer Science", etc.
Include technical and specific keywords in the output.
Give output in the following schema:
{{
    "keywords": list[str]
}}

Only output the json data in pure string format.
DO NOT output any other text like "Here is the json data".

Keywords:
{keywords}
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
    EXTRACT_KEY_WORDS_PROMPT_TEXT,
)

extract_all_keywords_prompt = PromptTemplate.from_template(
    EXTRACT_ALL_KEYWORDS_PROMPT_TEXT,
)

extract_gist_keywords_prompt = PromptTemplate.from_template(
    EXTRACT_GIST_KEYWORDS_PROMPT_TEXT,
)
