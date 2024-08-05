from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


EXTRACT_KEY_WORDS_PROMPT_TEXT = """\
Given the abstract and summary of a paper extract the most important technical keywords in json format.
Give output in the following schema:
{{
    "keywords": list[str]
}}

Do not extarct keywords in fully abbrivated form, instead give the keywords in their full form.

Abstract:
{abstract}

Summary:
{summary}

Only output the json data in pure string format.
DO NOT output any other text like "Here is the json data".
"""

extract_keywords_prompt = PromptTemplate.from_template(
    EXTRACT_KEY_WORDS_PROMPT_TEXT,
)
