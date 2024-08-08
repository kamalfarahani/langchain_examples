from langchain_core.prompts import PromptTemplate


IS_THERE_ABSTRACT_PROMPT_TEXT = """\
Given the following text check if there is an article abstract or summary in it.
If there is an abstract return true else return false.

Text:
{text}

Only output the boolean value in pure string format.
DO NOT output any other text like "Here is the boolean value".
"""


EXTRACT_PAPER_INFO_PROMPT_TEXT = """\
Given the first page of a paper extract the information in JSON format in the following schema:
{{
    "title": str,
    "authors": list[str],
    "year": int,
}}

If any of the information is missing just put `null` in the json.

Paper First Page:
{page}

Output the JSON data as a plain text string with no additional formatting, no code blocks, no markdown, and no additional text.
"""

EXTRACT_ABSTRACT_PROMPT_TEXT = """\
Given the first page extract the abstract in at least 200 and at most 500 words.

Paper First Page:
{page}

Only output the abstarct in pure string format.
DO NOT output any other text like "Here is the abstract".
"""

is_there_abstract_prompt = PromptTemplate.from_template(
    IS_THERE_ABSTRACT_PROMPT_TEXT,
)

extract_paper_info_prompt = PromptTemplate.from_template(
    EXTRACT_PAPER_INFO_PROMPT_TEXT,
)

extarct_abstract_prompt = PromptTemplate.from_template(
    EXTRACT_ABSTRACT_PROMPT_TEXT,
)
