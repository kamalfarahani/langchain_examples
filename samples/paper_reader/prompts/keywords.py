from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


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


extract_keywords_prompt = PromptTemplate.from_template(
    EXTRACT_KEY_WORDS_PROMPT_TEXT,
)

extract_all_keywords_prompt = PromptTemplate.from_template(
    EXTRACT_ALL_KEYWORDS_PROMPT_TEXT,
)

extract_gist_keywords_prompt = PromptTemplate.from_template(
    EXTRACT_GIST_KEYWORDS_PROMPT_TEXT,
)
