from langchain_core.prompts import PromptTemplate


EXTRACT_KEY_WORDS_PROMPT_TEXT = """\
Given the abstract and summary of a paper, extract the most important and specific technical and scientific keywords in JSON format. Focus on keywords that indicate specific technical or scientific trends, technologies, methods, or concepts.

Give output in the following schema:
{{
    "keywords": list[str]
}}

Criteria:
1. Keywords should be specific and not general terms.
2. Keywords should represent technical or scientific trends, technologies, methods, or concepts.
3. Do not extract keywords in fully abbreviated form; instead, provide the keywords in their full form.
4. Avoid extracting very general keywords or terms that do not indicate a specific trend or concept.

Abstract:
{abstract}

Summary:
{summary}

Only output the JSON data in pure string format.
DO NOT output any other text like "Here is the JSON data".

"""

extract_keywords_prompt = PromptTemplate.from_template(
    EXTRACT_KEY_WORDS_PROMPT_TEXT,
)
