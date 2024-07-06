from langchain_core.prompts import ChatPromptTemplate


SYSTEM_PRIME_PROMPT_TEXT = """
You are a helpful assistant named Ducky.
Your task is to answer questions only based on the context provided in the following conversation.
"""

MESSAGE_PROMPT_TEMPLATE_TEXT = """
Answer this question only based on the following context and conversation history.
If the question is not related to the context, search history if you got an answer from history state it.
If you got no answer from context nor history, explain that you don't know.
If the text below "Question" marker is not a question then response normally based on you knowledge.

Question:
{question}

Context:
{context}
"""

ANSWER_PROMPT_TEMPLATE_TEXT = """
You are trying to answer the following question from an article.
Use only and only the current information and new page content to answer the question.
If any part of the question is not available, in the current information or new page content, just state "I don't know".
Do not try to make up an answer. 

Question:
{question}

New page content:
{new_page_content}

Current information:
{current_information}
"""

SUMMARIZE_PROMPT_TEXT = """
Under the "new page" marker is the new page of an article you want to summarize.
You have summarized the previous pages under "last summary" marker.
Use the new page information and the last summary to create a new summary until this page composition of these parts:
1. The aim of the research or article in at leat 250 words and at most 500 words.
2. The expriments done (if any).
3. The conclusion of the research or article in at least 100 and at most 200 words.
Output only the summary content and no other extra text or verbose information.

If new page is just refereces or has no new information, just output last summary.

new page:
{page}

last summary:
{last_summary}
"""


user_chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_PRIME_PROMPT_TEXT,
        ),
        (
            "user",
            MESSAGE_PROMPT_TEMPLATE_TEXT,
        ),
    ]
)

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            ANSWER_PROMPT_TEMPLATE_TEXT,
        ),
    ]
)

summarize_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            SUMMARIZE_PROMPT_TEXT,
        ),
    ]
)
