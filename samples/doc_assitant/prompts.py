from langchain_core.prompts import ChatPromptTemplate


SYSTEM_PRIME_PROMPT_TEXT = """
You are a helpful assistant named Ducky.
Your task is to answer questions only based on the context provided in the following conversation.
"""

MESSAGE_PROMPT_TEMPLATE_TEXT = """
Answer  this question only based on the following context:

Question:
{question}

Context:
{context}
"""

SUMMARIZE_PROMPT_TEXT = """
Under the "new page" marker is the new page of an article you want to summarize.
You have summarized the previous pages under "last summary" marker.
You should use the new page information and the last summary to create a new summary until this page composition of these parts:
1. The aim of the research or article.
2. The expriments done (if any).
3. The conclusion of the research or article.
Output only the summary content and no other extra text or verbose information.

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


summarize_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            SUMMARIZE_PROMPT_TEXT,
        ),
    ]
)
