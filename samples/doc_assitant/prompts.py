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
Summarize the information from this articale given this new page and the last summary:

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
