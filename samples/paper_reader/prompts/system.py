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

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT_TEXT),
        ("human", "{input}"),
    ]
)
