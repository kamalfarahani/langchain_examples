from langchain_core.prompts import PromptTemplate


EXTRACT_THEMES_PROMPT_TEXT = """\
"Given the following list of keywords extracted from scientific papers, identify the overarching theme or main area of advancement they collectively represent. Consider how these keywords might be connected in terms of technological, scientific, or social progress. Provide a concise summary describing the primary focus or trend indicated by these keywords.

Instructions:
Analyze the list of keywords to find commonalities, patterns, or related concepts.
Consider the broader context in which these keywords might be relevant, such as emerging technologies, innovative methodologies, or societal shifts.
Provide a brief summary (3-5 sentences) of the main theme or area of advancement that these keywords collectively suggest.

Example:
Keywords: "quantum computing," "superconducting qubits," "error correction," "quantum cryptography"
Main Theme: Advancements in quantum computing, particularly focusing on improving quantum hardware (superconducting qubits), error mitigation techniques, and the application of quantum principles in cryptography.
Please apply a similar analysis to the given keywords."

Keywords:
{keywords}

Only output the main theme in pure string format.
Do NOT output any other text like "Here is the main theme".
"""


extract_theme_prompt = PromptTemplate.from_template(
    EXTRACT_THEMES_PROMPT_TEXT,
)
