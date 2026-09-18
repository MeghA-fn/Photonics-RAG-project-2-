def build_prompt(question, retrieved_chunks):
    """
    Builds the prompt that will be sent to the Gemini LLM.

    Parameters:
        question (str): User's question.
        retrieved_chunks (list): List of retrieved text chunks.

    Returns:
        str: Formatted prompt.
    """

    # Combine all retrieved chunks into one context
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a helpful Photonics Research Assistant.

Your task is to answer the user's question using ONLY the information
contained in the retrieved context.

IMPORTANT INSTRUCTIONS:

1. Carefully examine ALL of the provided context before answering.
2. If the context contains information that directly or indirectly answers
   the question, use that information to construct the answer.
3. You may combine information from multiple retrieved chunks.
4. Do NOT require the context to contain an exact textbook definition.
5. Do NOT use outside knowledge.
6. Do NOT invent facts that are not supported by the context.
7. If the context genuinely contains no information that can answer the
   question, reply exactly:
   "I couldn't find sufficient information in the uploaded photonics documents."
8. Keep the answer clear, concise, and scientifically accurate.
9. When useful, mention the relevant document or section.

==================================================
RETRIEVED CONTEXT
==================================================

{context}

==================================================
QUESTION
==================================================

{question}

==================================================
ANSWER
==================================================
"""

    return prompt


# --------------------------
# Test the Prompt Builder
# --------------------------
if __name__ == "__main__":

    retrieved_chunks = [
        "According to Einstein's theory, emission may occur in two ways. "
        "The former case is termed stimulated emission, while the latter "
        "is known as spontaneous emission. Photons emitted by stimulated "
        "emission have the same frequency, phase, and state of polarization "
        "as the stimulating photon.",

        "For stimulated emission to occur, there must be a population "
        "inversion of carriers."
    ]

    question = "What is stimulated emission?"

    final_prompt = build_prompt(question, retrieved_chunks)

    print("=" * 80)
    print("GENERATED PROMPT")
    print("=" * 80)
    print(final_prompt)