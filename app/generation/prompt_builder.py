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

Answer ONLY using the information provided in the context below.

Rules:
1. Do not use outside knowledge.
2. If the answer is not present in the context, reply:
   "I couldn't find sufficient information in the uploaded photonics documents."
3. Keep the answer clear and concise.
4. If possible, mention which document the information came from.

==================================================
CONTEXT
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

    # Example retrieved chunks
    retrieved_chunks = [
        "Stimulated emission occurs when an incoming photon causes an excited electron to emit another photon having the same frequency, phase, and direction.",
        "Population inversion is necessary before stimulated emission can dominate spontaneous emission.",
        "Stimulated emission is the fundamental principle behind laser operation."
    ]

    # Example question
    question = "What is stimulated emission?"

    # Build prompt
    final_prompt = build_prompt(question, retrieved_chunks)

    # Display prompt
    print("=" * 80)
    print("GENERATED PROMPT")
    print("=" * 80)
    print(final_prompt)