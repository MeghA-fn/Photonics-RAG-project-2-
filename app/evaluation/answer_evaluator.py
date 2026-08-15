import subprocess
import json


OLLAMA_PATH = r"C:\Users\megha\AppData\Local\Programs\Ollama\ollama.exe"
MODEL_NAME = "gemma3:4b"


def evaluate_answer(question, context, answer):
    """
    Evaluate a Gemini-generated RAG answer using
    local Gemma 3:4B.

    Gemma is used ONLY as an evaluator.
    It does not generate the RAG answer.
    """

    evaluation_prompt = f"""
You are an evaluator for a Retrieval-Augmented Generation (RAG) system.

Your task is ONLY to evaluate the provided answer.
Do NOT answer the question yourself.
Do NOT use outside knowledge.

Evaluate the answer using ONLY the provided context.

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Evaluate the generated answer on these three criteria:

1. Faithfulness:
Is the answer supported by the retrieved context?

2. Relevance:
Does the answer directly address the question?

3. Completeness:
Does the answer include the important information available
in the retrieved context?

Give each criterion a score from 1 to 5.

Then calculate the average score.

Return ONLY valid JSON in exactly this format:

{{
    "faithfulness": 1,
    "relevance": 1,
    "completeness": 1,
    "overall_score": 1,
    "reason": "Brief explanation"
}}
"""

    result = subprocess.run(
        [
            OLLAMA_PATH,
            "run",
            MODEL_NAME,
            evaluation_prompt
        ],
        capture_output=True,
        text=True,
        encoding="utf-8"
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Ollama evaluation failed:\n{result.stderr}"
        )

    output = result.stdout.strip()

    # ---------------------------------------------------------
    # Clean Gemma output
    # ---------------------------------------------------------

    import re

    # Remove ANSI terminal escape sequences
    output = re.sub(
        r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])',
        '',
        output
    )

    # Remove Markdown code fences
    output = output.replace("```json", "")
    output = output.replace("```", "")

    output = output.strip()


    # ---------------------------------------------------------
    # Extract JSON
    # ---------------------------------------------------------

    try:

        start = output.find("{")
        end = output.rfind("}") + 1

        if start == -1 or end == 0:
            raise ValueError("No JSON object found.")

        json_text = output[start:end]


        # -----------------------------------------------------
        # Repair raw newlines inside JSON string values
        # -----------------------------------------------------

        repaired = []
        inside_string = False
        escaped = False

        for char in json_text:

            if char == '"' and not escaped:
                inside_string = not inside_string

            if char == "\n" and inside_string:
                repaired.append(" ")

            elif char == "\r" and inside_string:
                repaired.append(" ")

            else:
                repaired.append(char)

            if char == "\\" and not escaped:
                escaped = True
            else:
                escaped = False

        json_text = "".join(repaired)


        # -----------------------------------------------------
        #Parse JSON
        # -----------------------------------------------------

        evaluation = json.loads(json_text)

        return evaluation

    except (json.JSONDecodeError, ValueError) as e:

        return {
            "error": "Invalid evaluation response",
            "raw_output": output,
            "details": str(e)
        }


if __name__ == "__main__":

    question = "What is stimulated emission?"

    context = """
According to Einstein's theory, emission may occur in two ways:
stimulated emission and spontaneous emission. In stimulated emission,
an incoming photon induces the emission of another photon.
The emitted photon has the same frequency, phase and polarization
as the stimulating photon.
"""

    answer = """
Stimulated emission occurs when an incoming photon causes an excited
atom to emit another photon. The emitted photon has the same frequency,
phase and polarization as the incoming photon.
"""

    print("=" * 80)
    print("GEMMA ANSWER EVALUATION")
    print("=" * 80)

    evaluation = evaluate_answer(
        question,
        context,
        answer
    )

    print(json.dumps(
        evaluation,
        indent=4
    ))