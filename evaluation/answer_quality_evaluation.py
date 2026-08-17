from pathlib import Path
import sys
import os
import re
import time
import subprocess
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT HYBRID RETRIEVER
# ============================================================

from app.retrieval.hybrid_retriever import hybrid_search


# ============================================================
# FILE PATHS
# ============================================================

QUESTIONS_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "rag_evaluation_questions.csv"
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "answer_quality_results.csv"
)


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set.\n"
        "Set it in PowerShell before running this script."
    )


genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-3.6-flash"

model = genai.GenerativeModel(MODEL_NAME)


# ============================================================
# SETTINGS
# ============================================================

TOP_K = 5

OLLAMA_EXE = (
    r"C:\Users\megha\AppData\Local\Programs\Ollama\ollama.exe"
)

OLLAMA_MODEL = "llama3.2:1b"


# ============================================================
# SAFE TEXT EXTRACTION
# ============================================================

def get_text(item):

    if not isinstance(item, dict):
        return ""

    return str(
        item.get("text")
        or item.get("document")
        or item.get("chunk")
        or ""
    )


def get_document(item):

    if not isinstance(item, dict):
        return ""

    return str(
        item.get("document_name")
        or item.get("source")
        or ""
    )


def get_page(item):

    if not isinstance(item, dict):
        return ""

    value = (
        item.get("page_number")
        or item.get("page")
        or ""
    )

    return str(value)


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):

    context_parts = []

    for rank, item in enumerate(
        results[:TOP_K],
        start=1
    ):

        text = get_text(item)
        document = get_document(item)
        page = get_page(item)

        if not text:
            continue

        context_parts.append(
            f"""
SOURCE {rank}

Document: {document}
Page: {page}

{text}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# GENERATE ANSWER USING GEMINI
# ============================================================

def generate_answer(question, context):

    prompt = f"""
You are a scientific assistant specialized in photonics.

Answer the user's question using ONLY the supplied context.

Do not use outside knowledge.

If the context does not contain enough information to answer
the question, say:

"I don't know based on the provided documents."

Keep the answer concise and scientifically accurate.

Question:
{question}

Context:
{context}

Answer:
"""

    try:

        response = model.generate_content(prompt)

        if not response or not response.text:
            return ""

        return response.text.strip()

    except Exception as e:

        print("\nERROR generating answer:")
        print(e)

        return ""


# ============================================================
# EVALUATE ANSWER USING LOCAL LLAMA
# ============================================================

def evaluate_with_llama(
    question,
    context,
    answer
):

    prompt = f"""
You are an evaluator for a scientific Photonics RAG system.

Evaluate the generated answer using ONLY the supplied context.

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Evaluate these four criteria:

1. FAITHFULNESS:
Is the answer supported by the retrieved context?

2. RELEVANCE:
Does the answer directly answer the question?

3. COMPLETENESS:
Does the answer include the important information available
in the retrieved context?

4. OVERALL:
Give an overall quality score.

Give every score from 1 to 5.

Return ONLY ONE LINE in EXACTLY this format:

FAITHFULNESS=5 RELEVANCE=5 COMPLETENESS=5 OVERALL=5

Rules:

- Replace the numbers with your actual scores.
- All scores must be integers from 1 to 5.
- Do not add explanations.
- Do not use JSON.
- Do not use markdown.
- Do not add any other text.
"""

    try:

        result = subprocess.run(
            [
                OLLAMA_EXE,
                "run",
                OLLAMA_MODEL,
                prompt
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180
        )

        raw_output = result.stdout.strip()

        print("\nLlama evaluation output:")
        print(raw_output)

        # ----------------------------------------------------
        # Extract four scores
        # ----------------------------------------------------

        faithfulness_match = re.search(
            r"FAITHFULNESS\s*=\s*([1-5])",
            raw_output,
            re.IGNORECASE
        )

        relevance_match = re.search(
            r"RELEVANCE\s*=\s*([1-5])",
            raw_output,
            re.IGNORECASE
        )

        completeness_match = re.search(
            r"COMPLETENESS\s*=\s*([1-5])",
            raw_output,
            re.IGNORECASE
        )

        overall_match = re.search(
            r"OVERALL\s*=\s*([1-5])",
            raw_output,
            re.IGNORECASE
        )

        faithfulness = (
            faithfulness_match.group(1)
            if faithfulness_match
            else ""
        )

        relevance = (
            relevance_match.group(1)
            if relevance_match
            else ""
        )

        completeness = (
            completeness_match.group(1)
            if completeness_match
            else ""
        )

        overall = (
            overall_match.group(1)
            if overall_match
            else ""
        )

        # ----------------------------------------------------
        # Validate extraction
        # ----------------------------------------------------

        if not all([
            faithfulness,
            relevance,
            completeness,
            overall
        ]):

            return {
                "faithfulness": "",
                "relevance": "",
                "completeness": "",
                "overall": "",
                "reason":
                    f"Could not parse Llama output: {raw_output}"
            }

        return {
            "faithfulness": faithfulness,
            "relevance": relevance,
            "completeness": completeness,
            "overall": overall,
            "reason": "Evaluated by Llama 3.2:1B."
        }

    except subprocess.TimeoutExpired:

        return {
            "faithfulness": "",
            "relevance": "",
            "completeness": "",
            "overall": "",
            "reason":
                "Llama evaluation timed out."
        }

    except Exception as e:

        return {
            "faithfulness": "",
            "relevance": "",
            "completeness": "",
            "overall": "",
            "reason":
                f"Llama evaluation error: {e}"
        }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("PHOTONICS RAG - ANSWER QUALITY EVALUATION")
    print("=" * 80)

    # --------------------------------------------------------
    # LOAD QUESTIONS
    # --------------------------------------------------------

    questions_df = pd.read_csv(
        QUESTIONS_FILE
    )

    print(
        f"\nLoaded {len(questions_df)} "
        f"evaluation questions."
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results = []

    processed = 0
    answers_generated = 0

    # --------------------------------------------------------
    # PROCESS QUESTIONS
    # --------------------------------------------------------

    for index, row in questions_df.iterrows():

        question = str(
            row["question"]
        ).strip()

        processed += 1

        print("\n")
        print("=" * 80)
        print(
            f"QUESTION {index + 1}/{len(questions_df)}"
        )
        print("=" * 80)

        print(
            f"\nQuestion: {question}"
        )

        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        try:

            retrieved_results = hybrid_search(
                question,
                top_k=TOP_K
            )

        except TypeError:

            try:

                retrieved_results = hybrid_search(
                    question,
                    TOP_K
                )

            except Exception as e:

                print(
                    f"\nRetrieval error: {e}"
                )

                retrieved_results = []

        except Exception as e:

            print(
                f"\nRetrieval error: {e}"
            )

            retrieved_results = []

        # ----------------------------------------------------
        # BUILD CONTEXT
        # ----------------------------------------------------

        context = build_context(
            retrieved_results
        )

        # ----------------------------------------------------
        # GENERATE ANSWER
        # ----------------------------------------------------

        print(
            "\nGenerating answer using Gemini..."
        )

        answer = generate_answer(
            question,
            context
        )

        if answer:

            answers_generated += 1

        print(
            f"\nAnswer:\n{answer}"
        )

        # ----------------------------------------------------
        # EVALUATE USING LLAMA
        # ----------------------------------------------------

        if answer:

            print(
                "\nEvaluating answer using "
                "Llama 3.2:1B..."
            )

            evaluation = evaluate_with_llama(
                question,
                context,
                answer
            )

        else:

            evaluation = {
                "faithfulness": "",
                "relevance": "",
                "completeness": "",
                "overall": "",
                "reason":
                    "Answer generation failed."
            }

        # ----------------------------------------------------
        # DISPLAY EVALUATION
        # ----------------------------------------------------

        print("\nANSWER QUALITY:")

        print(
            f"Relevance     : "
            f"{evaluation['relevance']}"
        )

        print(
            f"Completeness  : "
            f"{evaluation['completeness']}"
        )

        print(
            f"Faithfulness  : "
            f"{evaluation['faithfulness']}"
        )

        print(
            f"Overall       : "
            f"{evaluation['overall']}"
        )

        print(
            f"Reason        : "
            f"{evaluation['reason']}"
        )

        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        results.append(
            {
                "question":
                    question,

                "answer":
                    answer,

                "faithfulness":
                    evaluation["faithfulness"],

                "relevance":
                    evaluation["relevance"],

                "completeness":
                    evaluation["completeness"],

                "overall_score":
                    evaluation["overall"],

                "reason":
                    evaluation["reason"]
            }
        )

        # Small delay to avoid excessive requests
        time.sleep(1)

    # ========================================================
    # SAVE CSV
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    # ========================================================
    # CALCULATE AVERAGE SCORE
    # ========================================================

    numeric_scores = pd.to_numeric(
        results_df["overall_score"],
        errors="coerce"
    )

    valid_scores = numeric_scores.dropna()

    if len(valid_scores) > 0:

        average_score = (
            valid_scores.mean()
        )

        average_display = (
            f"{average_score:.2f}/5"
        )

    else:

        average_display = "N/A"

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 80)
    print(
        "ANSWER QUALITY EVALUATION COMPLETE"
    )
    print("=" * 80)

    print(
        f"Total questions       : "
        f"{len(questions_df)}"
    )

    print(
        f"Processed             : "
        f"{processed}"
    )

    print(
        f"Answers generated     : "
        f"{answers_generated}"
    )

    print(
        f"Average answer score  : "
        f"{average_display}"
    )

    print("\nResults saved to:")

    print(
        RESULTS_FILE
    )

    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()