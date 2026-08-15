
from pathlib import Path
import sys
import os
import re
import time
import json
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

# IMPORTANT:
# Gemini is used ONLY for answer generation.
# Gemma 3:4B is used ONLY for answer evaluation.
#
# Pipeline:
# Hybrid RRF -> Gemini -> Gemma evaluator

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
# EXTRACT SCORE
# ============================================================

def extract_score(text, field_name):

    pattern = rf"{field_name}\s*:\s*([1-5])"

    match = re.search(
        pattern,
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1)

    return ""


# ============================================================
# EXTRACT REASON
# ============================================================

def extract_reason(text):

    match = re.search(
        r"Reason\s*:\s*(.*)",
        text,
        re.IGNORECASE | re.DOTALL
    )

    if match:

        reason = match.group(1).strip()

        return reason

    return ""


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

    response = model.generate_content(prompt)

    if not response or not response.text:
        return ""

    return response.text.strip()


# ============================================================
# EVALUATE ANSWER USING LOCAL GEMMA
# ============================================================

def evaluate_with_gemma(
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

Evaluate the answer on these four criteria:

1. Faithfulness:
Is the answer supported by the retrieved context?

2. Relevance:
Does the answer directly answer the question?

3. Completeness:
Does the answer include the important information available
in the retrieved context?

4. Overall:
Give an overall quality score.

Give each score from 1 to 5.

Return ONLY valid JSON.
Do not use markdown.
Do not use code fences.

Use exactly this structure:

{{
  "faithfulness": 1,
  "relevance": 1,
  "completeness": 1,
  "overall_score": 1,
  "reason": "short explanation"
}}
"""

    try:

        result = subprocess.run(
            [
                r"C:\Users\megha\AppData\Local\Programs\Ollama\ollama.exe",
                "run",
                "gemma3:4b",
                prompt
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180
        )

        raw_output = result.stdout.strip()

        # Remove possible markdown fences
        raw_output = re.sub(
            r"```json\s*",
            "",
            raw_output,
            flags=re.IGNORECASE
        )

        raw_output = re.sub(
            r"```\s*$",
            "",
            raw_output
        )

        # Remove control characters that previously
        # caused JSON parsing errors.
        raw_output = re.sub(
            r"[\x00-\x08\x0B\x0C\x0E-\x1F]",
            " ",
            raw_output
        )

        evaluation = json.loads(
            raw_output
        )

        return {
            "faithfulness":
                evaluation.get("faithfulness", ""),

            "relevance":
                evaluation.get("relevance", ""),

            # "correctness":
            #     evaluation.get("correctness", ""),    

            "completeness":
                evaluation.get("completeness", ""),

            "overall":
                evaluation.get("overall_score", ""),

            "reason":
                evaluation.get("reason", "")
        }

    except Exception as e:

        return {
            "faithfulness": "",
            "relevance": "",
            "completeness": "",
            "overall": "",
            "reason":
                f"Gemma evaluation error: {e}"
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

    print(
        "\nGemini requests per question: 1"
    )

    print(
        "Maximum Gemini requests for this run: "
        f"{len(questions_df)}"
    )


    results = []


    # ========================================================
    # PROCESS QUESTIONS
    # ========================================================

    for index, row in questions_df.iterrows():

        question = str(
            row["question"]
        )

        expected_answer = str(
            row["expected_answer"]
        )

        # expected_document = str(
        #     row["expected_document"]
        # )

        # expected_page = str(
        #     row["expected_page"]
        # )


        print("\n" + "=" * 80)

        print(
            f"Question {index + 1}/"
            f"{len(questions_df)}"
        )

        print("=" * 80)

        print(
            f"Question: {question}"
        )


        try:

            # =================================================
            # RETRIEVAL
            # =================================================

            print(
                "\nRunning hybrid retrieval..."
            )

            retrieval_results = hybrid_search(
                question,
                top_k=TOP_K
            )


            # -------------------------------------------------
            # SAFETY CHECK
            # -------------------------------------------------

            if not isinstance(
                retrieval_results,
                list
            ):

                retrieval_results = []


            retrieval_results = [
                item
                for item in retrieval_results
                if isinstance(item, dict)
            ]


            # =================================================
            # BUILD CONTEXT
            # =================================================

            context = build_context(
                retrieval_results
            )


            # =================================================
            # RETRIEVED SOURCES
            # =================================================

            retrieved_documents = []

            retrieved_pages = []


            for item in retrieval_results:

                document = get_document(
                    item
                )

                page = get_page(
                    item
                )


                if document:

                    retrieved_documents.append(
                        document
                    )


                if page:

                    retrieved_pages.append(
                        page
                    )


            # # =================================================
            # # EXPECTED SOURCE CHECK
            # # =================================================

            # expected_source_found = "NO"


            # for item in retrieval_results:

            #     document = get_document(
            #         item
            #     )

            #     page = get_page(
            #         item
            #     )


            #     if (
            #         document == expected_document
            #         and page == expected_page
            #     ):

            #         expected_source_found = "YES"

            #         break


            # =================================================
            # GEMINI
            # =================================================

            print(
                "\nGenerating answer using Gemini..."
            )

            answer = generate_answer(
                question,
                context
            )

            print(
                "\nEvaluating answer using local Gemma 3:4B..."
            )

            evaluation = evaluate_with_gemma(
                question,
                context,
                answer
            )


            # =================================================
            # PRINT ANSWER
            # =================================================

            print(
                "\nANSWER:"
            )

            print(answer)


            # =================================================
            # PRINT SCORES
            # =================================================

            print(
                "\nANSWER QUALITY:"
            )

            print(
                f"Relevance     : "
                f"{evaluation['relevance']}"
            )

            # print(
            #     f"Correctness   : "
            #     f"{evaluation['correctness']}"
            # )

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


            # =================================================
            # SAVE RESULT
            # =================================================

            results.append({

                "question":
                    question,

                # "expected_document":
                #     expected_document,

                # "expected_page":
                #     expected_page,

                "retrieved_documents":
                    " | ".join(
                        retrieved_documents
                    ),

                "retrieved_pages":
                    " | ".join(
                        retrieved_pages
                    ),

                # "expected_source_found":
                #     expected_source_found,

                "answer":
                    answer,

                "relevance":
                    evaluation["relevance"],

                # "correctness":
                #     evaluation["correctness"],

                "completeness":
                    evaluation["completeness"],

                "faithfulness":
                    evaluation["faithfulness"],

                "overall_score":
                    evaluation["overall"],

                "reason":
                    evaluation["reason"]
            })


        except Exception as e:

            print(
                "\nERROR:"
            )

            print(e)


            results.append({

                "question":
                    question,

                # "expected_document":
                #     expected_document,

                # "expected_page":
                #     expected_page,

                "retrieved_documents":
                    "",

                "retrieved_pages":
                    "",

                # "expected_source_found":
                #     "ERROR",

                "answer":
                    "",

                "relevance":
                    "",

                # "correctness":
                #     "",

                "completeness":
                    "",

                "faithfulness":
                    "",

                "overall_score":
                    "",

                "reason":
                    str(e)
            })


            # -------------------------------------------------
            # If Gemini quota is exhausted, stop immediately.
            # -------------------------------------------------

            if "429" in str(e):

                print(
                    "\nStopping because Gemini quota "
                    "has been exhausted."
                )

                break


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )


    results_df.to_csv(
        RESULTS_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    if len(results_df) == 0:

        print(
            "\nNo evaluation results were generated."
        )

        return


    numeric_scores = pd.to_numeric(
        results_df["overall_score"],
        errors="coerce"
    )


    average_score = numeric_scores.mean()


    # successful = (
    #     results_df[
    #         "expected_source_found"
    #     ] != "ERROR"
    # ).sum()


    # source_found = (
    #     results_df[
    #         "expected_source_found"
    #     ] == "YES"
    # ).sum()


    answered = (
        results_df["answer"]
        .fillna("")
        .str.strip()
        .ne("")
    ).sum()


    # ========================================================
    # FINAL OUTPUT
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
        f"{len(results_df)}"
    )


    # print(
    #     f"Successful            : "
    #     f"{successful}"
    # )


    print(
        f"Answers generated     : "
        f"{answered}"
    )


    # print(
    #     f"Expected source found: "
    #     f"{source_found}/{len(results_df)}"
    # )


    if pd.notna(average_score):

        print(
            f"Average answer score : "
            f"{average_score:.2f}/5"
        )

    else:

        print(
            "Average answer score : "
            "N/A"
        )


    print(
        "\nResults saved to:"
    )

    print(
        RESULTS_FILE
    )


    print("=" * 80)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()

