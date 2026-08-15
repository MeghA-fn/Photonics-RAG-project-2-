
from pathlib import Path
import sys
import os
import re
import time
import pandas as pd
import google.generativeai as genai


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
    / "evaluation_questions.csv"
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

MODEL_NAME = "gemini-2.5-flash"

model = genai.GenerativeModel(MODEL_NAME)


# ============================================================
# SETTINGS
# ============================================================

TOP_K = 5

# IMPORTANT:
# We intentionally use ONE Gemini request per question.
# The previous implementation used two requests:
#   1. Generate answer
#   2. Evaluate answer
#
# This version combines both operations into ONE request.


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
# ONE GEMINI REQUEST
# ============================================================

def generate_and_evaluate(question, context):

    prompt = f"""
You are a scientific assistant and evaluator for a
Photonics Retrieval-Augmented Generation (RAG) system.

You must perform TWO tasks in ONE response.

TASK 1:
Answer the question using ONLY the supplied context.

TASK 2:
Evaluate the answer you generated against the supplied context.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not invent information.
3. If the context does not contain enough information,
   answer:

"I don't know based on the provided documents."

4. The answer should be concise and technically accurate.
5. Evaluate whether the answer is supported by the context.
6. Give scores from 1 to 5.

Question:
{question}

Context:
{context}


Return EXACTLY in this format:

ANSWER:
<your answer>

Relevance: <1-5>
Correctness: <1-5>
Completeness: <1-5>
Faithfulness: <1-5>
Overall: <1-5>
Reason: <short explanation>
"""

    try:

        response = model.generate_content(prompt)

        if not response or not response.text:

            return {
                "answer": "No answer generated.",
                "relevance": "",
                "correctness": "",
                "completeness": "",
                "faithfulness": "",
                "overall": "",
                "reason": "Gemini returned an empty response."
            }

        text = response.text.strip()


        # ----------------------------------------------------
        # EXTRACT ANSWER
        # ----------------------------------------------------

        answer_match = re.search(
            r"ANSWER\s*:\s*(.*?)(?=\n\s*Relevance\s*:)",
            text,
            re.IGNORECASE | re.DOTALL
        )

        if answer_match:

            answer = answer_match.group(1).strip()

        else:

            answer = text


        # ----------------------------------------------------
        # EXTRACT SCORES
        # ----------------------------------------------------

        relevance = extract_score(
            text,
            "Relevance"
        )

        correctness = extract_score(
            text,
            "Correctness"
        )

        completeness = extract_score(
            text,
            "Completeness"
        )

        faithfulness = extract_score(
            text,
            "Faithfulness"
        )

        overall = extract_score(
            text,
            "Overall"
        )

        reason = extract_reason(text)


        return {
            "answer": answer,
            "relevance": relevance,
            "correctness": correctness,
            "completeness": completeness,
            "faithfulness": faithfulness,
            "overall": overall,
            "reason": reason
        }


    except Exception as e:

        error_text = str(e)

        print("\nGemini error:")
        print(error_text)

        if "429" in error_text:

            print(
                "\nGemini quota/rate limit reached."
            )

            print(
                "Stopping evaluation instead of repeatedly "
                "retrying and wasting requests."
            )

        return {
            "answer": "",
            "relevance": "",
            "correctness": "",
            "completeness": "",
            "faithfulness": "",
            "overall": "",
            "reason": f"Gemini error: {error_text}"
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

        expected_document = str(
            row["expected_document"]
        )

        expected_page = str(
            row["expected_page"]
        )


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


            # =================================================
            # EXPECTED SOURCE CHECK
            # =================================================

            expected_source_found = "NO"


            for item in retrieval_results:

                document = get_document(
                    item
                )

                page = get_page(
                    item
                )


                if (
                    document == expected_document
                    and page == expected_page
                ):

                    expected_source_found = "YES"

                    break


            # =================================================
            # GEMINI
            # =================================================

            print(
                "\nGenerating and evaluating answer..."
            )

            evaluation = generate_and_evaluate(
                question,
                context
            )


            # =================================================
            # PRINT ANSWER
            # =================================================

            print(
                "\nANSWER:"
            )

            print(
                evaluation["answer"]
            )


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

            print(
                f"Correctness   : "
                f"{evaluation['correctness']}"
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


            # =================================================
            # SAVE RESULT
            # =================================================

            results.append({

                "question":
                    question,

                "expected_document":
                    expected_document,

                "expected_page":
                    expected_page,

                "retrieved_documents":
                    " | ".join(
                        retrieved_documents
                    ),

                "retrieved_pages":
                    " | ".join(
                        retrieved_pages
                    ),

                "expected_source_found":
                    expected_source_found,

                "answer":
                    evaluation["answer"],

                "relevance":
                    evaluation["relevance"],

                "correctness":
                    evaluation["correctness"],

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

                "expected_document":
                    expected_document,

                "expected_page":
                    expected_page,

                "retrieved_documents":
                    "",

                "retrieved_pages":
                    "",

                "expected_source_found":
                    "ERROR",

                "answer":
                    "",

                "relevance":
                    "",

                "correctness":
                    "",

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


    successful = (
        results_df[
            "expected_source_found"
        ] != "ERROR"
    ).sum()


    source_found = (
        results_df[
            "expected_source_found"
        ] == "YES"
    ).sum()


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


    print(
        f"Successful            : "
        f"{successful}"
    )


    print(
        f"Answers generated     : "
        f"{answered}"
    )


    print(
        f"Expected source found: "
        f"{source_found}/{len(results_df)}"
    )


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

