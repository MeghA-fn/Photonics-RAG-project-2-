from pathlib import Path
import sys
import csv
import os

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
# GEMINI
# ============================================================

from google import genai


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
    / "answer_evaluation_results.csv"
)


# ============================================================
# GEMINI CLIENT
# ============================================================

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set."
    )

client = genai.Client(api_key=API_KEY)


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "gemini-2.5-flash"

TOP_K = 5


# ============================================================
# LOAD QUESTIONS
# ============================================================

def load_questions():

    questions = []

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            questions.append({
                "question": row["question"].strip(),
                "expected_document": row[
                    "expected_document"
                ].strip(),
                "expected_page": row[
                    "expected_page"
                ].strip()
            })

    return questions


# ============================================================
# EXTRACT RESULT INFORMATION
# ============================================================

def extract_result_information(results):

    documents = []
    pages = []

    for item in results[:TOP_K]:

        if not isinstance(item, dict):
            continue

        document = item.get(
            "document_name",
            ""
        )

        page = item.get(
            "page_number",
            ""
        )

        if document:
            documents.append(document)

        if page != "":
            pages.append(str(page))

    return documents, pages


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):

    context_parts = []

    for rank, item in enumerate(
        results[:TOP_K],
        start=1
    ):

        if not isinstance(item, dict):
            continue

        document = item.get(
            "document_name",
            "Unknown document"
        )

        page = item.get(
            "page_number",
            "Unknown page"
        )

        text = item.get(
            "text",
            ""
        )

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
# GENERATE GEMINI ANSWER
# ============================================================

def generate_answer(question, context):

    prompt = f"""
You are a scientific assistant specialized in photonics.

Answer the user's question using ONLY the information
provided in the retrieved context below.

Do not use outside knowledge.

If the context does not contain enough information to answer
the question, say:

"I don't know based on the provided context."

Keep the answer scientifically accurate and concise.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

ANSWER:
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text.strip()


# ============================================================
# CHECK EXPECTED SOURCE
# ============================================================

def check_expected_source(
    results,
    expected_document,
    expected_page
):

    for item in results[:TOP_K]:

        if not isinstance(item, dict):
            continue

        document = item.get(
            "document_name",
            ""
        )

        page = str(
            item.get(
                "page_number",
                ""
            )
        )

        if (
            document == expected_document
            and page == str(expected_page)
        ):
            return "YES"

    return "NO"


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 80)
    print("RAG ANSWER EVALUATION")
    print("=" * 80)

    print()

    questions = load_questions()

    print(
        f"Loaded {len(questions)} evaluation questions."
    )

    print()

    results_rows = []

    for index, item in enumerate(
        questions,
        start=1
    ):

        question = item["question"]

        expected_document = item[
            "expected_document"
        ]

        expected_page = item[
            "expected_page"
        ]

        print("=" * 80)
        print(
            f"QUESTION {index}/{len(questions)}"
        )
        print("=" * 80)

        print(
            f"Question: {question}"
        )

        print(
            "Running hybrid retrieval..."
        )

        try:

            retrieval_results = hybrid_search(
                question
            )

        except Exception as error:

            print(
                f"Retrieval error: {error}"
            )

            results_rows.append({

                "question": question,

                "expected_document":
                    expected_document,

                "expected_page":
                    expected_page,

                "generated_answer":
                    "",

                "retrieved_documents":
                    "",

                "retrieved_pages":
                    "",

                "expected_source_found":
                    "ERROR",

                "status":
                    "RETRIEVAL_ERROR"
            })

            continue


        # ----------------------------------------------------
        # Extract source information
        # ----------------------------------------------------

        documents, pages = (
            extract_result_information(
                retrieval_results
            )
        )


        # ----------------------------------------------------
        # Check expected source
        # ----------------------------------------------------

        expected_source_found = (
            check_expected_source(
                retrieval_results,
                expected_document,
                expected_page
            )
        )


        # ----------------------------------------------------
        # Build context
        # ----------------------------------------------------

        context = build_context(
            retrieval_results
        )


        if not context:

            print(
                "No usable context found."
            )

            generated_answer = (
                "I don't know based on "
                "the provided context."
            )

        else:

            print(
                "Sending retrieved context to Gemini..."
            )

            try:

                generated_answer = (
                    generate_answer(
                        question,
                        context
                    )
                )

            except Exception as error:

                print(
                    f"Gemini error: {error}"
                )

                generated_answer = ""


        # ----------------------------------------------------
        # Display answer
        # ----------------------------------------------------

        print()
        print("-" * 80)
        print("GENERATED ANSWER")
        print("-" * 80)

        print(generated_answer)

        print()

        print(
            f"Expected document: "
            f"{expected_document}"
        )

        print(
            f"Expected page: "
            f"{expected_page}"
        )

        print(
            f"Expected source found in Top-{TOP_K}: "
            f"{expected_source_found}"
        )

        print()


        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results_rows.append({

            "question":
                question,

            "expected_document":
                expected_document,

            "expected_page":
                expected_page,

            "generated_answer":
                generated_answer,

            "retrieved_documents":
                " | ".join(documents),

            "retrieved_pages":
                " | ".join(pages),

            "expected_source_found":
                expected_source_found,

            "status":
                "SUCCESS"
        })


    # ========================================================
    # SAVE CSV
    # ========================================================

    fieldnames = [
        "question",
        "expected_document",
        "expected_page",
        "generated_answer",
        "retrieved_documents",
        "retrieved_pages",
        "expected_source_found",
        "status"
    ]

    with open(
        RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results_rows
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    successful = sum(
        1
        for row in results_rows
        if row["status"] == "SUCCESS"
    )

    source_found = sum(
        1
        for row in results_rows
        if row["expected_source_found"]
        == "YES"
    )

    print()
    print("=" * 80)
    print("RAG ANSWER EVALUATION COMPLETE")
    print("=" * 80)

    print(
        f"Total questions : {len(questions)}"
    )

    print(
        f"Successful      : {successful}"
    )

    print(
        f"Expected source found in Top-{TOP_K}: "
        f"{source_found}/{len(questions)}"
    )

    if questions:

        source_accuracy = (
            source_found
            / len(questions)
            * 100
        )

        print(
            f"Source accuracy : "
            f"{source_accuracy:.2f}%"
        )

    print()

    print(
        f"Results saved to:"
    )

    print(
        RESULTS_FILE
    )

    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()