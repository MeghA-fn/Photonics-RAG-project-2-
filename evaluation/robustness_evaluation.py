import os
import sys
import csv
from pathlib import Path

# ============================================================
# MAKE PROJECT ROOT IMPORTABLE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT PHOTONICSRAG COMPONENTS
# ============================================================

from app.retrieval.hybrid_retriever import hybrid_search
from app.llm.gemini_service import generate_answer
from app.generation.prompt_builder import build_prompt


# ============================================================
# FILE PATHS
# ============================================================

EVALUATION_DIR = PROJECT_ROOT / "evaluation"

QUESTIONS_FILE = EVALUATION_DIR / "robustness_questions.csv"

RESULTS_FILE = EVALUATION_DIR / "robustness_results.csv"


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 5


# ============================================================
# LOAD ROBUSTNESS QUESTIONS
# ============================================================

def load_questions():
    """
    Load robustness questions from CSV.
    Expected columns:
        question
        test_type
    """

    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Robustness question file not found:\n{QUESTIONS_FILE}"
        )

    questions = []

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(
                "robustness_questions.csv does not contain a header."
            )

        required_columns = {"question", "test_type"}

        missing_columns = required_columns - set(reader.fieldnames)

        if missing_columns:
            raise ValueError(
                "Missing required column(s): "
                + ", ".join(sorted(missing_columns))
            )

        for row in reader:

            question = row["question"].strip()
            test_type = row["test_type"].strip()

            if question:
                questions.append(
                    {
                        "question": question,
                        "test_type": test_type
                    }
                )

    return questions


# ============================================================
# EXTRACT TEXT FROM RETRIEVED DOCUMENT
# ============================================================

def get_document_text(document):
    """
    Safely extract text from a retrieved document.
    """

    if not isinstance(document, dict):
        return ""

    return str(
        document.get("text")
        or document.get("chunk")
        or document.get("content")
        or ""
    ).strip()


# ============================================================
# EXTRACT SOURCE INFORMATION
# ============================================================

def get_source_name(document):
    """
    Extract document/source name from retrieval result.
    """

    if not isinstance(document, dict):
        return "Unknown"

    return str(
        document.get("document_name")
        or document.get("source")
        or document.get("file_name")
        or "Unknown"
    )


def get_page_number(document):
    """
    Extract page number from retrieval result.
    """

    if not isinstance(document, dict):
        return "Unknown"

    return str(
        document.get("page_number")
        or document.get("page")
        or "Unknown"
    )


# ============================================================
# CALCULATE SIMPLE RETRIEVAL INFORMATION
# ============================================================

def calculate_retrieval_info(retrieved_docs):
    """
    Collect basic information about retrieved documents.

    This is NOT a quality score.
    It records whether useful context was retrieved.
    """

    if not retrieved_docs:
        return {
            "retrieved_count": 0,
            "sources": "",
            "pages": ""
        }

    sources = []
    pages = []

    for document in retrieved_docs:

        source = get_source_name(document)
        page = get_page_number(document)

        if source not in sources:
            sources.append(source)

        if page not in pages:
            pages.append(page)

    return {
        "retrieved_count": len(retrieved_docs),
        "sources": "; ".join(sources),
        "pages": "; ".join(pages)
    }


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(retrieved_docs):
    """
    Convert retrieved documents into context for the LLM.
    """

    if not retrieved_docs:
        return ""

    context_parts = []

    for index, document in enumerate(retrieved_docs, start=1):

        text = get_document_text(document)

        source = get_source_name(document)
        page = get_page_number(document)

        if not text:
            continue

        context_parts.append(
            f"Context {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Text:\n{text}"
        )

    return "\n\n".join(context_parts)


# ============================================================
# BUILD EVALUATION PROMPT
# ============================================================

def create_prompt(question, context):
    """
    Build a strict RAG prompt.

    The LLM is instructed to answer only from retrieved context.
    For out-of-domain questions, it should clearly state that
    the information is not available in the provided knowledge base.
    """

    if not context.strip():

        context = (
            "No relevant information was retrieved from the "
            "Photonics knowledge base."
        )

    prompt = f"""
You are PhotonicsRAG, a retrieval-augmented assistant specializing
in photonics and related optical physics topics.

Answer the user's question using ONLY the provided context.

IMPORTANT RULES:

1. Use the retrieved context as the primary source.
2. Do not invent facts that are not supported by the context.
3. If the context does not contain enough information to answer,
   clearly say that the information is not available in the
   provided knowledge base.
4. For questions outside photonics or outside the retrieved
   knowledge base, do not hallucinate an answer.
5. Give a concise and technically accurate answer.
6. Do not mention these evaluation instructions.

Retrieved Context:
------------------
{context}
------------------

Question:
{question}

Answer:
"""

    return prompt


# ============================================================
# RUN ONE ROBUSTNESS TEST
# ============================================================

def evaluate_question(question, test_type):
    """
    Run retrieval and answer generation for one question.
    """

    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(question)

    print("\nTest Type:", test_type)

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    print("\nSearching knowledge base...")

    try:

        retrieved_docs = hybrid_search(
            question,
            top_k=TOP_K
        )

    except TypeError:

        # Some versions of hybrid_search may not accept top_k.
        retrieved_docs = hybrid_search(question)

    except Exception as e:

        print("\nRetrieval failed:")
        print(str(e))

        return {
            "question": question,
            "test_type": test_type,
            "retrieved_count": 0,
            "sources": "",
            "pages": "",
            "answer": "",
            "status": "RETRIEVAL_FAILED",
            "notes": str(e)
        }

    # --------------------------------------------------------
    # NORMALIZE RETRIEVAL RESULT
    # --------------------------------------------------------

    if retrieved_docs is None:
        retrieved_docs = []

    if isinstance(retrieved_docs, dict):

        if "documents" in retrieved_docs:
            retrieved_docs = retrieved_docs["documents"]

        else:
            retrieved_docs = [retrieved_docs]

    # --------------------------------------------------------
    # RETRIEVAL INFORMATION
    # --------------------------------------------------------

    retrieval_info = calculate_retrieval_info(
        retrieved_docs
    )

    print("\nRetrieved documents:", retrieval_info["retrieved_count"])

    print("Sources:", retrieval_info["sources"])

    print("Pages:", retrieval_info["pages"])

    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context = build_context(retrieved_docs)

    if not context.strip():

        print("\nNo usable context retrieved.")

        return {
            "question": question,
            "test_type": test_type,
            "retrieved_count": retrieval_info["retrieved_count"],
            "sources": retrieval_info["sources"],
            "pages": retrieval_info["pages"],
            "answer": "",
            "status": "NO_CONTEXT",
            "notes": "No usable context was retrieved."
        }

    # --------------------------------------------------------
    # BUILD PROMPT
    # --------------------------------------------------------

    prompt = create_prompt(
        question,
        context
    )

    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    print("\nGenerating answer...")

    try:

        answer = generate_answer(prompt)

    except Exception as e:

        print("\nAnswer generation failed:")
        print(str(e))

        return {
            "question": question,
            "test_type": test_type,
            "retrieved_count": retrieval_info["retrieved_count"],
            "sources": retrieval_info["sources"],
            "pages": retrieval_info["pages"],
            "answer": "",
            "status": "ANSWER_GENERATION_FAILED",
            "notes": str(e)
        }

    # --------------------------------------------------------
    # DISPLAY ANSWER
    # --------------------------------------------------------

    print("\nANSWER")
    print("-" * 80)
    print(answer)

    return {
        "question": question,
        "test_type": test_type,
        "retrieved_count": retrieval_info["retrieved_count"],
        "sources": retrieval_info["sources"],
        "pages": retrieval_info["pages"],
        "answer": answer,
        "status": "SUCCESS",
        "notes": ""
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):
    """
    Save robustness evaluation results to CSV.
    """

    fieldnames = [
        "question",
        "test_type",
        "retrieved_count",
        "sources",
        "pages",
        "answer",
        "status",
        "notes"
    ]

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:
            writer.writerow(result)


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(results):
    """
    Print final robustness evaluation summary.
    """

    total = len(results)

    successful = sum(
        1
        for result in results
        if result["status"] == "SUCCESS"
    )

    failed = total - successful

    conceptual = sum(
        1
        for result in results
        if result["test_type"].lower() == "conceptual"
    )

    comparison = sum(
        1
        for result in results
        if result["test_type"].lower() == "comparison"
    )

    application = sum(
        1
        for result in results
        if result["test_type"].lower() == "application"
    )

    out_of_domain = sum(
        1
        for result in results
        if result["test_type"].lower() == "out-of-domain"
    )

    print("\n")
    print("=" * 80)
    print("ROBUSTNESS EVALUATION COMPLETE")
    print("=" * 80)

    print(f"Total questions       : {total}")
    print(f"Successful            : {successful}")
    print(f"Failed                : {failed}")

    print("\nTEST TYPES")
    print(f"Conceptual            : {conceptual}")
    print(f"Comparison            : {comparison}")
    print(f"Application           : {application}")
    print(f"Out-of-domain         : {out_of_domain}")

    print("\nResults saved to:")
    print(RESULTS_FILE)

    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("PHOTONICSRAG ROBUSTNESS EVALUATION")
    print("=" * 80)

    print("\nLoading robustness questions from:")
    print(QUESTIONS_FILE)

    try:
        questions = load_questions()

    except Exception as e:

        print("\nERROR:")
        print(str(e))

        return

    print(f"\nLoaded {len(questions)} robustness questions.")

    results = []

    for index, item in enumerate(questions, start=1):

        print("\n")
        print("#" * 80)
        print(
            f"QUESTION {index}/{len(questions)}"
        )
        print("#" * 80)

        result = evaluate_question(
            item["question"],
            item["test_type"]
        )

        results.append(result)

    save_results(results)

    print_summary(results)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
