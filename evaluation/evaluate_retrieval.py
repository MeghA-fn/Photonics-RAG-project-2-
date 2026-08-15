from pathlib import Path
import sys
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT RETRIEVERS
# ============================================================

from app.retrieval.retriever import retrieve_documents
from app.retrieval.hybrid_retriever import (
    bm25_search,
    hybrid_search
)


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
    / "retrieval_results.csv"
)


# ============================================================
# HELPER
# ============================================================

def make_list(results):
    """
    Convert retrieval output into a list.

    Successful retrieval:
        list of dictionaries

    No-answer retrieval:
        dictionary
    """

    if isinstance(results, list):
        return results

    return []


# ============================================================
# DOCUMENT + PAGE MATCH
# ============================================================

def is_correct(
    result,
    expected_document,
    expected_page
):

    if not isinstance(result, dict):
        return False

    result_document = str(
        result.get("document_name", "")
    ).strip().lower()

    expected_document = str(
        expected_document
    ).strip().lower()

    try:

        result_page = int(
            result.get("page_number")
        )

        expected_page = int(
            expected_page
        )

    except (ValueError, TypeError):

        return False

    return (
        result_document == expected_document
        and result_page == expected_page
    )


# ============================================================
# FIND RANK
# ============================================================

def find_rank(
    results,
    expected_document,
    expected_page
):

    results = make_list(results)

    for rank, result in enumerate(
        results,
        start=1
    ):

        if is_correct(
            result,
            expected_document,
            expected_page
        ):

            return rank

    return None


# ============================================================
# HIT@K
# ============================================================

def hit_at_k(
    results,
    expected_document,
    expected_page,
    k
):

    results = make_list(results)

    for result in results[:k]:

        if is_correct(
            result,
            expected_document,
            expected_page
        ):

            return 1

    return 0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("PHOTONICS RAG RETRIEVAL EVALUATION")
    print("=" * 80)

    # --------------------------------------------------------
    # LOAD QUESTIONS
    # --------------------------------------------------------

    if not QUESTIONS_FILE.exists():

        print(
            "\nERROR: evaluation_questions.csv "
            "not found."
        )

        print(
            f"Expected location:\n"
            f"{QUESTIONS_FILE}"
        )

        return

    questions = pd.read_csv(
        QUESTIONS_FILE
    )

    print(
        f"\nLoaded {len(questions)} "
        f"evaluation questions."
    )

    results_table = []

    # ========================================================
    # EACH QUESTION
    # ========================================================

    for index, row in questions.iterrows():

        question = str(
            row["question"]
        ).strip()

        expected_document = str(
            row["expected_document"]
        ).strip()

        expected_page = int(
            row["expected_page"]
        )

        print("\n")
        print("=" * 80)
        print(
            f"QUESTION {index + 1}/"
            f"{len(questions)}"
        )
        print("=" * 80)

        print(
            f"Question: {question}"
        )

        print(
            f"Expected: "
            f"{expected_document} "
            f"(Page {expected_page})"
        )

        # ====================================================
        # SEMANTIC
        # ====================================================

        print(
            "\nRunning Semantic Search..."
        )

        try:

            semantic_results = (
                retrieve_documents(question)
            )

            semantic_results = make_list(
                semantic_results
            )

        except Exception as error:

            print(
                "Semantic Search Error:",
                error
            )

            semantic_results = []

        # ====================================================
        # BM25
        # ====================================================

        print(
            "\nRunning BM25 Search..."
        )

        try:

            bm25_results = bm25_search(
                question,
                top_k=5
            )

            bm25_results = make_list(
                bm25_results
            )

        except Exception as error:

            print(
                "BM25 Search Error:",
                error
            )

            bm25_results = []

        # ====================================================
        # RRF
        # ====================================================

        print(
            "\nRunning RRF Hybrid Search..."
        )

        try:

            hybrid_results = hybrid_search(
                question,
                top_k=5
            )

            hybrid_results = make_list(
                hybrid_results
            )

        except Exception as error:

            print(
                "RRF Hybrid Search Error:",
                error
            )

            hybrid_results = []

        # ====================================================
        # METRICS
        # ====================================================

        semantic_rank = find_rank(
            semantic_results,
            expected_document,
            expected_page
        )

        semantic_hit1 = hit_at_k(
            semantic_results,
            expected_document,
            expected_page,
            1
        )

        semantic_hit3 = hit_at_k(
            semantic_results,
            expected_document,
            expected_page,
            3
        )

        semantic_hit5 = hit_at_k(
            semantic_results,
            expected_document,
            expected_page,
            5
        )

        # ----------------------------------------------------

        bm25_rank = find_rank(
            bm25_results,
            expected_document,
            expected_page
        )

        bm25_hit1 = hit_at_k(
            bm25_results,
            expected_document,
            expected_page,
            1
        )

        bm25_hit3 = hit_at_k(
            bm25_results,
            expected_document,
            expected_page,
            3
        )

        bm25_hit5 = hit_at_k(
            bm25_results,
            expected_document,
            expected_page,
            5
        )

        # ----------------------------------------------------

        hybrid_rank = find_rank(
            hybrid_results,
            expected_document,
            expected_page
        )

        hybrid_hit1 = hit_at_k(
            hybrid_results,
            expected_document,
            expected_page,
            1
        )

        hybrid_hit3 = hit_at_k(
            hybrid_results,
            expected_document,
            expected_page,
            3
        )

        hybrid_hit5 = hit_at_k(
            hybrid_results,
            expected_document,
            expected_page,
            5
        )

        # ====================================================
        # PRINT QUESTION RESULT
        # ====================================================

        print("\nRESULT")

        print(
            f"Semantic : "
            f"Rank={semantic_rank}, "
            f"Hit@1={semantic_hit1}, "
            f"Hit@3={semantic_hit3}, "
            f"Hit@5={semantic_hit5}"
        )

        print(
            f"BM25     : "
            f"Rank={bm25_rank}, "
            f"Hit@1={bm25_hit1}, "
            f"Hit@3={bm25_hit3}, "
            f"Hit@5={bm25_hit5}"
        )

        print(
            f"RRF      : "
            f"Rank={hybrid_rank}, "
            f"Hit@1={hybrid_hit1}, "
            f"Hit@3={hybrid_hit3}, "
            f"Hit@5={hybrid_hit5}"
        )

        # ====================================================
        # SAVE
        # ====================================================

        results_table.append({

            "question":
                question,

            "expected_document":
                expected_document,

            "expected_page":
                expected_page,

            "semantic_rank":
                semantic_rank,

            "semantic_hit@1":
                semantic_hit1,

            "semantic_hit@3":
                semantic_hit3,

            "semantic_hit@5":
                semantic_hit5,

            "bm25_rank":
                bm25_rank,

            "bm25_hit@1":
                bm25_hit1,

            "bm25_hit@3":
                bm25_hit3,

            "bm25_hit@5":
                bm25_hit5,

            "rrf_rank":
                hybrid_rank,

            "rrf_hit@1":
                hybrid_hit1,

            "rrf_hit@3":
                hybrid_hit3,

            "rrf_hit@5":
                hybrid_hit5
        })

    # ========================================================
    # DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results_table
    )

    # ========================================================
    # OVERALL RESULTS
    # ========================================================

    print("\n\n")
    print("=" * 80)
    print("OVERALL RETRIEVAL RESULTS")
    print("=" * 80)

    if results_df.empty:

        print(
            "\nNo results available."
        )

        return

    semantic_1 = results_df[
        "semantic_hit@1"
    ].mean()

    semantic_3 = results_df[
        "semantic_hit@3"
    ].mean()

    semantic_5 = results_df[
        "semantic_hit@5"
    ].mean()

    bm25_1 = results_df[
        "bm25_hit@1"
    ].mean()

    bm25_3 = results_df[
        "bm25_hit@3"
    ].mean()

    bm25_5 = results_df[
        "bm25_hit@5"
    ].mean()

    rrf_1 = results_df[
        "rrf_hit@1"
    ].mean()

    rrf_3 = results_df[
        "rrf_hit@3"
    ].mean()

    rrf_5 = results_df[
        "rrf_hit@5"
    ].mean()

    print(
        "\nMethod          Hit@1       Hit@3       Hit@5"
    )

    print("-" * 60)

    print(
        f"Semantic       "
        f"{semantic_1:.2%}       "
        f"{semantic_3:.2%}       "
        f"{semantic_5:.2%}"
    )

    print(
        f"BM25           "
        f"{bm25_1:.2%}       "
        f"{bm25_3:.2%}       "
        f"{bm25_5:.2%}"
    )

    print(
        f"RRF Hybrid     "
        f"{rrf_1:.2%}       "
        f"{rrf_3:.2%}       "
        f"{rrf_5:.2%}"
    )

    # ========================================================
    # BEST METHOD
    # ========================================================

    scores = {

        "Semantic": semantic_5,

        "BM25": bm25_5,

        "RRF Hybrid": rrf_5
    }

    best_method = max(
        scores,
        key=scores.get
    )

    print("\n")
    print("=" * 80)

    print(
        f"Best method based on Hit@5: "
        f"{best_method}"
    )

    print(
        f"Hit@5: "
        f"{scores[best_method]:.2%}"
    )

    print("=" * 80)

    # ========================================================
    # SAVE CSV
    # ========================================================

    results_df.to_csv(
        RESULTS_FILE,
        index=False
    )

    print(
        f"\nDetailed results saved to:\n"
        f"{RESULTS_FILE}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()