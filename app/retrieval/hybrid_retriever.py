import re
from collections import defaultdict
import pandas as pd
from rank_bm25 import BM25Okapi
from app.retrieval.retriever import retrieve_documents


# =============================================================================
# CONFIGURATION
# =============================================================================

TOP_K = 5
RRF_K = 60

# Minimum number of accepted results
MIN_RELEVANT_RESULTS = 1

# Minimum RRF score required
RRF_MIN_SCORE = 0.010


# =============================================================================
# STOP WORDS
# =============================================================================

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
}


# =============================================================================
# LOAD CHUNKS FOR BM25
# =============================================================================

print("Loading chunks for BM25...")

CHUNKS_FILE = "data/processed/chunks.csv"

chunks_df = pd.read_csv(CHUNKS_FILE)

print(f"Loaded {len(chunks_df)} chunks for BM25.")


# =============================================================================
# PREPROCESS TEXT
# =============================================================================

def preprocess_text(text):
    """
    Convert text into BM25-friendly tokens.
    """

    if not isinstance(text, str):
        return []

    text = text.lower()

    tokens = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text
    )

    tokens = [
        token
        for token in tokens
        if token not in STOP_WORDS
    ]

    return tokens


# =============================================================================
# PREPARE BM25 CORPUS
# =============================================================================

print("Preparing BM25 corpus...")

bm25_corpus = [
    preprocess_text(text)
    for text in chunks_df["chunk_text"].fillna("")
]

bm25 = BM25Okapi(bm25_corpus)

print("BM25 index ready.")


# =============================================================================
# BM25 SEARCH
# =============================================================================

def bm25_search(query, top_k=TOP_K):
    """
    Search the chunk database using BM25.
    """

    query_tokens = preprocess_text(query)

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for index in ranked_indices[:top_k]:

        row = chunks_df.iloc[index]

        results.append(
            {
                "chunk_id": row.get(
                    "chunk_id",
                    index
                ),

                "document_name": row[
                    "document_name"
                ],

                "page_number": row[
                    "page_number"
                ],

                # IMPORTANT:
                # CSV column is chunk_text,
                # but downstream RAG expects "text".
                "text": row[
                    "chunk_text"
                ],

                "score": float(
                    scores[index]
                ),

                "source": "BM25",
            }
        )

    return results


# =============================================================================
# RESULT KEY
# =============================================================================

def get_result_key(result):
    """
    Create a stable identifier for a retrieved chunk.
    """

    chunk_id = result.get("chunk_id")

    if chunk_id is not None:
        return str(chunk_id)

    return (
        f"{result.get('document_name', '')}"
        f"::{result.get('page_number', '')}"
        f"::{result.get('text', '')[:100]}"
    )


# =============================================================================
# RRF HYBRID SEARCH
# =============================================================================

def hybrid_search(query, top_k=TOP_K):
    """
    Hybrid retrieval pipeline:

        Semantic Search
              +
            BM25
              |
              v
        Reciprocal Rank Fusion
              |
              v
        Relevance Gate
              |
              v
        Final Results

    The existing RAG architecture is preserved.
    """

    # =========================================================================
    # STEP 1: SEMANTIC SEARCH
    # =========================================================================

    semantic_results = retrieve_documents(
        query,
        top_k=top_k
    )

    # Handle the special dictionary returned by the semantic retriever
    # when no sufficiently relevant information is found.
    if isinstance(
        semantic_results,
        dict
    ):

        if semantic_results.get(
            "status"
        ) == "no_answer_found":

            semantic_results = []

        else:

            semantic_results = semantic_results.get(
                "documents",
                []
            )

    if semantic_results is None:
        semantic_results = []


    # =========================================================================
    # STEP 2: BM25 SEARCH
    # =========================================================================

    bm25_results = bm25_search(
        query,
        top_k=top_k
    )


    # =========================================================================
    # STEP 3: RECIPROCAL RANK FUSION
    # =========================================================================

    rrf_scores = defaultdict(float)

    result_store = {}

    # -------------------------------------------------------------------------
    # Add semantic results
    # -------------------------------------------------------------------------

    for rank, result in enumerate(
        semantic_results,
        start=1
    ):

        key = get_result_key(result)

        rrf_scores[key] += (
            1.0 / (RRF_K + rank)
        )

        result_store[key] = result.copy()


    # -------------------------------------------------------------------------
    # Add BM25 results
    # -------------------------------------------------------------------------

    for rank, result in enumerate(
        bm25_results,
        start=1
    ):

        key = get_result_key(result)

        rrf_scores[key] += (
            1.0 / (RRF_K + rank)
        )

        if key not in result_store:
            result_store[key] = result.copy()


    # =========================================================================
    # STEP 4: SORT BY RRF SCORE
    # =========================================================================

    ranked_keys = sorted(
        rrf_scores.keys(),
        key=lambda key: rrf_scores[key],
        reverse=True
    )


    # =========================================================================
    # STEP 5: BUILD FINAL RESULTS
    # =========================================================================

    final_results = []

    semantic_keys = {
        get_result_key(item)
        for item in semantic_results
    }

    bm25_keys = {
        get_result_key(item)
        for item in bm25_results
    }


    for key in ranked_keys[:top_k]:

        result = result_store[key].copy()

        result["rrf_score"] = float(
            rrf_scores[key]
        )


        # ---------------------------------------------------------------------
        # Determine retrieval source
        # ---------------------------------------------------------------------

        semantic_match = (
            key in semantic_keys
        )

        bm25_match = (
            key in bm25_keys
        )


        if semantic_match and bm25_match:

            result["source"] = (
                "Semantic + BM25"
            )

        elif semantic_match:

            result["source"] = "Semantic"

        elif bm25_match:

            result["source"] = "BM25"

        else:

            result["source"] = "Hybrid"


        # ---------------------------------------------------------------------
        # Ensure downstream code always receives "text"
        # ---------------------------------------------------------------------

        if "text" not in result:

            if "chunk_text" in result:

                result["text"] = result[
                    "chunk_text"
                ]

            else:

                result["text"] = ""


        final_results.append(result)


    # =========================================================================
    # STEP 6: HANDLE NO RESULTS
    # =========================================================================

    if not final_results:

        print(
            "\nNo retrieval results found."
        )

        return []


    # =========================================================================
    # STEP 7: BEST RRF SCORE
    # =========================================================================

    best_rrf_score = final_results[0].get(
        "rrf_score",
        0.0
    )


    # =========================================================================
    # STEP 8: RELEVANT RESULTS
    # =========================================================================

    relevant_results = [
        result
        for result in final_results
        if result.get(
            "rrf_score",
            0.0
        ) >= RRF_MIN_SCORE
    ]


    # =========================================================================
    # STEP 9: DEBUG INFORMATION
    # =========================================================================

    print("\n" + "=" * 80)
    print("HYBRID SEARCH RESULTS")
    print("=" * 80)

    print(
        f"\nQuestion : {query}"
    )

    print(
        f"Semantic results : "
        f"{len(semantic_results)}"
    )

    print(
        f"BM25 results     : "
        f"{len(bm25_results)}"
    )

    print(
        f"Final results    : "
        f"{len(final_results)}"
    )

    print(
        f"Best RRF score   : "
        f"{best_rrf_score:.6f}"
    )


    # =========================================================================
    # STEP 10: RELEVANCE GATE
    # =========================================================================

    if (
        len(relevant_results)
        < MIN_RELEVANT_RESULTS
        or best_rrf_score
        < RRF_MIN_SCORE
    ):

        print(
            "\nNo sufficiently relevant "
            "information found in the "
            "knowledge base."
        )

        return []


    # =========================================================================
    # STEP 11: DISPLAY RESULTS
    # =========================================================================

    for rank, result in enumerate(
        final_results,
        start=1
    ):

        print(
            "\n" + "-" * 80
        )

        print(
            f"Rank     : {rank}"
        )

        print(
            f"Document : "
            f"{result.get('document_name', 'N/A')}"
        )

        print(
            f"Page     : "
            f"{result.get('page_number', 'N/A')}"
        )

        print(
            f"Source   : "
            f"{result.get('source', 'N/A')}"
        )

        print(
            f"RRF Score: "
            f"{result.get('rrf_score', 0):.6f}"
        )

        if result.get("distance") is not None:

            print(
                f"Distance : "
                f"{result['distance']:.4f}"
            )


    print(
        "\n" + "=" * 80
    )

    return final_results


# =============================================================================
# DIRECT TEST
# =============================================================================

if __name__ == "__main__":

    print("\n" + "=" * 80)
    print("HYBRID RETRIEVER TEST")
    print("=" * 80)

    while True:

        question = input(
            "\nEnter your question "
            "(type 'exit' to quit): "
        )

        if question.lower().strip() == "exit":
            break

        if not question.strip():

            print(
                "Please enter a valid question."
            )

            continue


        results = hybrid_search(
            question
        )


        print("\n" + "=" * 80)
        print("FINAL RETRIEVAL OUTPUT")
        print("=" * 80)


        if not results:

            print(
                "\nI couldn't find sufficient "
                "information in the uploaded "
                "photonics documents."
            )

        else:

            # Group pages belonging to the same document
            grouped_sources = {}

            for result in results:

                document = result["document_name"]
                page = result["page_number"]

                if document not in grouped_sources:
                    grouped_sources[document] = []

                if page not in grouped_sources[document]:
                    grouped_sources[document].append(page)

            # Print each document only once
            for rank, (document, pages) in enumerate(
                grouped_sources.items(),
                start=1
            ):

                pages = sorted(pages)

                if len(pages) == 1:

                    page_text = f"Page {pages[0]}"

                else:

                    page_text = (
                        "Pages "
                        + ", ".join(
                            str(page)
                            for page in pages
                        )
                    )

                print(
                    f"\n{rank}. "
                    f"{document} "
                    f"({page_text})"
                )