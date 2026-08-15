# import pandas as pd
# from rank_bm25 import BM25Okapi

# from app.retrieval.retriever import retrieve_documents


# # =====================================================
# # STOP WORDS
# # =====================================================

# STOP_WORDS = {
#     "what", "is", "the", "a", "an", "of", "to", "for",
#     "in", "on", "and", "or", "with", "by", "from",
#     "who", "where", "when", "which", "how", "does",
#     "do", "did", "are", "was", "were"
# }


# # =====================================================
# # LOAD CHUNKS
# # =====================================================

# chunks = pd.read_csv("data/processed/chunks.csv")

# documents = chunks["chunk_text"].fillna("").tolist()


# # =====================================================
# # TOKENIZE DOCUMENTS
# # =====================================================

# tokenized_documents = [
#     [
#         word
#         for word in doc.lower().split()
#         if word not in STOP_WORDS
#     ]
#     for doc in documents
# ]


# # =====================================================
# # BUILD BM25 INDEX
# # =====================================================

# bm25 = BM25Okapi(tokenized_documents)


# # =====================================================
# # BM25 KEYWORD SEARCH
# # =====================================================

# def keyword_search(query, top_k=5):
#     """
#     Perform BM25 keyword-based retrieval.
#     Returns dictionaries in the same format expected by RRF.
#     """

#     tokenized_query = [
#         word
#         for word in query.lower().split()
#         if word not in STOP_WORDS
#     ]

#     scores = bm25.get_scores(tokenized_query)

#     top_indices = sorted(
#         range(len(scores)),
#         key=lambda i: scores[i],
#         reverse=True
#     )[:top_k]

#     results = []

#     for idx in top_indices:

#         row = chunks.iloc[idx]

#         results.append({
#             "chunk_id": str(row["chunk_id"]),
#             "document_name": str(row["document_name"]),
#             "page_number": int(row["page_number"]),
#             "text": str(row["chunk_text"]),
#             "score": float(scores[idx]),
#             "source": "BM25"
#         })

#     return results


# # =====================================================
# # CREATE UNIQUE RESULT KEY
# # =====================================================

# def get_result_key(item):
#     """
#     Create a unique key for a retrieved document chunk.
#     """

#     if not isinstance(item, dict):
#         raise TypeError(
#             f"Invalid retrieval result type: {type(item)}\n"
#             f"Value: {item}"
#         )

#     return (
#         item.get("document_name", ""),
#         item.get("page_number", "")
#     )


# # =====================================================
# # RRF HYBRID SEARCH
# # =====================================================

# def hybrid_search(query, top_k=5):
#     """
#     Combine Semantic Search and BM25 using
#     Reciprocal Rank Fusion (RRF).
#     """

#     # -------------------------------------------------
#     # Semantic Search
#     # -------------------------------------------------

#     print("\nRunning Semantic Search...")

#     semantic_results = retrieve_documents(query)

#     # -------------------------------------------------
#     # BM25 Search
#     # -------------------------------------------------

#     print("Running BM25 Search...")

#     bm25_results = keyword_search(query)
#     print("\nDEBUG BM25 TYPE:")
#     print(type(bm25_results))

#     print("DEBUG FIRST BM25 RESULT:")
#     print(
#         type(bm25_results[0])
#         if bm25_results
#         else "EMPTY"
#     )

#     print(
#         bm25_results[0]
#         if bm25_results
#         else "EMPTY"
#     )

#     # -------------------------------------------------
#     # RRF Configuration
#     # -------------------------------------------------

#     RRF_K = 60

#     rrf_scores = {}

#     # =================================================
#     # ADD SEMANTIC RESULTS
#     # =================================================

#     for rank, item in enumerate(
#         semantic_results,
#         start=1
#     ):

#         key = get_result_key(item)

#         score = 1 / (RRF_K + rank)

#         if key not in rrf_scores:

#             data = item.copy()

#             data["source"] = "Semantic"

#             rrf_scores[key] = {
#                 "data": data,
#                 "score": score,
#                 "sources": ["Semantic"]
#             }

#         else:

#             rrf_scores[key]["score"] += score

#             if (
#                 "Semantic"
#                 not in rrf_scores[key]["sources"]
#             ):

#                 rrf_scores[key]["sources"].append(
#                     "Semantic"
#                 )

#     # =================================================
#     # ADD BM25 RESULTS
#     # =================================================

#     for rank, item in enumerate(
#         bm25,
#         start=1
#     ):

#         key = get_result_key(item)

#         score = 1 / (RRF_K + rank)

#         if key not in rrf_scores:

#             data = item.copy()

#             data["source"] = "BM25"

#             rrf_scores[key] = {
#                 "data": data,
#                 "score": score,
#                 "sources": ["BM25"]
#             }

#         else:

#             rrf_scores[key]["score"] += score

#             if (
#                 "BM25"
#                 not in rrf_scores[key]["sources"]
#             ):

#                 rrf_scores[key]["sources"].append(
#                     "BM25"
#                 )

#     # =================================================
#     # CREATE FINAL RESULTS
#     # =================================================

#     final_results = []

#     for value in rrf_scores.values():

#         result = value["data"].copy()

#         # Store RRF score
#         result["rrf_score"] = float(
#             value["score"]
#         )

#         # Store retrieval sources
#         result["source"] = " + ".join(
#             value["sources"]
#         )

#         final_results.append(result)

#     # =================================================
#     # SORT BY RRF SCORE
#     # =================================================

#     final_results.sort(
#         key=lambda x: x["rrf_score"],
#         reverse=True
#     )

#     # =================================================
#     # RETURN TOP K
#     # =================================================

#     return final_results[:top_k]


# # =====================================================
# # TEST
# # =====================================================

# if __name__ == "__main__":

#     question = input(
#         "Enter your question: "
#     )

#     results = hybrid_search(question)

#     print("\n")
#     print("=" * 80)
#     print("RRF HYBRID SEARCH RESULTS")
#     print("=" * 80)

#     for i, result in enumerate(
#         results,
#         start=1
#     ):

#         print("\n" + "-" * 60)

#         print(f"Rank       : {i}")

#         print(
#             "Source     :",
#             result.get(
#                 "source",
#                 "Unknown"
#             )
#         )

#         print(
#             "Document   :",
#             result.get(
#                 "document_name",
#                 "Unknown"
#             )
#         )

#         print(
#             "Page       :",
#             result.get(
#                 "page_number",
#                 "Unknown"
#             )
#         )

#         print(
#             "RRF Score  :",
#             round(
#                 result.get(
#                     "rrf_score",
#                     0
#                 ),
#                 6
#             )
#         )

#         if "distance" in result:

#             print(
#                 "Semantic Distance :",
#                 round(
#                     result["distance"],
#                     6
#                 )
#             )

#         if "score" in result:

#             print(
#                 "BM25 Score        :",
#                 round(
#                     result["score"],
#                     6
#                 )
#             )

#         print("\nChunk:\n")

#         print(
#             result.get(
#                 "text",
#                 result.get(
#                     "chunk_text",
#                     ""
#                 )
#             )[:500]
#         )

# app/retrieval/hybrid_retriever.py

from pathlib import Path
import re

import pandas as pd
from rank_bm25 import BM25Okapi


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunks.csv"

TOP_K = 5
RRF_K = 60

# Common English stop words
STOP_WORDS = {
    "a", "an", "the",
    "and", "or", "but",
    "is", "are", "was", "were",
    "be", "been", "being",
    "am",
    "of", "to", "in", "on", "at",
    "for", "from", "by", "with",
    "about", "into", "through",
    "during", "before", "after",
    "above", "below",
    "between",
    "this", "that", "these", "those",
    "it", "its",
    "as",
    "what", "which", "who", "whom",
    "when", "where", "why", "how",
    "can", "could", "should", "would",
    "do", "does", "did",
    "has", "have", "had",
    "will", "shall",
    "than",
    "then",
    "also"
}


# ============================================================
# LOAD DATA
# ============================================================

print("Loading chunks for BM25...")

df = pd.read_csv(CHUNKS_FILE)

print(f"Loaded {len(df)} chunks for BM25.")


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):
    """
    Tokenize text and remove common English stop words.

    This preprocessing is used only for BM25.
    Semantic search continues to use the original text.
    """

    if not isinstance(text, str):
        text = ""

    # Lowercase
    text = text.lower()

    # Keep words and numbers
    tokens = re.findall(r"\b[a-zA-Z0-9]+\b", text)

    # Remove stop words
    tokens = [
        token
        for token in tokens
        if token not in STOP_WORDS
    ]

    return tokens


# ============================================================
# PREPARE BM25 CORPUS
# ============================================================

print("Preparing BM25 corpus...")

tokenized_corpus = [
    preprocess_text(text)
    for text in df["chunk_text"]
]

bm25_index = BM25Okapi(tokenized_corpus)

print("BM25 index ready.")


# ============================================================
# BM25 SEARCH
# ============================================================

def bm25_search(query, top_k=TOP_K):
    """
    Perform BM25 lexical search.

    Returns a list of dictionaries with a consistent format.
    """

    print("Running BM25 Search...")

    query_tokens = preprocess_text(query)

    # If query becomes empty after stop-word removal
    if not query_tokens:
        print("BM25 query contains no useful tokens.")
        return []

    # Calculate BM25 scores
    scores = bm25_index.get_scores(query_tokens)

    # Get indices sorted by score, highest first
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for index in ranked_indices[:top_k]:

        score = float(scores[index])

        # Skip zero-score results
        if score <= 0:
            continue

        row = df.iloc[index]

        result = {
            "chunk_id": str(row["chunk_id"]),
            "document_name": str(row["document_name"]),
            "page_number": int(row["page_number"]),
            "text": str(row["chunk_text"]),
            "score": score,
            "source": "BM25"
        }

        results.append(result)

    return results


# ============================================================
# RESULT KEY
# ============================================================

def get_result_key(item):
    """
    Create a stable key for RRF fusion.

    If chunk_id is available, use:
        document + page + chunk_id

    If chunk_id is missing, use:
        document + page

    This allows Semantic Search and BM25 results
    to be compared even when Semantic Search does
    not provide a chunk_id.
    """

    if not isinstance(item, dict):
        raise TypeError(
            f"Invalid retrieval result type: {type(item)}\n"
            f"Value: {item}"
        )

    document = str(
        item.get("document_name", "")
    ).strip()

    page = str(
        item.get("page_number", "")
    ).strip()

    chunk_id = item.get("chunk_id")

    if chunk_id is not None and str(chunk_id).strip():
        return (
            document,
            page,
            str(chunk_id).strip()
        )

    return (
        document,
        page
    )


# ============================================================
# RRF HYBRID SEARCH
# ============================================================

def hybrid_search(
    query,
    semantic_results=None,
    top_k=TOP_K
):
    """
    Combine Semantic Search and BM25 using
    Reciprocal Rank Fusion (RRF).

    RRF formula:

        RRF score = 1 / (k + rank)

    where k = 60 by default.
    """

    # --------------------------------------------------------
    # Semantic Search
    # --------------------------------------------------------

    if semantic_results is None:

        print("\nRunning Semantic Search...")

        # Import here to avoid circular imports
        from app.retrieval.retriever import retrieve_documents

        semantic_results = retrieve_documents(query)

    # --------------------------------------------------------
    # Make sure semantic results are a list
    # --------------------------------------------------------

    if isinstance(semantic_results, dict):

        # Some versions of retriever may return:
        # {"status": "...", "results": [...]}

        if "results" in semantic_results:
            semantic_results = semantic_results["results"]

        else:
            semantic_results = []

    if not isinstance(semantic_results, list):
        semantic_results = []

    # Remove invalid entries
    semantic_results = [
        item
        for item in semantic_results
        if isinstance(item, dict)
    ]

    # --------------------------------------------------------
    # BM25 Search
    # --------------------------------------------------------

    bm25_results = bm25_search(
        query,
        top_k=top_k
    )

    # --------------------------------------------------------
    # RRF SCORE STORAGE
    # --------------------------------------------------------

    rrf_scores = {}

    # ========================================================
    # ADD SEMANTIC RESULTS
    # ========================================================

    for rank, item in enumerate(
        semantic_results,
        start=1
    ):

        key = get_result_key(item)

        score = 1 / (RRF_K + rank)

        if key not in rrf_scores:

            # Make a copy so original result is not modified
            data = item.copy()

            data["source"] = "Semantic"

            rrf_scores[key] = {
                "data": data,
                "score": score
            }

        else:

            rrf_scores[key]["score"] += score

            current_source = rrf_scores[key]["data"].get(
                "source",
                ""
            )

            if "Semantic" not in current_source:
                rrf_scores[key]["data"]["source"] = (
                    current_source + " + Semantic"
                )


    # ========================================================
    # ADD BM25 RESULTS
    # ========================================================

    for rank, item in enumerate(
        bm25_results,
        start=1
    ):

        key = get_result_key(item)

        score = 1 / (RRF_K + rank)

        if key not in rrf_scores:

            data = item.copy()

            data["source"] = "BM25"

            rrf_scores[key] = {
                "data": data,
                "score": score
            }

        else:

            rrf_scores[key]["score"] += score

            current_source = rrf_scores[key]["data"].get(
                "source",
                ""
            )

            if "BM25" not in current_source:

                if current_source:
                    rrf_scores[key]["data"]["source"] = (
                        current_source + " + BM25"
                    )
                else:
                    rrf_scores[key]["data"]["source"] = "BM25"


    # ========================================================
    # SORT BY RRF SCORE
    # ========================================================

    ranked_results = sorted(
        rrf_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    # ========================================================
    # BUILD FINAL RESULTS
    # ========================================================

    final_results = []

    for rank, entry in enumerate(
        ranked_results[:top_k],
        start=1
    ):

        data = entry["data"].copy()

        data["rrf_score"] = entry["score"]

        final_results.append(data)


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")
    print("=" * 80)
    print("RRF HYBRID SEARCH RESULTS")
    print("=" * 80)

    if not final_results:

        print("No hybrid search results found.")
        return []

    for rank, result in enumerate(
        final_results,
        start=1
    ):

        print("\n" + "-" * 60)

        print(f"Rank       : {rank}")

        print(
            f"Source     : "
            f"{result.get('source', 'Unknown')}"
        )

        print(
            f"Document   : "
            f"{result.get('document_name', 'Unknown')}"
        )

        print(
            f"Page       : "
            f"{result.get('page_number', 'Unknown')}"
        )

        print(
            f"RRF Score  : "
            f"{result.get('rrf_score', 0):.6f}"
        )

        if "distance" in result:

            print(
                f"Semantic Distance : "
                f"{result['distance']:.6f}"
            )

        if "score" in result:

            print(
                f"BM25 Score        : "
                f"{result['score']:.6f}"
            )

        print("\nChunk:\n")

        print(
            result.get(
                "text",
                result.get("chunk_text", "")
            )
        )

    return final_results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    question = input(
        "Enter your question: "
    ).strip()

    if question.lower() == "exit":
        print("Exiting...")
        raise SystemExit

    # Import semantic retriever
    from app.retrieval.retriever import retrieve_documents

    print("\nRunning Semantic Search...")

    semantic_results = retrieve_documents(
        question
    )

    # Run hybrid search
    results = hybrid_search(
        question,
        semantic_results=semantic_results,
        top_k=TOP_K
    )

    print("\n")
    print("=" * 80)
    print("HYBRID SEARCH COMPLETE")
    print("=" * 80)