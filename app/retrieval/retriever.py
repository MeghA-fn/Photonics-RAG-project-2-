# import chromadb
# from sentence_transformers import SentenceTransformer

# # =============================================================================
# # Load Embedding Model ONLY ONCE
# # =============================================================================

# print("Loading embedding model...")
# model = SentenceTransformer("all-MiniLM-L6-v2")
# print("Embedding model loaded.")

# # =============================================================================
# # Connect to ChromaDB ONLY ONCE
# # =============================================================================

# print("Connecting to ChromaDB...")
# client = chromadb.PersistentClient(path="vector_db")
# collection = client.get_collection("photonics_rag")
# print("Connected to ChromaDB.")

# # =============================================================================
# # Configuration
# # =============================================================================

# CONFIDENCE_THRESHOLD = 0.80
# TOP_K = 5


# def retrieve_documents(query):
#     """
#     Retrieve the most relevant chunks from ChromaDB.
#     """

#     # Convert question into embedding
#     query_embedding = model.encode(query).tolist()

#     # Retrieve top matching chunks
#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=TOP_K
#     )

#     documents = results["documents"][0]
#     metadatas = results["metadatas"][0]
#     distances = results["distances"][0]

#     best_distance = distances[0]

#     print("\n" + "=" * 80)
#     print("TOP RETRIEVED CHUNKS")
#     print("=" * 80)

#     for i, (doc, meta, dist) in enumerate(
#         zip(documents, metadatas, distances),
#         start=1
#     ):

#         print(f"\nRank : {i}")
#         print("-" * 80)
#         print(f"Document : {meta['document_name']}")
#         print(f"Page     : {meta['page_number']}")
#         print(f"Distance : {dist:.4f}")

#         print("\nChunk:\n")
#         print(doc)

#         print("-" * 80)

#     print(f"\nBest Distance : {best_distance:.4f}")

#     # ------------------------------------------------------------
#     # Confidence Check
#     # ------------------------------------------------------------

#     if best_distance > CONFIDENCE_THRESHOLD:

#         print("\nNo sufficiently relevant information found.")

#         return {
#             "status": "no_answer_found",
#             "confidence": best_distance,
#             "message": "I couldn't find sufficient information in the uploaded photonics documents.",
#             "metadatas": metadatas
#         }

#     print("\nRelevant information found.")

#     return {
#         "status": "success",
#         "confidence": best_distance,
#         "documents": documents,
#         "metadatas": metadatas,
#         "distances": distances
#     }


# # =============================================================================
# # Test Retriever
# # =============================================================================

# if __name__ == "__main__":

#     while True:

#         question = input("\nEnter your question (type 'exit' to quit): ")

#         if question.lower() == "exit":
#             break

#         response = retrieve_documents(question)

#         print("\n" + "=" * 80)
#         print("FINAL RESPONSE")
#         print("=" * 80)

#         print(response)


import chromadb
from sentence_transformers import SentenceTransformer

# ==========================================================
# CONFIGURATION
# ==========================================================

CHROMA_DB_PATH = "data/chroma_db"
COLLECTION_NAME = "photonics_collection"

# Lower distance = better match
CONFIDENCE_THRESHOLD = 0.80

# ==========================================================
# LOAD MODEL (ONLY ONCE)
# ==========================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Embedding model loaded.")

# ==========================================================
# CONNECT TO CHROMADB (ONLY ONCE)
# ==========================================================

print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

collection = client.get_collection(COLLECTION_NAME)
print(client.list_collections())

print("Connected to ChromaDB.")

# ==========================================================
# RETRIEVE DOCUMENTS
# ==========================================================

def retrieve_documents(query, top_k=5):

    # Create embedding
    query_embedding = embedding_model.encode(query).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print("\n" + "=" * 80)
    print("TOP RETRIEVED CHUNKS")
    print("=" * 80)

    retrieved_chunks = []

    for i, (doc, meta, dist) in enumerate(
        zip(documents, metadatas, distances),
        start=1
    ):

        print(f"\nRank : {i}")
        print("-" * 80)
        print("Document :", meta["document_name"])
        print("Page     :", meta["page_number"])
        print(f"Distance : {dist:.4f}")

        print("\nChunk:\n")
        print(doc[:600])

        retrieved_chunks.append({
            "document_name": meta["document_name"],
            "page_number": meta["page_number"],
            "text": doc,
            "distance": float(dist),
            "source": "Semantic"
        })

    best_distance = distances[0]

    print(f"\nBest Distance : {best_distance:.4f}")

    # Confidence Check
    if best_distance > CONFIDENCE_THRESHOLD:

        print("\nNo sufficiently relevant information found.")

        return {
            "status": "no_answer_found",
            "confidence": float(best_distance),
            "message": "I couldn't find sufficient information in the uploaded photonics documents.",
            "documents_checked": [
                {
                    "document_name": item["document_name"],
                    "page_number": item["page_number"]
                }
                for item in retrieved_chunks
            ]
        }

    print("\nRelevant information found.")

    return retrieved_chunks


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    while True:

        question = input("\nEnter your question: ")

        if question.lower() == "exit":
            break

        output = retrieve_documents(question)

        print("\n")
        print("=" * 80)
        print("RETURN VALUE")
        print("=" * 80)
        print(output)