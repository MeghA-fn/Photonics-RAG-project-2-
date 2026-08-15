from pathlib import Path

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


def create_vector_database(input_file):
    """
    Reads chunks from CSV, generates embeddings,
    and stores them in ChromaDB.
    """

    # Load chunks
    df = pd.read_csv(input_file)

    print(f"Loaded {len(df)} chunks.")

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding model loaded.")

    # Create persistent ChromaDB client
    # client = chromadb.PersistentClient(path="vector_db")
    client = chromadb.PersistentClient(path="data/chroma_db")


    # Create (or get) collection
    # collection = client.get_or_create_collection(
    #     name="photonics_rag"
    # )
    
    collection = client.get_or_create_collection(
        name="photonics_collection"
    )
    
    # Add every chunk to ChromaDB
    for _, row in df.iterrows():

        embedding = model.encode(row["chunk_text"]).tolist()

        collection.add(
            ids=[str(row["chunk_id"])],
            embeddings=[embedding],
            documents=[row["chunk_text"]],
            metadatas=[
                {
                    "document_name": row["document_name"],
                    "page_number": int(row["page_number"]),
                    "character_count": int(row["character_count"])
                }
            ]
        )

    print("\nEmbeddings stored successfully!")

    print(f"Total vectors in database: {collection.count()}")


if __name__ == "__main__":

    create_vector_database("data/processed/chunks.csv")