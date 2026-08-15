from pathlib import Path
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(input_file):
    """
    Reads extracted pages from a CSV file,
    splits the text into smaller chunks,
    and saves the chunks to a new CSV file.
    """

    # Read extracted pages
    df = pd.read_csv(input_file)

    print(f"Loaded {len(df)} pages.")

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunked_data = []
    chunk_id = 1

    # Process each page
    for _, row in df.iterrows():

        document_name = row["document_name"]
        page_number = row["page_number"]
        text = row["text"]

        # Skip empty pages
        if pd.isna(text):
            continue

        # Split the page into chunks
        chunks = text_splitter.split_text(text)

        # Store each chunk with metadata
        for chunk in chunks:
            chunked_data.append(
                {
                    "chunk_id": chunk_id,
                    "document_name": document_name,
                    "page_number": page_number,
                    "chunk_text": chunk,
                    "character_count": len(chunk)
                }
            )
            chunk_id += 1

    # Convert to DataFrame
    chunks_df = pd.DataFrame(chunked_data)

    # Create output folder
    output_folder = Path("data/processed")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Save chunks
    output_file = output_folder / "chunks.csv"
    chunks_df.to_csv(output_file, index=False)

    print(f"\nTotal chunks created: {len(chunks_df)}")
    print(f"Chunks saved to: {output_file}")

    print("\nFirst 5 chunks:\n")
    print(chunks_df.head())


if __name__ == "__main__":

    create_chunks("data/raw/extracted_pages.csv")