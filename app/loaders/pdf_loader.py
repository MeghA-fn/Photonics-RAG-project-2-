from pathlib import Path
import fitz  # PyMuPDF
import pandas as pd


def load_pdfs(documents_folder):
    """
    Reads all PDF files from the documents folder
    and extracts text page by page.
    """

    extracted_pages = []

    pdf_files = Path(documents_folder).glob("*.pdf")

    for pdf_file in pdf_files:

        print(f"Reading: {pdf_file.name}")

        document = fitz.open(pdf_file)

        for page_number in range(len(document)):

            page = document.load_page(page_number)

            text = page.get_text()

            extracted_pages.append(
            {
              "document_name": pdf_file.name,
              "page_number": page_number + 1,
              "text": text,
              "character_count": len(text),
              "source": "pdf"
            }
            )

        document.close()

    return extracted_pages

if __name__ == "__main__":

    pages = load_pdfs("documents")

    print(f"\nTotal pages extracted: {len(pages)}")

    # Convert to DataFrame
    df = pd.DataFrame(pages)

    # Create output folder if it doesn't exist
    output_folder = Path("data/raw")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Save CSV
    output_file = output_folder / "extracted_pages.csv"
    df.to_csv(output_file, index=False)

    print(f"\nData saved successfully to: {output_file}")

    print("\nFirst 5 rows:\n")
    print(df.head())