# from app.retrieval.retriever import retrieve_documents
from app.retrieval.hybrid_retriever import hybrid_search
from app.generation.prompt_builder import build_prompt
from app.llm.gemini_service import generate_answer
from app.evaluation.answer_evaluator import evaluate_answer

# def ask_question(question):
#     """
#     Complete RAG Pipeline
#     """

#     print("\nSearching knowledge base...\n")

#     # retrieval_result = retrieve_documents(question)
#     retrieval_result = hybrid_search(question)

#     # ==========================================================
#     # No Answer Found
#     # ==========================================================

#     if retrieval_result["status"] == "no_answer_found":

#         print("\n" + "=" * 80)
#         print("FINAL RESPONSE")
#         print("=" * 80)

#         print(retrieval_result["message"])

#         print("\nDocuments Checked:")

#         source_pages = {}

#         for metadata in retrieval_result["metadatas"]:

#             document = metadata["document_name"]
#             page = metadata["page_number"]

#             if document not in source_pages:
#                 source_pages[document] = []

#             source_pages[document].append(page)

#         for document, pages in source_pages.items():

#             unique_pages = sorted(set(pages))
#             page_text = ", ".join(str(page) for page in unique_pages)

#             print(f"- {document} (Pages: {page_text})")

#         return

#     # ==========================================================
#     # Build Prompt
#     # ==========================================================

#     prompt = build_prompt(
#         question,
#         retrieval_result["documents"]
#     )

#     print("\nSending context to Gemini...\n")

#     answer = generate_answer(prompt)

#     # ==========================================================
#     # Display Final Answer
#     # ==========================================================

#     print("\n" + "=" * 80)
#     print("FINAL ANSWER")
#     print("=" * 80)

#     print(answer)

#     # ==========================================================
#     # Display Sources
#     # ==========================================================

#     print("\n" + "=" * 80)
#     print("SOURCES")
#     print("=" * 80)

#     source_pages = {}

#     for metadata in retrieval_result["metadatas"]:

#         document = metadata["document_name"]
#         page = metadata["page_number"]

#         if document not in source_pages:
#             source_pages[document] = []

#         source_pages[document].append(page)

#     for document, pages in source_pages.items():

#         unique_pages = sorted(set(pages))
#         page_text = ", ".join(str(page) for page in unique_pages)

#         print(f"{document} (Pages: {page_text})")

def ask_question(question):

    print("\nSearching knowledge base...\n")

    # Run hybrid search
    retrieval_result = hybrid_search(question)
    print("\n" + "=" * 80)
    print("HYBRID SEARCH RESULTS (RRF)")
    print("=" * 80)

    for rank, result in enumerate(retrieval_result, start=1):
        print(f"\nRank : {rank}")
        print("-" * 80)
        print(f"Source   : {result.get('source', 'Hybrid')}")
        print(f"Document : {result['document_name']}")
        print(f"Page     : {result['page_number']}")
        print(f"Distance : {result.get('distance', 'N/A')}")

    # No relevant documents found
    if not retrieval_result:
        print("\nNo sufficiently relevant information found.\n")

        print("=" * 80)
        print("FINAL RESPONSE")
        print("=" * 80)

        print(
            "I couldn't find sufficient information "
            "in the uploaded photonics documents."
        )

        return

    # Extract retrieved text
    documents = [
        result["text"]
        for result in retrieval_result
    ]

    # Build context for Gemini
    context = "\n\n".join(documents)

    prompt = f"""
You are a Photonics expert.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:

"I couldn't find sufficient information in the uploaded photonics documents."

Do not use outside knowledge.

Context:
{context}

Question:
{question}

Provide a clear and concise answer.
"""

    print("\nSending context to Gemini...\n")

    answer = generate_answer(prompt)

    # Display final answer
    print("=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)

    print(answer)

    # ==========================================================
    # GEMMA ANSWER EVALUATION
    # ==========================================================

    print("\n")
    print("=" * 80)
    print("GEMMA ANSWER EVALUATION")
    print("=" * 80)

    evaluation = evaluate_answer(
        question,
        context,
        answer
    )

    print(f"\nFaithfulness : {evaluation.get('faithfulness', 'N/A')}/5")
    print(f"Relevance    : {evaluation.get('relevance', 'N/A')}/5")
    print(f"Completeness : {evaluation.get('completeness', 'N/A')}/5")
    print(f"Overall Score: {evaluation.get('overall_score', 'N/A')}/5")

    print(
        f"\nReason: "
        f"{evaluation.get('reason', 'N/A')}"
    )

    # Display sources
    print("\n")
    print("=" * 80)
    print("SOURCES")
    print("=" * 80)

    # seen_sources = set()

    # for result in retrieval_result:

    #     source = (
    #         result["document_name"],
    #         result["page_number"]
    #     )

    #     if source not in seen_sources:

    #         print(
    #             f"{result['document_name']} "
    #             f"(Page {result['page_number']})"
    #         )

    #         seen_sources.add(source)

    # Group pages by document
    sources = {}

    for result in retrieval_result:

        document = result["document_name"]
        page = result["page_number"]

        if document not in sources:
            sources[document] = set()

        sources[document].add(page)


    # Display grouped sources
    print("\n")
    print("=" * 80)
    print("SOURCES")
    print("=" * 80)

    for document, pages in sources.items():

        page_list = sorted(pages)

        print(f"\n{document}")
        print(f"  Pages: {', '.join(map(str, page_list))}")


def main():

    print("=" * 80)
    print("PHOTONICS RAG SYSTEM")
    print("=" * 80)

    while True:

        question = input("\nAsk a Photonics Question (type 'exit' to quit): ")

        # Exit
        if question.lower() == "exit":
            print("\nThank you for using the Photonics RAG System!")
            break

        # Empty Question
        if question.strip() == "":
            print("Please enter a valid question.")
            continue

        ask_question(question)


if __name__ == "__main__":
    main()