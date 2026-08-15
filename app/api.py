from fastapi import FastAPI
from pydantic import BaseModel

# from app.retrieval.retriever import retrieve_documents
from app.retrieval.hybrid_retriever import hybrid_search
from app.generation.prompt_builder import build_prompt
from app.llm.gemini_service import generate_answer

app = FastAPI()


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Photonics RAG API is running!"}


# @app.post("/ask")
# def ask_question(request: QuestionRequest):

#     # Step 1: Retrieve relevant documents
#     retrieval_result = retrieve_documents(request.question)

#     # Step 2: Handle "I don't know" case
#     if retrieval_result["status"] == "no_answer_found":

#         source_pages = {}

#         for metadata in retrieval_result["metadatas"]:

#             document = metadata["document_name"]
#             page = metadata["page_number"]

#             if document not in source_pages:
#                 source_pages[document] = []

#             source_pages[document].append(page)

#         sources = []

#         for document, pages in source_pages.items():

#             sources.append({
#                 "document": document,
#                 "pages": sorted(set(pages))
#             })

#         return {
#             "status": "no_answer_found",
#             "question": request.question,
#             "confidence": retrieval_result["confidence"],
#             "message": retrieval_result["message"],
#             "sources": sources
#         }

#     # Step 3: Build Prompt
#     prompt = build_prompt(
#         request.question,
#         retrieval_result["documents"]
#     )

#     # Step 4: Generate Answer
#     answer = generate_answer(prompt)

#     # Step 5: Format Sources
#     source_pages = {}

#     for metadata in retrieval_result["metadatas"]:

#         document = metadata["document_name"]
#         page = metadata["page_number"]

#         if document not in source_pages:
#             source_pages[document] = []

#         source_pages[document].append(page)

#     sources = []

#     for document, pages in source_pages.items():

#         sources.append({
#             "document": document,
#             "pages": sorted(set(pages))
#         })

#     # Step 6: Return JSON Response
#     return {
#         "status": "success",
#         "question": request.question,
#         "confidence": retrieval_result["confidence"],
#         "answer": answer,
#         "sources": sources
#     }

@app.post("/ask")
def ask_question(request: QuestionRequest):

    # Step 1: Retrieve relevant documents
    # retrieved_docs = retrieve_documents(request.question)
    retrieved_docs = hybrid_search(request.question)

    # Step 2: No relevant documents found
    if len(retrieved_docs) == 0:
        return {
            "status": "no_answer_found",
            "question": request.question,
            "confidence": 0.0,
            "message": "I couldn't find sufficient information in the uploaded photonics documents.",
            "sources": []
        }

    # Step 3: Build prompt
    documents = [doc["text"] for doc in retrieved_docs]

    prompt = build_prompt(
        request.question,
        documents
    )

    # Step 4: Generate answer
    answer = generate_answer(prompt)

    # Step 5: Build sources
    source_pages = {}

    for doc in retrieved_docs:

        document = doc["document_name"]
        page = doc["page_number"]

        if document not in source_pages:
            source_pages[document] = []

        source_pages[document].append(page)

    sources = []

    for document, pages in source_pages.items():

        sources.append(
            {
                "document": document,
                "pages": sorted(set(pages))
            }
        )

    # Step 6: Confidence score
    confidence = round(
        1 - min(doc["distance"] for doc in retrieved_docs),
        3
    )

    # Step 7: Return response
    return {
        "status": "success",
        "question": request.question,
        "confidence": confidence,
        "answer": answer,
        "sources": sources
    }