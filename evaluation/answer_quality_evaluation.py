# from pathlib import Path
# import sys
# import os
# import re
# import time
# import subprocess
# import pandas as pd
# import google.generativeai as genai
# from dotenv import load_dotenv

# load_dotenv()


# # ============================================================
# # PROJECT ROOT
# # ============================================================

# PROJECT_ROOT = Path(__file__).resolve().parents[1]

# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))


# # ============================================================
# # IMPORT HYBRID RETRIEVER
# # ============================================================

# from app.retrieval.hybrid_retriever import hybrid_search


# # ============================================================
# # FILE PATHS
# # ============================================================

# QUESTIONS_FILE = (
#     PROJECT_ROOT
#     / "evaluation"
#     / "rag_evaluation_questions.csv"
# )

# RESULTS_FILE = (
#     PROJECT_ROOT
#     / "evaluation"
#     / "answer_quality_results.csv"
# )


# # ============================================================
# # GEMINI CONFIGURATION
# # ============================================================

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# if not GEMINI_API_KEY:
#     raise RuntimeError(
#         "GEMINI_API_KEY is not set.\n"
#         "Set it in PowerShell before running this script."
#     )

# genai.configure(api_key=GEMINI_API_KEY)

# MODEL_NAME = "gemini-3.6-flash"

# model = genai.GenerativeModel(MODEL_NAME)


# # ============================================================
# # SETTINGS
# # ============================================================

# TOP_K = 5

# OLLAMA_EXE = (
#     r"C:\Users\megha\AppData\Local\Programs\Ollama\ollama.exe"
# )

# OLLAMA_MODEL = "llama3.2:1b"


# # ============================================================
# # SAFE TEXT EXTRACTION
# # ============================================================

# def get_text(item):

#     if not isinstance(item, dict):
#         return ""

#     return str(
#         item.get("text")
#         or item.get("document")
#         or item.get("chunk")
#         or ""
#     )


# def get_document(item):

#     if not isinstance(item, dict):
#         return ""

#     return str(
#         item.get("document_name")
#         or item.get("source")
#         or ""
#     )


# def get_page(item):

#     if not isinstance(item, dict):
#         return ""

#     value = (
#         item.get("page_number")
#         or item.get("page")
#         or ""
#     )

#     return str(value)


# # ============================================================
# # BUILD CONTEXT
# # ============================================================

# def build_context(results):

#     context_parts = []

#     for rank, item in enumerate(
#         results[:TOP_K],
#         start=1
#     ):

#         text = get_text(item)
#         document = get_document(item)
#         page = get_page(item)

#         if not text:
#             continue

#         context_parts.append(
#             f"""
# SOURCE {rank}

# Document: {document}
# Page: {page}

# {text}
# """
#         )

#     return "\n".join(context_parts)


# # ============================================================
# # GENERATE ANSWER USING GEMINI
# # ============================================================

# def generate_answer(question, context):

#     prompt = f"""
# You are a scientific assistant specialized in photonics.

# Answer the user's question using ONLY the supplied context.

# Do not use outside knowledge.

# If the context does not contain enough information to answer
# the question, say:

# "I don't know based on the provided documents."

# Keep the answer concise and scientifically accurate.

# Question:
# {question}

# Context:
# {context}

# Answer:
# """

#     try:

#         response = model.generate_content(prompt)

#         if not response or not response.text:
#             return ""

#         return response.text.strip()

#     except Exception as e:

#         print("\nERROR generating answer:")
#         print(e)

#         return ""


# # ============================================================
# # EVALUATE ANSWER USING LOCAL LLAMA
# # ============================================================

# def evaluate_with_llama(
#     question,
#     expected_answer,
#     context,
#     answer
# ):

#     prompt = f"""
# You are a strict evaluator for a scientific Photonics RAG system.

# You must evaluate BOTH:
# 1. The quality of the retrieved context.
# 2. The quality of the generated answer.

# Use the expected answer as the reference for what important
# information the answer should contain.

# Do not use outside knowledge.

# ============================================================
# QUESTION
# ============================================================

# {question}

# ============================================================
# EXPECTED ANSWER
# ============================================================

# {expected_answer}

# ============================================================
# RETRIEVED CONTEXT
# ============================================================

# {context}

# ============================================================
# GENERATED ANSWER
# ============================================================

# {answer}

# ============================================================
# EVALUATION CRITERIA
# ============================================================

# 1. RETRIEVAL_QUALITY:

# Evaluate whether the retrieved context contains information
# that is relevant and sufficient for answering the question.

# Score:
# 5 = Highly relevant and sufficient context.
# 4 = Mostly relevant and sufficient, with minor missing information.
# 3 = Partially relevant but important information is missing.
# 2 = Mostly irrelevant or insufficient context.
# 1 = Context does not meaningfully support the question.

# 2. FAITHFULNESS:

# Does the generated answer contain claims supported by the
# retrieved context?

# Score:
# 5 = Fully supported by the retrieved context.
# 4 = Mostly supported, with very minor unsupported wording.
# 3 = Partially supported, with some unsupported claims.
# 2 = Several unsupported claims.
# 1 = Mostly unsupported or contradicts the context.

# 3. RELEVANCE:

# Does the generated answer directly answer the question?

# Score:
# 5 = Direct, focused, and clearly answers the question.
# 4 = Mostly direct with minor unnecessary information.
# 3 = Partially answers the question.
# 2 = Mostly off-topic.
# 1 = Does not answer the question.

# 4. COMPLETENESS:

# Compare the generated answer with the expected answer.

# Score:
# 5 = Covers essentially all important points from the expected answer.
# 4 = Covers most important points, with minor omissions.
# 3 = Covers some important points but misses significant information.
# 2 = Covers very little of the expected answer.
# 1 = Does not provide the expected information.

# IMPORTANT:
# Do NOT give a high completeness score simply because the
# generated answer is supported by the retrieved context.

# If the retrieved context itself is incomplete for the question,
# Retrieval Quality should be reduced.

# If the generated answer misses important information from the
# Expected Answer, Completeness should be reduced.

# 5. OVERALL:

# Give an overall score considering retrieval quality and answer
# quality.

# Use your judgment based on the four previous criteria.

# ============================================================
# OUTPUT FORMAT
# ============================================================

# Return ONLY ONE LINE in EXACTLY this format:

# RETRIEVAL_QUALITY=5 FAITHFULNESS=5 RELEVANCE=5 COMPLETENESS=5 OVERALL=5

# Rules:

# - Replace the numbers with your actual scores.
# - All scores must be integers from 1 to 5.
# - Do not add explanations.
# - Do not use JSON.
# - Do not use markdown.
# - Do not add any other text.
# """

#     try:

#         result = subprocess.run(
#             [
#                 OLLAMA_EXE,
#                 "run",
#                 OLLAMA_MODEL,
#                 prompt
#             ],
#             capture_output=True,
#             text=True,
#             encoding="utf-8",
#             errors="replace",
#             timeout=180
#         )

#         raw_output = result.stdout.strip()

#         print("\nLlama evaluation output:")
#         print(raw_output)

#         # ----------------------------------------------------
#         # Extract scores
#         # ----------------------------------------------------

#         retrieval_match = re.search(
#             r"RETRIEVAL_QUALITY\s*=\s*([1-5])",
#             raw_output,
#             re.IGNORECASE
#         )

#         faithfulness_match = re.search(
#             r"FAITHFULNESS\s*=\s*([1-5])",
#             raw_output,
#             re.IGNORECASE
#         )

#         relevance_match = re.search(
#             r"RELEVANCE\s*=\s*([1-5])",
#             raw_output,
#             re.IGNORECASE
#         )

#         completeness_match = re.search(
#             r"COMPLETENESS\s*=\s*([1-5])",
#             raw_output,
#             re.IGNORECASE
#         )

#         overall_match = re.search(
#             r"OVERALL\s*=\s*([1-5])",
#             raw_output,
#             re.IGNORECASE
#         )

#         retrieval_quality = (
#             retrieval_match.group(1)
#             if retrieval_match
#             else ""
#         )

#         faithfulness = (
#             faithfulness_match.group(1)
#             if faithfulness_match
#             else ""
#         )

#         relevance = (
#             relevance_match.group(1)
#             if relevance_match
#             else ""
#         )

#         completeness = (
#             completeness_match.group(1)
#             if completeness_match
#             else ""
#         )

#         overall = (
#             overall_match.group(1)
#             if overall_match
#             else ""
#         )

#         # ----------------------------------------------------
#         # Validate extraction
#         # ----------------------------------------------------

#         if not all([
#             retrieval_quality,
#             faithfulness,
#             relevance,
#             completeness,
#             overall
#         ]):

#             return {
#                 "retrieval_quality": "",
#                 "faithfulness": "",
#                 "relevance": "",
#                 "completeness": "",
#                 "overall": "",
#                 "reason":
#                     f"Could not parse Llama output: {raw_output}"
#             }

#         return {
#             "retrieval_quality": retrieval_quality,
#             "faithfulness": faithfulness,
#             "relevance": relevance,
#             "completeness": completeness,
#             "overall": overall,
#             "reason": "Evaluated by Llama 3.2:1B."
#         }

#     except subprocess.TimeoutExpired:

#         return {
#             "retrieval_quality": "",
#             "faithfulness": "",
#             "relevance": "",
#             "completeness": "",
#             "overall": "",
#             "reason":
#                 "Llama evaluation timed out."
#         }

#     except Exception as e:

#         return {
#             "retrieval_quality": "",
#             "faithfulness": "",
#             "relevance": "",
#             "completeness": "",
#             "overall": "",
#             "reason":
#                 f"Llama evaluation error: {e}"
#         }


# # ============================================================
# # MAIN
# # ============================================================

# def main():

#     print("=" * 80)
#     print("PHOTONICS RAG - ANSWER QUALITY EVALUATION")
#     print("=" * 80)

#     # --------------------------------------------------------
#     # LOAD QUESTIONS
#     # --------------------------------------------------------

#     questions_df = pd.read_csv(
#         QUESTIONS_FILE
#     )

#     # --------------------------------------------------------
#     # CHECK REQUIRED COLUMNS
#     # --------------------------------------------------------

#     required_columns = [
#         "question",
#         "expected_answer"
#     ]

#     missing_columns = [
#         column
#         for column in required_columns
#         if column not in questions_df.columns
#     ]

#     if missing_columns:

#         raise ValueError(
#             "Missing required column(s) in evaluation CSV: "
#             + ", ".join(missing_columns)
#         )

#     print(
#         f"\nLoaded {len(questions_df)} "
#         f"evaluation questions."
#     )

#     # --------------------------------------------------------
#     # RESULTS
#     # --------------------------------------------------------

#     results = []

#     processed = 0
#     answers_generated = 0

#     # --------------------------------------------------------
#     # PROCESS QUESTIONS
#     # --------------------------------------------------------

#     for index, row in questions_df.iterrows():

#         question = str(
#             row["question"]
#         ).strip()

#         expected_answer = str(
#             row["expected_answer"]
#         ).strip()

#         processed += 1

#         print("\n")
#         print("=" * 80)
#         print(
#             f"QUESTION {index + 1}/{len(questions_df)}"
#         )
#         print("=" * 80)

#         print(
#             f"\nQuestion: {question}"
#         )

#         print(
#             f"\nExpected answer:\n{expected_answer}"
#         )

#         # ----------------------------------------------------
#         # RETRIEVAL
#         # ----------------------------------------------------

#         print(
#             "\nRunning hybrid retrieval..."
#         )

#         try:

#             retrieved_results = hybrid_search(
#                 question,
#                 top_k=TOP_K
#             )

#         except TypeError:

#             try:

#                 retrieved_results = hybrid_search(
#                     question,
#                     TOP_K
#                 )

#             except Exception as e:

#                 print(
#                     f"\nRetrieval error: {e}"
#                 )

#                 retrieved_results = []

#         except Exception as e:

#             print(
#                 f"\nRetrieval error: {e}"
#             )

#             retrieved_results = []

#         # ----------------------------------------------------
#         # DISPLAY RETRIEVAL COUNT
#         # ----------------------------------------------------

#         print(
#             f"\nRetrieved {len(retrieved_results)} "
#             f"results."
#         )

#         # ----------------------------------------------------
#         # BUILD CONTEXT
#         # ----------------------------------------------------

#         context = build_context(
#             retrieved_results
#         )

#         # ----------------------------------------------------
#         # GENERATE ANSWER
#         # ----------------------------------------------------

#         print(
#             "\nGenerating answer using Gemini..."
#         )

#         answer = generate_answer(
#             question,
#             context
#         )

#         if answer:

#             answers_generated += 1

#         print(
#             f"\nGenerated answer:\n{answer}"
#         )

#         # ----------------------------------------------------
#         # EVALUATE USING LLAMA
#         # ----------------------------------------------------

#         if answer:

#             print(
#                 "\nEvaluating retrieval and answer using "
#                 "Llama 3.2:1B..."
#             )

#             evaluation = evaluate_with_llama(
#                 question,
#                 expected_answer,
#                 context,
#                 answer
#             )

#         else:

#             evaluation = {
#                 "retrieval_quality": "",
#                 "faithfulness": "",
#                 "relevance": "",
#                 "completeness": "",
#                 "overall": "",
#                 "reason":
#                     "Answer generation failed."
#             }

#         # ----------------------------------------------------
#         # DISPLAY EVALUATION
#         # ----------------------------------------------------

#         print("\nEVALUATION:")

#         print(
#             f"Retrieval Quality : "
#             f"{evaluation['retrieval_quality']}"
#         )

#         print(
#             f"Relevance         : "
#             f"{evaluation['relevance']}"
#         )

#         print(
#             f"Completeness      : "
#             f"{evaluation['completeness']}"
#         )

#         print(
#             f"Faithfulness      : "
#             f"{evaluation['faithfulness']}"
#         )

#         print(
#             f"Overall           : "
#             f"{evaluation['overall']}"
#         )

#         print(
#             f"Reason            : "
#             f"{evaluation['reason']}"
#         )

#         # ----------------------------------------------------
#         # SAVE RESULT
#         # ----------------------------------------------------

#         results.append(
#             {
#                 "question":
#                     question,

#                 "expected_answer":
#                     expected_answer,

#                 "answer":
#                     answer,

#                 "retrieval_quality":
#                     evaluation["retrieval_quality"],

#                 "faithfulness":
#                     evaluation["faithfulness"],

#                 "relevance":
#                     evaluation["relevance"],

#                 "completeness":
#                     evaluation["completeness"],

#                 "overall_score":
#                     evaluation["overall"],

#                 "reason":
#                     evaluation["reason"]
#             }
#         )

#         # Small delay
#         time.sleep(1)

#     # ========================================================
#     # SAVE CSV
#     # ========================================================

#     results_df = pd.DataFrame(
#         results
#     )

#     results_df.to_csv(
#         RESULTS_FILE,
#         index=False
#     )

#     # ========================================================
#     # CALCULATE AVERAGES
#     # ========================================================

#     retrieval_scores = pd.to_numeric(
#         results_df["retrieval_quality"],
#         errors="coerce"
#     )

#     faithfulness_scores = pd.to_numeric(
#         results_df["faithfulness"],
#         errors="coerce"
#     )

#     relevance_scores = pd.to_numeric(
#         results_df["relevance"],
#         errors="coerce"
#     )

#     completeness_scores = pd.to_numeric(
#         results_df["completeness"],
#         errors="coerce"
#     )

#     overall_scores = pd.to_numeric(
#         results_df["overall_score"],
#         errors="coerce"
#     )

#     def calculate_average(series):

#         valid = series.dropna()

#         if len(valid) == 0:
#             return "N/A"

#         return f"{valid.mean():.2f}/5"

#     retrieval_average = calculate_average(
#         retrieval_scores
#     )

#     faithfulness_average = calculate_average(
#         faithfulness_scores
#     )

#     relevance_average = calculate_average(
#         relevance_scores
#     )

#     completeness_average = calculate_average(
#         completeness_scores
#     )

#     overall_average = calculate_average(
#         overall_scores
#     )

#     # ========================================================
#     # FINAL SUMMARY
#     # ========================================================

#     print("\n")
#     print("=" * 80)
#     print(
#         "ANSWER QUALITY EVALUATION COMPLETE"
#     )
#     print("=" * 80)

#     print(
#         f"Total questions        : "
#         f"{len(questions_df)}"
#     )

#     print(
#         f"Processed              : "
#         f"{processed}"
#     )

#     print(
#         f"Answers generated      : "
#         f"{answers_generated}"
#     )

#     print("\nAVERAGE SCORES")

#     print(
#         f"Retrieval Quality      : "
#         f"{retrieval_average}"
#     )

#     print(
#         f"Faithfulness           : "
#         f"{faithfulness_average}"
#     )

#     print(
#         f"Relevance              : "
#         f"{relevance_average}"
#     )

#     print(
#         f"Completeness           : "
#         f"{completeness_average}"
#     )

#     print(
#         f"Overall                : "
#         f"{overall_average}"
#     )

#     print("\nResults saved to:")

#     print(
#         RESULTS_FILE
#     )

#     print("=" * 80)


# # ============================================================
# # ENTRY POINT
# # ============================================================

# if __name__ == "__main__":
#     main()

import os
import re
import pandas as pd
from app.retrieval.hybrid_retriever import hybrid_search
from app.llm.gemini_service import generate_answer


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "evaluation",
    "rag_evaluation_questions.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "evaluation",
    "answer_quality_results.csv"
)

TOP_K = 5


# =============================================================================
# TEXT CLEANING
# =============================================================================

def clean_text(text):
    """
    Clean text before sending it to the evaluator.
    """

    if text is None:
        return ""

    text = str(text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =============================================================================
# BUILD EVALUATION PROMPT
# =============================================================================

def build_evaluation_prompt(
    question,
    expected_answer,
    generated_answer,
    retrieved_context
):
    """
    Create a structured prompt for Llama/Gemma evaluation.

    The evaluator judges the generated answer using BOTH:
    1. Expected answer
    2. Retrieved context

    This prevents correct answers from being unfairly penalized
    when they contain additional information supported by the documents.
    """

    prompt = f"""
You are evaluating a RAG system for a Photonics knowledge base.

Evaluate the GENERATED ANSWER using the EXPECTED ANSWER and RETRIEVED
CONTEXT.

IMPORTANT RULES:

1. Do NOT require the generated answer to use exactly the same wording
   as the expected answer.

2. Additional information is acceptable if it is supported by the
   retrieved context.

3. Do NOT penalize an answer simply because it contains more information
   than the expected answer.

4. Judge whether the generated answer actually answers the QUESTION.

5. FAITHFULNESS:
   Give a high score when the answer is supported by the retrieved
   context and does not introduce unsupported claims.

6. RELEVANCE:
   Give a high score when the answer directly addresses the question.
   Do not penalize useful supporting information.

7. COMPLETENESS:
   Compare the answer with the expected answer and retrieved context.
   If the important concepts required by the question are present,
   give a high score.

8. RETRIEVAL QUALITY:
   Judge whether the retrieved context contains information useful
   for answering the question.

SCORING:

5 = Excellent
4 = Good
3 = Acceptable
2 = Poor
1 = Very poor

QUESTION:
{question}

EXPECTED ANSWER:
{expected_answer}

GENERATED ANSWER:
{generated_answer}

RETRIEVED CONTEXT:
{retrieved_context}

Return ONLY the following format:

RETRIEVAL_QUALITY=<1-5>
FAITHFULNESS=<1-5>
RELEVANCE=<1-5>
COMPLETENESS=<1-5>
OVERALL=<1-5>
REASON=<short explanation>
"""

    return prompt


# =============================================================================
# PARSE EVALUATION
# =============================================================================

def parse_evaluation(response):
    """
    Extract scores from the evaluator response.
    """

    if response is None:
        return {
            "retrieval_quality": 0,
            "faithfulness": 0,
            "relevance": 0,
            "completeness": 0,
            "overall_score": 0,
            "reason": "No evaluation response."
        }

    response = str(response)

    def extract_score(pattern):
        match = re.search(pattern, response, re.IGNORECASE)

        if match:
            try:
                score = int(match.group(1))

                if 1 <= score <= 5:
                    return score

            except ValueError:
                pass

        return 0

    retrieval_quality = extract_score(
        r"RETRIEVAL[_ ]QUALITY\s*=\s*(\d+)"
    )

    faithfulness = extract_score(
        r"FAITHFULNESS\s*=\s*(\d+)"
    )

    relevance = extract_score(
        r"RELEVANCE\s*=\s*(\d+)"
    )

    completeness = extract_score(
        r"COMPLETENESS\s*=\s*(\d+)"
    )

    overall = extract_score(
        r"OVERALL\s*=\s*(\d+)"
    )

    reason_match = re.search(
        r"REASON\s*=\s*(.*)",
        response,
        re.IGNORECASE | re.DOTALL
    )

    if reason_match:
        reason = reason_match.group(1).strip()
    else:
        reason = "No reason provided."

    return {
        "retrieval_quality": retrieval_quality,
        "faithfulness": faithfulness,
        "relevance": relevance,
        "completeness": completeness,
        "overall_score": overall,
        "reason": reason
    }


# =============================================================================
# FORMAT RETRIEVED DOCUMENTS
# =============================================================================

def format_retrieved_context(results):
    """
    Convert retrieved chunks into readable context for the evaluator.
    """

    if not results:
        return "No relevant context was retrieved."

    context_parts = []

    for index, doc in enumerate(results, start=1):

        document_name = doc.get(
            "document_name",
            doc.get("document", "Unknown document")
        )

        page_number = doc.get(
            "page_number",
            doc.get("page", "Unknown")
        )

        text = doc.get("text", "")

        text = clean_text(text)

        if not text:
            continue

        context_parts.append(
            f"""
Context {index}
Document: {document_name}
Page: {page_number}

{text}
"""
        )

    if not context_parts:
        return "No usable text was retrieved."

    return "\n".join(context_parts)


# =============================================================================
# GENERATE PHOTONICS ANSWER
# =============================================================================

def generate_photonics_answer(question, retrieved_results):
    """
    Generate an answer using the retrieved context.
    """

    context = format_retrieved_context(retrieved_results)

    prompt = f"""
You are a Photonics question-answering assistant.

Answer the question using ONLY the information available in the
provided context.

If the context contains enough information, give a clear and concise
answer.

Do not unnecessarily introduce unrelated information.

Question:
{question}

Context:
{context}
"""

    try:
        answer = generate_answer(prompt)

        if answer is None:
            return ""

        return clean_text(answer)

    except Exception as e:
        print(f"Answer generation error: {e}")
        return ""


# =============================================================================
# EVALUATE ONE QUESTION
# =============================================================================

def evaluate_question(question, expected_answer):
    """
    Retrieve documents, generate an answer, and evaluate answer quality.
    """

    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1: HYBRID RETRIEVAL
    # -------------------------------------------------------------------------

    print("\nRunning Hybrid Search...")

    try:
        retrieved_results = hybrid_search(
            question
        )

    except Exception as e:
        print(f"Retrieval error: {e}")
        retrieved_results = []

    # Limit results
    if retrieved_results:
        retrieved_results = retrieved_results[:TOP_K]

    # -------------------------------------------------------------------------
    # STEP 2: DISPLAY RETRIEVED DOCUMENTS
    # -------------------------------------------------------------------------

    print("\nRETRIEVED CONTEXT")
    print("-" * 80)

    if not retrieved_results:
        print("No documents retrieved.")

    else:
        for rank, doc in enumerate(retrieved_results, start=1):

            document_name = doc.get(
                "document_name",
                doc.get("document", "Unknown")
            )

            page_number = doc.get(
                "page_number",
                doc.get("page", "Unknown")
            )

            distance = doc.get(
                "distance",
                "N/A"
            )

            print(f"\nRank : {rank}")
            print(f"Document : {document_name}")
            print(f"Page : {page_number}")
            print(f"Distance : {distance}")

    # -------------------------------------------------------------------------
    # STEP 3: GENERATE ANSWER
    # -------------------------------------------------------------------------

    print("\nGenerating answer...")

    generated_answer = generate_photonics_answer(
        question,
        retrieved_results
    )

    print("\nGENERATED ANSWER")
    print("-" * 80)
    print(generated_answer)

    # -------------------------------------------------------------------------
    # STEP 4: PREPARE EVALUATION CONTEXT
    # -------------------------------------------------------------------------

    retrieved_context = format_retrieved_context(
        retrieved_results
    )

    evaluation_prompt = build_evaluation_prompt(
        question=question,
        expected_answer=expected_answer,
        generated_answer=generated_answer,
        retrieved_context=retrieved_context
    )

    # -------------------------------------------------------------------------
    # STEP 5: RUN LLM EVALUATION
    # -------------------------------------------------------------------------

    print("\nRunning answer quality evaluation...")

    try:

        evaluation_response = generate_answer(
            evaluation_prompt
        )

    except Exception as e:

        print(f"Evaluation error: {e}")

        evaluation_response = ""

    # -------------------------------------------------------------------------
    # STEP 6: PARSE SCORES
    # -------------------------------------------------------------------------

    evaluation = parse_evaluation(
        evaluation_response
    )

    print("\n" + "=" * 80)
    print("ANSWER QUALITY EVALUATION")
    print("=" * 80)

    print(
        f"Retrieval Quality : "
        f"{evaluation['retrieval_quality']}"
    )

    print(
        f"Faithfulness      : "
        f"{evaluation['faithfulness']}"
    )

    print(
        f"Relevance         : "
        f"{evaluation['relevance']}"
    )

    print(
        f"Completeness      : "
        f"{evaluation['completeness']}"
    )

    print(
        f"Overall           : "
        f"{evaluation['overall_score']}"
    )

    print(
        f"Reason            : "
        f"{evaluation['reason']}"
    )

    # -------------------------------------------------------------------------
    # STEP 7: RETURN RESULT
    # -------------------------------------------------------------------------

    return {
        "question": question,
        "expected_answer": expected_answer,
        "answer": generated_answer,
        "retrieval_quality": evaluation["retrieval_quality"],
        "faithfulness": evaluation["faithfulness"],
        "relevance": evaluation["relevance"],
        "completeness": evaluation["completeness"],
        "overall_score": evaluation["overall_score"],
        "reason": evaluation["reason"]
    }


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def main():

    print("\n" + "=" * 80)
    print("PHOTONICSRAG ANSWER QUALITY EVALUATION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # CHECK INPUT FILE
    # -------------------------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        print(
            f"\nERROR: Evaluation file not found:\n"
            f"{INPUT_FILE}"
        )

        return

    # -------------------------------------------------------------------------
    # LOAD QUESTIONS
    # -------------------------------------------------------------------------

    print(
        f"\nLoading evaluation questions from:\n"
        f"{INPUT_FILE}"
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    required_columns = [
        "question",
        "expected_answer"
    ]

    for column in required_columns:

        if column not in df.columns:

            print(
                f"\nERROR: Missing required column: "
                f"{column}"
            )

            return

    print(
        f"Total questions: {len(df)}"
    )

    # -------------------------------------------------------------------------
    # EVALUATE QUESTIONS
    # -------------------------------------------------------------------------

    results = []

    for index, row in df.iterrows():

        question = clean_text(
            row["question"]
        )

        expected_answer = clean_text(
            row["expected_answer"]
        )

        print(
            f"\n\nProcessing question "
            f"{index + 1}/{len(df)}"
        )

        try:

            result = evaluate_question(
                question,
                expected_answer
            )

            results.append(result)

        except Exception as e:

            print(
                f"\nERROR processing question "
                f"{index + 1}: {e}"
            )

            results.append(
                {
                    "question": question,
                    "expected_answer": expected_answer,
                    "answer": "",
                    "retrieval_quality": 0,
                    "faithfulness": 0,
                    "relevance": 0,
                    "completeness": 0,
                    "overall_score": 0,
                    "reason": f"Evaluation failed: {e}"
                }
            )

    # -------------------------------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # -------------------------------------------------------------------------
    # CALCULATE AVERAGES
    # -------------------------------------------------------------------------

    valid_results = results_df[
        results_df["overall_score"] > 0
    ]

    if len(valid_results) > 0:

        avg_retrieval = valid_results[
            "retrieval_quality"
        ].mean()

        avg_faithfulness = valid_results[
            "faithfulness"
        ].mean()

        avg_relevance = valid_results[
            "relevance"
        ].mean()

        avg_completeness = valid_results[
            "completeness"
        ].mean()

        avg_overall = valid_results[
            "overall_score"
        ].mean()

    else:

        avg_retrieval = 0
        avg_faithfulness = 0
        avg_relevance = 0
        avg_completeness = 0
        avg_overall = 0

    # -------------------------------------------------------------------------
    # FINAL REPORT
    # -------------------------------------------------------------------------

    print("\n\n" + "=" * 80)
    print("ANSWER QUALITY EVALUATION COMPLETE")
    print("=" * 80)

    print(
        f"Total questions        : "
        f"{len(df)}"
    )

    print(
        f"Processed              : "
        f"{len(results)}"
    )

    print(
        f"Answers generated     : "
        f"{sum(bool(x['answer']) for x in results)}"
    )

    print("\nAVERAGE SCORES")

    print(
        f"Retrieval Quality      : "
        f"{avg_retrieval:.2f}/5"
    )

    print(
        f"Faithfulness           : "
        f"{avg_faithfulness:.2f}/5"
    )

    print(
        f"Relevance              : "
        f"{avg_relevance:.2f}/5"
    )

    print(
        f"Completeness           : "
        f"{avg_completeness:.2f}/5"
    )

    print(
        f"Overall                : "
        f"{avg_overall:.2f}/5"
    )

    print("\nResults saved to:")

    print(
        OUTPUT_FILE
    )

    print("=" * 80)


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    main()