import os
import time
from dotenv import load_dotenv
from google import genai

# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Gemini API Key not found in .env")


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(prompt):

    """
    Generate an answer using Gemini 3.6 Flash.

    Retries only temporary failures.
    Quota errors are reported immediately.
    """

    max_retries = 3

    for attempt in range(max_retries):

        try:

            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )

            if not response or not response.text:
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            return response.text.strip()

        except Exception as e:

            error_message = str(e)

            print(
                f"\nGemini request failed "
                f"(attempt {attempt + 1}/{max_retries})"
            )

            print(f"Error: {error_message}")

            # ------------------------------------------------
            # DO NOT RETRY QUOTA ERRORS
            # ------------------------------------------------

            if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:

                raise RuntimeError(
                    "Gemini API quota has been exhausted. "
                    "Please wait for the quota to reset or "
                    "check your Gemini API billing/quota."
                ) from e

            # ------------------------------------------------
            # RETRY TEMPORARY ERRORS
            # ------------------------------------------------

            if attempt < max_retries - 1:

                wait_time = 5 * (attempt + 1)

                print(
                    f"Retrying Gemini in "
                    f"{wait_time} seconds..."
                )

                time.sleep(wait_time)

            else:

                raise


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    prompt = """
You are a Photonics Research Assistant.

Answer using the supplied context.

Question:
What is stimulated emission?

Answer briefly.
"""

    answer = generate_answer(prompt)

    print("=" * 80)
    print("GEMINI RESPONSE")
    print("=" * 80)
    print(answer)