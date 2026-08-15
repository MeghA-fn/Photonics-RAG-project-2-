import os
from dotenv import load_dotenv
from google import genai
import time

# Load .env
load_dotenv()

# Read API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Gemini API Key not found in .env")

# Create Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)


def generate_answer(prompt):
    """
    Sends the prompt to Gemini and returns the generated answer.
    Retries automatically if Gemini temporarily returns a server error.
    """

    max_retries = 3

    for attempt in range(max_retries):

        try:

            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt
            )

            return response.text

        except Exception as e:

            print(
                f"\nGemini request failed "
                f"(attempt {attempt + 1}/{max_retries})"
            )

            print(f"Error: {e}")

            if attempt < max_retries - 1:

                wait_time = 5 * (attempt + 1)

                print(
                    f"Retrying Gemini in {wait_time} seconds..."
                )

                time.sleep(wait_time)

            else:

                raise


# -------------------------
# Test Gemini
# -------------------------
if __name__ == "__main__":

    prompt = """
You are a Photonics Research Assistant.

Question:
What is stimulated emission?

Answer briefly.
"""

    answer = generate_answer(prompt)

    print("=" * 80)
    print("GEMINI RESPONSE")
    print("=" * 80)
    print(answer)


# import os
# from dotenv import load_dotenv
# from google import genai

# load_dotenv()

# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# for model in client.models.list():
#     print(model.name)