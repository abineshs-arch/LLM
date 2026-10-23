"""Small helper around the Gemini API (Interactions API, google-genai SDK)."""
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()                      # reads GEMINI_API_KEY from the .env file
client = genai.Client()
MODEL = "gemini-3.8-flash"         # free tier on the Gemini API


def ask(prompt, system=None, retries=3):
    """Send one prompt. Returns (answer_text, input_tokens, output_tokens, seconds).
    If the connection drops, wait and try again (up to `retries` times)."""
    kwargs = {"model": MODEL, "input": prompt,
              "generation_config": {"thinking_level": "low"}}   # less "thinking" = faster replies
    if system:
        kwargs["system_instruction"] = system
    for attempt in range(1, retries + 1):
        try:
            start = time.time()
            interaction = client.interactions.create(**kwargs)
            seconds = time.time() - start
            break
        except Exception as e:
            print(f"  Attempt {attempt} failed: {type(e).__name__} - retrying in 15 s...")
            if attempt == retries:
                raise
            time.sleep(15)
    usage = getattr(interaction, "usage", None)
    in_tok = getattr(usage, "total_input_tokens", 0) or 0
    out_tok = (getattr(usage, "total_output_tokens", 0) or 0) + (getattr(usage, "total_thought_tokens", 0) or 0)
    time.sleep(4)                  # stay under the free-tier requests-per-minute limit
    return interaction.output_text, in_tok, out_tok, seconds


def md(text):
    """Make text safe for one cell of a Markdown table."""
    return str(text).strip().replace("|", "\\|").replace("\n", "<br>")


if __name__ == "__main__":
    answer, i, o, s = ask("Say hello in one short sentence.")
    print(answer)
    print(f"{i} input tokens, {o} output tokens, {s:.1f} seconds")