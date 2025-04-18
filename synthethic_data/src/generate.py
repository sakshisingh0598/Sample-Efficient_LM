import json
import sys
import re
import time
from pathlib import Path
import itertools

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from utils import GEMINI_API_KEYS, GEMINI_MODEL

# Embedded screenshot snippet:
embedded_image_text = """
max_vocab_id = list(vocab.keys())[-1]
tokenizer.special_tokens = {
    max_vocab_id+1: "<startoftext>",
    max_vocab_id+2: "<separator>",
    max_vocab_id+3: "<endoftext>",
    max_vocab_id+4: "<unk>",
}
"""

# CONFIGURATION
PROMPTS_DIR   = Path(__file__).parent.parent / "prompts"
OUTPUT_FILE   = Path(__file__).parent.parent / "outputs" / "gemini_dialogues.json"
PERSONA_COUNT = 30      # total personas
RETRIES       = 3       # retry on parse failure only


def load_personas(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip()]


def construct_system_prompt(personas: list[str]) -> str:
    return (
        "SYSTEM:\n"
        "IMPORTANT: Return ONLY a single valid JSON object (start '{' and end '}') per request. "
        "NO markdown, NO extra text.\n"
        "You are an expert conversational AI generator. Generate each conversation in natural Hinglish, "
        "code‑switching between Hindi and English. Keep utterances idiomatic and under 40 words per turn.\n\n"
        "Personas & Scenarios (one per request):\n"
        + "\n".join(personas)
    )


def call_gemini(system_p: str, user_p: str, api_key: str) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    cfg = genai.types.GenerationConfig(
        temperature=0.0,
        max_output_tokens=1024
    )

    resp = model.generate_content(f"{system_p}\n\n{user_p}", generation_config=cfg)
    return resp.text or "{}"


def process_raw_output(raw: str, src: str) -> dict:
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    if not cleaned.startswith("{"):
        cleaned = "{" + cleaned
    if not cleaned.endswith("}"):
        cleaned += "}"
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse failed for {src}: {e}", file=sys.stderr)
        return {}


def safe_generate(system_p: str, user_p: str, src: str, key_cycle: itertools.cycle) -> dict:
    current_key = next(key_cycle)
    keys_tried = {current_key}

    for attempt in range(1, RETRIES + 1):
        while True: # Loop for key rotation
            try:
                print(f"\tUsing key ending with ...{current_key[-4:]}", file=sys.stderr)
                raw = call_gemini(system_p, user_p, current_key)
                break # Break key rotation loop on success
            except ResourceExhausted:
                print(f"\t⚠️ Rate limit hit for key ...{current_key[-4:]}", file=sys.stderr)
                next_key = next(key_cycle)
                if next_key in keys_tried:
                    print(f"⚠️ All API keys exhausted for {src}, skipping.", file=sys.stderr)
                    return {} # All keys tried, skip this persona
                current_key = next_key
                keys_tried.add(current_key)
                print(f"\t🔄 Retrying with next key ...{current_key[-4:]}", file=sys.stderr)
                time.sleep(1) # Small delay before retrying with new key
            except Exception as e:
                print(f"❌ Unexpected error during API call for {src}: {e}", file=sys.stderr)
                return {} # Skip on other unexpected errors

        obj = process_raw_output(raw, src)
        if obj:
            return obj

        print(f"⚠️ Empty or invalid JSON for {src}, retry {attempt}/{RETRIES}", file=sys.stderr)
        time.sleep(2)
    return {}


def main():
    prompt_file = PROMPTS_DIR / "personas_scenarios.txt"
    personas = load_personas(prompt_file)
    system_p = construct_system_prompt(personas)
    key_cycle = itertools.cycle(GEMINI_API_KEYS)

    results = []
    for idx, line in enumerate(personas, start=1):
        user_p = (
            f"USER:\nGenerate a single JSON object for Persona #{idx}:\n"
            f"{line}\n"
            "Format:\n"
            "{\n"
            '  \"persona\": \"<Name>\",\n'
            '  \"scenario\": \"<Short scenario>\",\n'
            '  \"dialogue\": [\n'
            '    {\"speaker\":\"<Name>\", \"text\":\"…\"},\n'
            '    {\"speaker\":\"Interlocutor\",\"text\":\"…\"},\n'
            "    …6 turns total…\n"
            "  ]\n"
            "}"
        )
        src = f"Persona #{idx}"
        print(f"→ Requesting {src}", file=sys.stderr)
        obj = safe_generate(system_p, user_p, src, key_cycle)
        if obj:
            # include embedded image snippet in output
            obj["image_text"] = embedded_image_text
            results.append(obj)
        else:
            print(f"⚠️ Skipping {src}", file=sys.stderr)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote {len(results)} dialogues to {OUTPUT_FILE}", file=sys.stderr)


if __name__ == "__main__":
    main()
