import json
import sys
import re
import time
import random
from pathlib import Path
import itertools

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from utils import GEMINI_API_KEYS, GEMINI_MODEL

# CONFIGURATION
PROMPTS_DIR   = Path(__file__).parent.parent / "prompts"
OUTPUT_FILE   = Path(__file__).parent.parent / "outputs" / "hinglish_dialogues.json"
PERSONA_COUNT = 1500      # total requests
RETRIES       = 3       # retry on parse failure only


def construct_system_prompt() -> str:
    return (
        "SYSTEM:\n"
        "IMPORTANT: Return ONLY a JSON array of message objects per request. NO markdown, NO extra text.\n"
        "Each object must be {\"role\": \"user|assistant\", \"content\": \"...\"}.\n"
        "Generate a single Hinglish conversation on a random topic—like love, taxes, college days, space travel, movie discussion, cheating, etc.—"
        "code‑switching between Hindi and English. Each message must be 40–50 words long."
    )


def call_gemini(system_p: str, user_p: str, api_key: str) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    cfg = genai.types.GenerationConfig(
        temperature=0.0,
        max_output_tokens=1024
    )
    resp = model.generate_content(f"{system_p}\n\n{user_p}", generation_config=cfg)
    return resp.text or "[]"


def process_raw_output(raw: str, src: str) -> list:
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    if not cleaned.startswith("["):
        cleaned = "[" + cleaned.lstrip("{")
    if not cleaned.endswith("]"):
        cleaned += "]"
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse failed for {src}: {e}", file=sys.stderr)
        return []


def safe_generate(system_p: str, user_p: str, src: str, key_cycle: itertools.cycle) -> list:
    current_key = next(key_cycle)
    keys_tried = {current_key}

    for attempt in range(1, RETRIES + 1):
        while True:
            try:
                print(f"\tUsing key ending with ...{current_key[-4:]}", file=sys.stderr)
                raw = call_gemini(system_p, user_p, current_key)
                break
            except ResourceExhausted:
                print(f"\t⚠️ Rate limit hit for key ...{current_key[-4:]}", file=sys.stderr)
                next_key = next(key_cycle)
                if next_key in keys_tried:
                    print(f"⚠️ All API keys exhausted for {src}. Sleeping for 15-20 minutes before retrying.", file=sys.stderr)
                    rest_time = random.uniform(15*60, 20*60)
                    time.sleep(rest_time)
                    # Reset tried keys and start fresh
                    keys_tried.clear()
                    current_key = next(key_cycle)
                    keys_tried.add(current_key)
                    print(f"🔄 Resuming requests with key ending with ...{current_key[-4:]}", file=sys.stderr)
                    continue
                current_key = next_key
                keys_tried.add(current_key)
                time.sleep(1)
            except Exception as e:
                print(f"❌ Unexpected error during API call for {src}: {e}", file=sys.stderr)
                return []

        arr = process_raw_output(raw, src)
        if arr:
            return arr
        print(f"⚠️ Empty or invalid JSON for {src}, retry {attempt}/{RETRIES}", file=sys.stderr)
        time.sleep(2)
    return []


def main():
    system_p = construct_system_prompt()
    key_cycle = itertools.cycle(GEMINI_API_KEYS)

    all_conversations = []
    for idx in range(1, PERSONA_COUNT + 1):
        user_p = (
            f"USER:\nGenerate a single Hinglish conversation on a random topic (e.g., love, taxes, college days, space travel, movie discussion, cheating, etc.) "
            f"where each message is 40–50 words long. Return only a JSON array of message objects for example #{idx}."
        )
        src = f"Conversation #{idx}"
        print(f"→ Requesting {src}", file=sys.stderr)
        conv = safe_generate(system_p, user_p, src, key_cycle)
        if conv:
            all_conversations.append(conv)
        else:
            print(f"⚠️ Skipping {src}", file=sys.stderr)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_conversations, f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote {len(all_conversations)} conversations to {OUTPUT_FILE}", file=sys.stderr)


if __name__ == "__main__":
    main()