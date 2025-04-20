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
PERSONA_COUNT = 150      # total requests
RETRIES       = 3       # retry on parse failure only


def construct_system_prompt() -> str:
    return (
        "SYSTEM:\n"
        "You are an expert generator of natural Indian conversational data. Produce each example as a JSON array of six message objects (alternating user and assistant), e.g.:\n"
        "[\n"
        "  { \"role\": \"user\",      \"content\": \"…\" },\n"
        "  { \"role\": \"assistant\", \"content\": \"…\" },\n"
        "  { \"role\": \"user\",      \"content\": \"…\" },\n"
        "  { \"role\": \"assistant\", \"content\": \"…\" },\n"
        "  { \"role\": \"user\",      \"content\": \"…\" },\n"
        "  { \"role\": \"assistant\", \"content\": \"…\" }\n"
        "]\n\n"
        "Requirements:\n"
        "- Language: Authentic Hinglish or Indian English, reflecting local idioms, Bollywood references, cricket banter, chai-time chat, auto-rickshaw rides, metro commutes, street-food stalls, festival vibes like Diwali, Holi, Eid and Ganesh Chaturthi.\n"
        "- Tone: Casual, semi-casual, serious, or formal—suitable for Indian workplaces, families, friend groups, or online chats.\n"
        "- Message length: Each content must be approx. 40–50 words, with natural Indian context (monsoon rains, temple bells, roadside chai stall aroma, train platform announcements).\n"
        "- Conversation starters: Use diverse and authentic Indian conversation openers instead of repetitive phrases like 'arre yaar' or 'haa yaar'. Examples include: 'Hey, Aur Batao, Arey bhai', 'Suniye', 'Kya haal hai', 'Namaste ji', 'Hello bhaiya', 'Accha suno', 'Bataiye', 'Oye', 'Dekho na', 'Kya chal raha hai', 'Sunn', 'Bhai sahab', 'Didi', 'Bhabhi ji', 'Bolo', 'Batao', or regional greetings like 'Kem cho', 'Vanakkam', 'Sat sri akal', etc.\n"
        "- IMPORTANT: DO NOT include any name prefixes like 'Sana:', 'Rohan:', etc. at the beginning of messages.\n"
        "- IMPORTANT: DO NOT include any character names (like 'Rohan', 'Sunita', etc.) within the conversation text.\n"
        "- IMPORTANT: DO NOT include any emojis in the text.\n"
        "- Output must be valid JSON only, no extra keys or markdown.\n\n"
        "USER:\n"
        "Generate 100 diverse conversations set in various Indian scenarios, including but not limited to:\n"
        "- Casual chai pe charcha (gossip at a roadside tapri)\n"
        "- College hostel mess and exam stress\n"
        "- Office chai-break debates and cubicle gossip\n"
        "- Rickshaw vs. metro commute dilemmas\n"
        "- Cricket match strategies and IPL banter\n"
        "- Festival preparations: Diwali diyas, Holi rang, Eid sweets, Ganesh visarjan\n"
        "- Family events: shaadi invitations, bachchon ki PTM, ancestral home visits\n"
        "- Street-food hunts: golgappe, vada pav, samosa, chai-samosa combo\n"
        "- Bollywood movie reviews and song lip-sync battles\n"
        "- Startup hustle: pitch meetings, investor chai sessions, code-debugging at midnight\n"
        "- Personal finance talks: SIP, mutual funds, saving for shaadi or overseas trip\n"
        "- Health check-ins: yoga on the terrace, ayurvedic remedies, gym routines\n"
        "- Spiritual discussions at temple or mandir prasad lines\n"
        "- Technical support: mobile recharge issues, OTT subscription queries\n"
        "- Weather talk: Mumbai monsoon floods, Delhi winters ka fog, Chennai heatwaves\n"
        "- Miscellaneous: apology texts, customer service calls, invitation RSVPs, relationship, sex\n\n"
        "Ensure each conversation is unique, roles alternate correctly, and each message captures vibrant Indian life."
    )


def call_gemini(system_p: str, user_p: str, api_key: str) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    cfg = genai.types.GenerationConfig(
        temperature=0.7,  # Increased temperature for more diverse outputs
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
        data = json.loads(cleaned)
        # Clean any names or emojis that might still appear in the content
        return clean_conversation_data(data)
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse failed for {src}: {e}", file=sys.stderr)
        return []


def clean_conversation_data(data: list) -> list:
    """Clean conversation data by removing name prefixes, character names, and emojis."""
    if not data or not isinstance(data, list):
        return data
    
    # Regex patterns for cleaning
    name_prefix_pattern = r'^\s*([A-Z][a-z]+\s*:)\s*'  # Matches name prefixes like "Sana:" at start
    emoji_pattern = r'[\U0001F000-\U0001F9FF]|[\u2600-\u27BF]'  # Common emoji unicode ranges
    
    for item in data:
        if isinstance(item, dict) and 'content' in item:
            # Remove name prefixes at the beginning
            item['content'] = re.sub(name_prefix_pattern, '', item['content'])
            
            # Remove emojis
            item['content'] = re.sub(emoji_pattern, '', item['content'])
            
            # Remove common Indian names that might be used as character references
            # This is a simple approach and might need refinement based on results
            common_names = ['Rohan', 'Sunita', 'Priya', 'Arjun', 'Sana', 'Siddharth', 'Rahul', 'Neha', 'Amit']
            for name in common_names:
                # Replace name followed by a comma, space, or possessive
                item['content'] = re.sub(r'\b' + name + r'\b[,!\s]', ' ', item['content'])
                item['content'] = re.sub(r'\b' + name + r'\'s\b', '', item['content'])
            
            # Clean up any double spaces created by replacements
            item['content'] = re.sub(r'\s+', ' ', item['content']).strip()
    
    return data


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
                    print(f"⚠️ All API keys exhausted for {src}. Sleeping for 1-2 minutes before retrying.", file=sys.stderr)
                    rest_time = random.uniform(1*60, 1.5*60)
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


def get_random_scenario() -> str:
    # Load scenarios from files if available
    scenario_files = [
        PROMPTS_DIR / "1.txt",
        Path(__file__).parent.parent / "prompt2" / "hinglish_scenarios_part1.txt",
        Path(__file__).parent.parent / "prompt2" / "personas_scenarios.txt"
    ]
    
    scenarios = []
    for file_path in scenario_files:
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    scenarios.extend([line.strip() for line in f if line.strip()])
            except Exception as e:
                print(f"⚠️ Error reading {file_path}: {e}", file=sys.stderr)
    
    # If no scenarios found in files, use these default scenarios
    if not scenarios:
        scenarios = [
            "Casual chai pe charcha (gossip at a roadside tapri)",
            "College hostel mess and exam stress",
            "Office chai-break debates and cubicle gossip",
            "Rickshaw vs. metro commute dilemmas",
            "Cricket match strategies and IPL banter",
            "Festival preparations: Diwali diyas, Holi rang, Eid sweets",
            "Family events: shaadi invitations, bachchon ki PTM",
            "Street-food hunts: golgappe, vada pav, samosa",
            "Bollywood movie reviews and song lip-sync battles",
            "Startup hustle: pitch meetings, investor chai sessions",
            "Personal finance talks: SIP, mutual funds, saving for shaadi",
            "Health check-ins: yoga on the terrace, ayurvedic remedies",
            "Spiritual discussions at temple or mandir prasad lines",
            "Technical support: mobile recharge issues, OTT subscription",
            "Weather talk: Mumbai monsoon floods, Delhi winters ka fog"
        ]
    
    return random.choice(scenarios)

def main():
    system_p = construct_system_prompt()
    key_cycle = itertools.cycle(GEMINI_API_KEYS)

    all_conversations = []
    for idx in range(1, PERSONA_COUNT + 1):
        # Get a random scenario for each conversation
        scenario = get_random_scenario()
        
        user_p = (
            f"USER:\nGenerate a single Hinglish conversation about '{scenario}' "
            f"where each message is 40–50 words long. Return only a JSON array of message objects for example #{idx}."
        )
        src = f"Conversation #{idx} - {scenario[:30]}..."
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