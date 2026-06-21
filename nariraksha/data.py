"""
nariraksha/data.py
------------------
Dataset generation and formatting for NariRaksha.
No GPU/CUDA dependencies — safe to import anywhere.
"""
import os
import json
import random
import time
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
RISK_CATEGORIES = [
    "cyberstalking",
    "online harassment",
    "workplace harassment",
    "domestic violence",
    "coercive control",
    "blackmail / extortion",
    "trafficking indicators",
    "deepfake abuse",
    "revenge content (NCII)",
    "financial exploitation",
    "emotional manipulation",
    "public safety / stalking",
]

SEVERITY_LEVELS = ["low", "medium", "high", "critical"]

CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata",
    "Pune", "Jaipur", "Lucknow", "Patna", "Bhopal",
    "Hyderabad", "Ahmedabad", "Surat", "Visakhapatnam", "Coimbatore",
]

LEGAL_REFS = {
    "cyberstalking":           "BNS Section 78 (Stalking), IT Act Section 66E (privacy violation)",
    "online harassment":       "BNS Section 351 (criminal intimidation), IT Act Section 67 (obscene content)",
    "workplace harassment":    "POSH Act 2013 Sections 2(n) and 11, BNS Section 74",
    "domestic violence":       "Protection of Women from Domestic Violence Act 2005 Sections 3 and 18",
    "coercive control":        "BNS Section 85 (cruelty by husband/relatives), PWDVA Section 3",
    "blackmail / extortion":   "BNS Section 308(4) (extortion), IT Act Section 66C",
    "trafficking indicators":  "ITPA Section 5, BNS Sections 143-144 (trafficking in persons)",
    "deepfake abuse":          "IT Act Section 67A, BNS Section 73 (outraging modesty), DPDPA 2023",
    "revenge content (NCII)":  "IT Act Section 67A, BNS Section 73, NCPCR guidelines",
    "financial exploitation":  "BNS Section 316 (cheating), IPC Section 498A as applicable",
    "emotional manipulation":  "PWDVA Section 3(c) (emotional abuse), BNS Section 85",
    "public safety / stalking": "BNS Section 78 (stalking), BNS Section 351 (criminal intimidation)",
}


# ---------------------------------------------------------------------------
# Mock generator (no API key needed)
# ---------------------------------------------------------------------------
def generate_mock_example(risk: str, severity: str) -> Dict[str, Any]:
    city = random.choice(CITIES)
    age = random.randint(18, 52)
    legal = LEGAL_REFS.get(risk, "BNS Section 78, IT Act Section 66E")

    scenario_templates = [
        f"A {age}-year-old woman from {city} reported an incident of {risk}. "
        f"The perpetrator had been targeting her for several weeks, causing {severity}-level distress. "
        f"She sought help after the situation escalated and began affecting her professional life.",

        f"A {age}-year-old professional in {city} experienced {risk}. "
        f"The situation was assessed as {severity} severity. She had documented evidence "
        f"and approached the local cyber cell for assistance.",

        f"In {city}, a {age}-year-old woman filed a complaint regarding {risk}. "
        f"The harm caused was classified as {severity}. "
        f"She was advised on legal remedies and support organizations available in her area.",
    ]

    reasoning_templates = [
        f"This case constitutes {risk} based on the sustained, unwanted conduct directed at the victim. "
        f"The {severity} severity is consistent with the documented impact on her daily functioning and emotional well-being. "
        f"The pattern of behaviour meets the legal threshold under Indian law.",

        f"Analysis confirms {risk} due to the nature, frequency, and intent of the perpetrator's actions. "
        f"The {severity} classification reflects both the psychological harm and the potential for escalation. "
        f"Immediate legal and support intervention is warranted.",
    ]

    actions = [
        "Preserve all digital evidence (screenshots, call logs). Report immediately to the cyber cell or dial 1930 (National Cyber Helpline). Seek support from iCall or iDare.",
        "File an FIR at the nearest police station or via the National Cybercrime Reporting Portal (cybercrime.gov.in). Contact the State Women's Commission for additional support.",
        "Contact Shakti Shalini (011-24373736) or the National Commission for Women (7217735372). Ensure personal safety and avoid engaging with the perpetrator.",
    ]

    return {
        "scenario": random.choice(scenario_templates),
        "language": "English",
        "risk_type": risk,
        "severity": severity,
        "reasoning": random.choice(reasoning_templates),
        "recommended_action": random.choice(actions),
        "legal_context": legal,
        "confidence": round(random.uniform(0.87, 0.99), 2),
    }


# ---------------------------------------------------------------------------
# OpenAI-based generator (optional, requires api_key)
# ---------------------------------------------------------------------------
def generate_openai_example(
    risk: str,
    severity: str,
    client,
    retries: int = 3,
) -> Optional[Dict[str, Any]]:
    legal = LEGAL_REFS.get(risk, "BNS Section 78")
    prompt = (
        f"Generate a realistic, detailed women's safety scenario from India.\n"
        f"Risk type: {risk}\nSeverity: {severity}\n"
        f"Relevant law hint: {legal}\n\n"
        "Return ONLY a JSON object with these keys:\n"
        '  scenario, language, risk_type, severity, reasoning, recommended_action, legal_context, confidence\n'
        "Make it specific — real city, realistic age, concrete legal sections (e.g. BNS 78, IT Act 67A).\n"
        "confidence should be a float between 0.85 and 0.99."
    )
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a legal and safety expert specialising in Indian women's safety law."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=min(0.8 + attempt * 0.1, 1.0),
            )
            data = json.loads(resp.choices[0].message.content)
            # Basic validation
            required = {"scenario", "risk_type", "severity", "reasoning", "recommended_action", "legal_context"}
            if not required.issubset(data.keys()):
                raise ValueError(f"Missing keys: {required - data.keys()}")
            if "Section XX" in data.get("legal_context", "") or "Placeholder" in str(data):
                raise ValueError("Placeholder content detected")
            data.setdefault("language", "English")
            data.setdefault("confidence", 0.90)
            return data
        except Exception as exc:
            print(f"  [OpenAI attempt {attempt+1}/{retries} failed] {exc}")
            time.sleep(1 + attempt)
    return None


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------
def generate_dataset(
    target: int,
    output_path: str,
    use_mock: bool = True,
    openai_api_key: str = None,
) -> int:
    """
    Generate `target` examples and append to `output_path` (JSONL).
    Returns number of examples actually written.
    """
    client = None
    if not use_mock and openai_api_key:
        import openai as _openai
        client = _openai.OpenAI(api_key=openai_api_key)
        print(f"  Using OpenAI API to generate {target} examples.")
    else:
        use_mock = True
        print(f"  Using mock generator for {target} examples (no API key or --use_mock).")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    written = 0
    with open(output_path, "a", encoding="utf-8") as f:
        for i in range(target):
            risk = random.choice(RISK_CATEGORIES)
            severity = random.choice(SEVERITY_LEVELS)

            if use_mock or client is None:
                example = generate_mock_example(risk, severity)
            else:
                example = generate_openai_example(risk, severity, client)
                if example is None:
                    example = generate_mock_example(risk, severity)

            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            written += 1

            if (i + 1) % 100 == 0:
                print(f"  Generated {i+1}/{target} examples...")

    print(f"  ✅ {written} examples written to {output_path}")
    return written


# ---------------------------------------------------------------------------
# Prompt formatting for training
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are NariRaksha, an AI safety assistant specialising in Indian women's safety law. "
    "Analyse the given scenario and provide a structured safety assessment."
)

def format_for_training(example: Dict[str, Any]) -> str:
    """
    Formats one dataset example into a chat-style instruction string
    compatible with Unsloth's FastLanguageModel.
    """
    user_msg = (
        f"Analyse this safety scenario and provide your assessment:\n\n"
        f"{example['scenario']}"
    )
    assistant_msg = json.dumps({
        "risk_type": example["risk_type"],
        "severity": example["severity"],
        "reasoning": example["reasoning"],
        "recommended_action": example["recommended_action"],
        "legal_context": example["legal_context"],
        "confidence": example.get("confidence", 0.90),
    }, ensure_ascii=False, indent=2)

    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
    )


def load_and_format(jsonl_path: str) -> List[Dict[str, str]]:
    """Load a JSONL file and return list of {text: ...} dicts for SFTTrainer."""
    examples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                examples.append({"text": format_for_training(row)})
            except Exception:
                pass
    return examples
