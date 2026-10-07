import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
import requests
from pydantic import BaseModel, Field
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SARVAM_API_KEY, SARVAM_API_URL, SARVAM_MODEL, CACHE_DIR
from src.loader import Tutorial, load_all_target_courses


class ExtractionResult(BaseModel):
    tutorial_id: int
    concepts_taught: List[str] = Field(
        description="List of 3-5 specific, concise concepts taught in this lesson."
    )
    prerequisites: List[str] = Field(
        default_factory=list,
        description="List of prerequisite concepts required before taking this lesson."
    )
    summary: str = Field(
        description="1-2 sentence summary of what the learner will be able to do."
    )


def clean_json_response(raw_text: str) -> str:
    """Removes markdown code fences like ```json ... ``` if present."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_for_tutorial(tutorial: Tutorial) -> ExtractionResult:
    """Extracts concepts and prerequisites for a tutorial, with disk caching."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{tutorial.id}.json")

    # 1. Check local cache first
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
            return ExtractionResult.model_validate(cached_data)

    # 2. Build prompt for Sarvam glm5.3
    system_prompt = (
        "You are an expert computer science curriculum architect. "
        "Analyze this technical video tutorial and extract:\n"
        "1. 'concepts_taught': 3-5 concise, specific skills or concepts taught.\n"
        "2. 'prerequisites': 1-4 prerequisite concepts needed prior to this tutorial. "
        "If it is a beginner lesson with no prerequisites, return an empty list.\n"
        "3. 'summary': A concise 1-2 sentence description of learning outcomes.\n\n"
        "CRITICAL: Return ONLY a valid JSON object matching this schema:\n"
        "{\n"
        '  "concepts_taught": ["concept 1", "concept 2"],\n'
        '  "prerequisites": ["prereq 1"],\n'
        '  "summary": "..."\n'
        "}\n"
        "Do not include markdown fences or any conversational filler."
    )

    user_prompt = (
        f"Course: {tutorial.course_name}\n"
        f"Lesson Title: {tutorial.title} (Lesson #{tutorial.order_index})\n"
        f"Difficulty Level: {tutorial.level}\n"
        f"Keywords: {tutorial.keywords}\n"
        f"Outline: {tutorial.outline}\n"
        f"Narration / Prerequisite snippet: {tutorial.script_snippet}"
    )

    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }

    payload = {
        "model": SARVAM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
    }

    try:
        response = requests.post(SARVAM_API_URL, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        raw_content = response.json()["choices"][0]["message"]["content"]
        cleaned_json = clean_json_response(raw_content)
        parsed = json.loads(cleaned_json)

        result = ExtractionResult(
            tutorial_id=tutorial.id,
            concepts_taught=parsed.get("concepts_taught", []),
            prerequisites=parsed.get("prerequisites", []),
            summary=parsed.get("summary", ""),
        )

        # 3. Save to disk cache
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))

        return result

    except Exception as e:
        print(f"\n[Warning] Extraction failed for Tutorial {tutorial.id} ({tutorial.title}): {e}")
        # Fallback to basic keywords if API fails
        fallback_concepts = [k.strip() for k in tutorial.keywords.split(",") if k.strip()][:4]
        return ExtractionResult(
            tutorial_id=tutorial.id,
            concepts_taught=fallback_concepts or [tutorial.title],
            prerequisites=[],
            summary=f"Covers {tutorial.title} in {tutorial.course_name}.",
        )


def run_extraction() -> Dict[int, ExtractionResult]:
    """Runs extraction for all target courses with a progress bar."""
    courses = load_all_target_courses()
    all_tutorials: List[Tutorial] = []
    for c in courses:
        all_tutorials.extend(c.tutorials)

    print(f"Starting concept extraction for {len(all_tutorials)} tutorials across 4 courses...")
    results: Dict[int, ExtractionResult] = {}

    for tut in tqdm(all_tutorials, desc="Extracting Concepts"):
        res = extract_for_tutorial(tut)
        results[tut.id] = res

    print(f"\nSuccessfully processed {len(results)} tutorials! Cached in '{CACHE_DIR}'.")
    return results


if __name__ == "__main__":
    run_extraction()
