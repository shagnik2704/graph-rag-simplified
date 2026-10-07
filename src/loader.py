import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from src.config import COURSE_DATA_DIR, TARGET_COURSES


class Tutorial(BaseModel):
    id: int
    title: str
    course_name: str
    order_index: int
    level: str = "Basic"
    url: str = ""
    keywords: str = ""
    outline: str = ""
    script_snippet: str = Field(
        default="",
        description="First ~800 characters of narration where prerequisites and objectives are stated."
    )

class Course(BaseModel):
    id: int
    name: str
    categories: List[str] = Field(default_factory=list)
    tutorials: List[Tutorial] = Field(default_factory=list)

def clean_text(text: Optional[str]) -> str:
    """Normalize whitespace and newlines."""
    if not text:
        return ""
    return " ".join(text.split()).strip()


def load_course(filename:str):
    """Loads a single course JSON and normalizes its tutorials using Pydantic."""
    filepath=os.path.join(COURSE_DATA_DIR,filename)
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None

    with open(filepath,"r",encoding='utf-8') as f:
        data=json.load(f)


    course_id = data.get("course_id", 0)
    course_name = data.get("course", "Unknown")
    categories = data.get("categories", [])
    raw_tutorials = data.get("tutorials", [])
    tutorials: List[Tutorial] = []

    for index, tut in enumerate(raw_tutorials, start=1):
        # Filter for English tutorials in this pilot
        lang = tut.get("language", "English")
        if lang != "English":
            continue
        raw_script = tut.get("tutorial_script", "")
        snippet = clean_text(raw_script)[:800]
        # Tutorial ID fallback logic
        raw_id = tut.get("tutorial_detail_id") or tut.get("tutorial_resource_id") or index
        tutorials.append(
            Tutorial(
                id=int(raw_id),
                title=clean_text(tut.get("title", f"Tutorial {index}")),
                course_name=course_name,
                order_index=index,
                level=clean_text(tut.get("level", "Basic")),
                url=tut.get("url", ""),
                keywords=clean_text(tut.get("keywords", "")),
                outline=clean_text(tut.get("outline", "")),
                script_snippet=snippet,
            )
        )
    return Course(
        id=int(course_id),
        name=course_name,
        categories=categories,
        tutorials=tutorials,
    )

def load_all_target_courses() -> List[Course]:
    """Loads all courses configured in TARGET_COURSES."""
    courses = []
    for filename in TARGET_COURSES:
        course = load_course(filename)
        if course:
            courses.append(course)
    return courses

if __name__ == "__main__":
    courses = load_all_target_courses()
    print(f"Loaded {len(courses)} courses successfully via Pydantic:\n")
    for c in courses:
        print(f"  • {c.name} ({len(c.tutorials)} tutorials)")
        if c.tutorials:
            print(f"      First: {c.tutorials[0].title} [{c.tutorials[0].level}]")
            print(f"      Last:  {c.tutorials[-1].title} [{c.tutorials[-1].level}]")
    



