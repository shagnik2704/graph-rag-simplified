import os
from dotenv import load_dotenv
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

SARVAM_API_KEY="sk_4bbrw5ki_NsGYKWzmdnT7QCqSGbNYQ9En"
SARVAM_API_URL = "https://api.sarvam.ai/v2/chat/completions"
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "glm5.3")


COURSE_DATA_DIR = "course_data"
CACHE_DIR = "data/extracted"

TARGET_COURSES = [
    "103_html.json",
    "142_css.json",
    "147_bootstrap.json",
    "144_javascript.json"
]

