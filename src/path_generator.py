import os
import sys

# Prevent OpenMP / ONNX Runtime multi-threaded mutex lock crash on macOS Apple Silicon
os.environ["OMP_NUM_THREADS"] = "1"
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict, deque
import numpy as np
import requests
from fastembed import TextEmbedding
from neo4j import GraphDatabase

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    SARVAM_API_KEY,
    SARVAM_API_URL,
    SARVAM_MODEL,
)


class LearningPathGenerator:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
        self.embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", threads=1)
        self.course_priority = {"HTML": 1, "CSS": 2, "Bootstrap": 3, "JavaScript": 4}

    def close(self):
        self.driver.close()

    # =========================================================================
    # 1. VECTOR SEARCH: Find Target Destination Tutorials
    # =========================================================================
    def find_seed_tutorials(self, user_goal: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Finds destination tutorials directly using Neo4j's native C++ vector index."""
        goal_vec = [float(v) for v in list(self.embed_model.embed([user_goal]))[0]]

        query = """
        CALL db.index.vector.queryNodes('tutorial_index', $top_k, $goal_vec)
        YIELD node, score
        RETURN node.id AS id, node.title AS title, node.course_name AS course,
               node.summary AS summary, node.url AS url, node.level AS level,
               node.order_index AS order_index, score
        """
        with self.driver.session() as session:
            return session.run(query, top_k=top_k, goal_vec=goal_vec).data()

    # =========================================================================
    # 2. GRAPH TRAVERSAL & TOPOLOGICAL SORT (Kahn's Algorithm)
    # =========================================================================
    def traverse_and_topological_sort(
        self, seed_tutorials: List[Dict[str, Any]], known_topics: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Traverses backward in Neo4j to collect prerequisites,
        then runs Kahn's Algorithm to mathematically order them.
        """
        seed_ids = [t["id"] for t in seed_tutorials]

        with self.driver.session() as session:
            # Query backward along DEPENDS_ON to collect the full prerequisite DAG
            query = """
            MATCH path = (target:Tutorial)-[:DEPENDS_ON*0..4]->(prereq:Tutorial)
            WHERE target.id IN $seed_ids
            WITH collect(path) AS paths
            UNWIND paths AS p
            UNWIND nodes(p) AS n
            WITH DISTINCT n, paths
            UNWIND paths AS p
            UNWIND relationships(p) AS r
            RETURN collect(DISTINCT {
                id: n.id, 
                title: n.title, 
                course: n.course_name, 
                level: n.level, 
                order: n.order_index, 
                url: n.url, 
                summary: n.summary
            }) AS tutorials,
            collect(DISTINCT {
                consumer: startNode(r).id, 
                provider: endNode(r).id
            }) AS dependencies
            """
            result = session.run(query, seed_ids=seed_ids).single()

        tuts = {t["id"]: t for t in result["tutorials"]}
        dependencies = result["dependencies"]

        # If user already knows a subject (e.g. "I know basic HTML"), filter it out
        if "html" in known_topics.lower() and "basic" in known_topics.lower():
            tuts = {
                tid: t for tid, t in tuts.items()
                if not (t["course"] == "HTML" and t["level"] == "Basic")
            }

        # Build DAG: provider (prerequisite) --> consumer (dependent)
        adj = defaultdict(list)
        in_degree = defaultdict(int)
        for t_id in tuts:
            in_degree[t_id] = 0

        for dep in dependencies:
            provider = dep["provider"]  # Must be learned first
            consumer = dep["consumer"]  # Depends on provider
            if provider in tuts and consumer in tuts and provider != consumer:
                adj[provider].append(consumer)
                in_degree[consumer] += 1

        # Kahn's Algorithm: Start with tutorials having 0 unsatisfied prerequisites
        zero_in_degree = [tid for tid in tuts if in_degree[tid] == 0]
        # Sort initial batch by pedagogical course priority
        zero_in_degree.sort(
            key=lambda tid: (
                self.course_priority.get(tuts[tid]["course"], 99),
                tuts[tid]["order"],
            )
        )
        queue = deque(zero_in_degree)
        sorted_syllabus = []

        while queue:
            curr_id = queue.popleft()
            sorted_syllabus.append(tuts[curr_id])

            # For each lesson that depended on curr_id, decrement its prerequisite count
            newly_ready = []
            for neighbor_id in adj[curr_id]:
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    newly_ready.append(neighbor_id)

            # Sort newly unlocked lessons before adding to queue
            newly_ready.sort(
                key=lambda tid: (
                    self.course_priority.get(tuts[tid]["course"], 99),
                    tuts[tid]["order"],
                )
            )
            queue.extend(newly_ready)

        return sorted_syllabus

    # =========================================================================
    # 3. LLM GENERATION: Synthesize Curriculum with Sarvam glm5.3
    # =========================================================================
    def synthesize_with_llm(
        self, user_goal: str, user_background: str, syllabus: List[Dict[str, Any]]
    ) -> str:
        """Prompts Sarvam's glm5.3 to generate an encouraging, structured syllabus."""
        syllabus_text = ""
        for i, item in enumerate(syllabus, 1):
            syllabus_text += (
                f"{i}. [{item['course']}] {item['title']} (Level: {item['level']})\n"
                f"   Summary: {item['summary']}\n"
                f"   Video URL: {item['url']}\n\n"
            )

        system_prompt = (
            "You are an inspiring computer science mentor for the Spoken Tutorial project. "
            "Format a motivating, structured learning path for the student based strictly on the retrieved lessons. "
            "Directly output the final response with: Timeline Table, Phased Lessons (with video links and practice tasks), and a Capstone Project."
        )

        user_prompt = (
            f"Student Learning Goal: {user_goal}\n"
            f"Current Background: {user_background or 'Absolute Beginner'}\n\n"
            f"Retrieved Prerequisite-Ordered Lessons:\n\n"
            f"{syllabus_text}\n"
            "Directly synthesize this into a personalized study plan."
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
            "max_tokens": 4096,
            "temperature": 0.2,
        }

        # 1. Use Sarvam v1 conversational endpoint for fast, direct, rich syllabus generation
        v1_url = "https://api.sarvam.ai/v1/chat/completions"
        payload_v1 = {
            "model": "sarvam-105b-conversations",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 3000,
            "temperature": 0.2,
        }

        try:
            res = requests.post(v1_url, headers=headers, json=payload_v1, timeout=60)
            if res.status_code == 200:
                data = res.json()
                content = data["choices"][0]["message"]["content"]
                if content:
                    return content
        except Exception:
            pass

        # 2. Fallback to configured model and endpoint
        response = requests.post(SARVAM_API_URL, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        choice = response.json()["choices"][0]
        msg = choice["message"]
        content = msg.get("content")
        if not content and msg.get("reasoning_content"):
            content = msg["reasoning_content"]
        return content or "No content generated."

    # =========================================================================
    # END-TO-END RUNNER
    # =========================================================================
    def generate(self, user_goal: str, user_background: str = "") -> str:
        print(f"\n[1/3] Finding destination tutorials for: '{user_goal}'...")
        seeds = self.find_seed_tutorials(user_goal, top_k=3)
        print("      Destination Lessons Identified:")
        for s in seeds:
            print(f"      • [{s['course']}] {s['title']}")

        print("\n[2/3] Traversing Graph & Running Topological Sort (Kahn's Algorithm)...")
        syllabus = self.traverse_and_topological_sort(seeds, known_topics=user_background)
        print(f"      Ordered learning path length: {len(syllabus)} tutorials.")

        print("\n[3/3] Synthesizing curriculum with Sarvam glm5.3...")
        roadmap = self.synthesize_with_llm(user_goal, user_background, syllabus)
        return roadmap


def main():
    generator = LearningPathGenerator()
    try:
        print("=" * 60)
        print("        GRAPHRAG LEARNING PATH GENERATOR")
        print("=" * 60)

        if len(sys.argv) > 1:
            goal = " ".join(sys.argv[1:])
            background = ""
        else:
            print("Enter your learning goal (e.g., 'I want to build interactive web forms')")
            goal = input("🎯 Learning Goal: ").strip()
            if not goal:
                goal = "I want to build an interactive web form with modern styling and responsive layout"
            background = input("🎓 Current Background (press Enter if Beginner): ").strip()

        roadmap = generator.generate(user_goal=goal, user_background=background)
        print("\n" + "=" * 60)
        print("                 GENERATED SYLLABUS")
        print("=" * 60 + "\n")
        print(roadmap)

    finally:
        generator.close()


if __name__ == "__main__":
    main()

