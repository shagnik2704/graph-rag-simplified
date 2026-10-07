import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
from neo4j import GraphDatabase
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, CACHE_DIR
from src.loader import load_all_target_courses, Course, Tutorial


class KnowledgeGraph:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def create_constraints(self):
        """Creates unique constraints to prevent duplicate nodes."""
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Course) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Tutorial) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (cp:Concept) REQUIRE cp.name IS UNIQUE",
        ]
        with self.driver.session() as session:
            for q in queries:
                session.run(q)
        print("✓ Database constraints verified.")

    def clear_database(self):
        """Removes existing nodes to allow a clean rebuild."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("✓ Existing graph data cleared for fresh ingestion.")

    def ingest_courses_and_tutorials(self, courses: List[Course]):
        """Ingests Course nodes, Tutorial nodes, and NEXT_IN_SERIES edges."""
        with self.driver.session() as session:
            for course in courses:
                # 1. Create Course node
                session.run(
                    """
                    MERGE (c:Course {id: $id})
                    SET c.name = $name, c.categories = $categories
                    """,
                    id=course.id,
                    name=course.name,
                    categories=course.categories,
                )

                # 2. Create Tutorial nodes and CONTAINS edges
                prev_tutorial: Tutorial = None
                for tut in course.tutorials:
                    # Fetch cached summary if available
                    summary = ""
                    cache_file = os.path.join(CACHE_DIR, f"{tut.id}.json")
                    if os.path.exists(cache_file):
                        with open(cache_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            summary = data.get("summary", "")

                    session.run(
                        """
                        MERGE (t:Tutorial {id: $id})
                        SET t.title = $title,
                            t.course_name = $course_name,
                            t.order_index = $order_index,
                            t.level = $level,
                            t.url = $url,
                            t.summary = $summary

                        WITH t
                        MATCH (c:Course {id: $course_id})
                        MERGE (c)-[:CONTAINS]->(t)
                        """,
                        id=tut.id,
                        title=tut.title,
                        course_name=tut.course_name,
                        order_index=tut.order_index,
                        level=tut.level,
                        url=tut.url,
                        summary=summary,
                        course_id=course.id,
                    )

                    # 3. Create NEXT_IN_SERIES edge between consecutive lessons
                    if prev_tutorial:
                        session.run(
                            """
                            MATCH (t1:Tutorial {id: $prev_id})
                            MATCH (t2:Tutorial {id: $curr_id})
                            MERGE (t1)-[:NEXT_IN_SERIES]->(t2)
                            """,
                            prev_id=prev_tutorial.id,
                            curr_id=tut.id,
                        )
                    prev_tutorial = tut

        print(f"✓ Ingested {len(courses)} courses and their sequential tutorial backbones.")

    def ingest_concepts_and_prerequisites(self):
        """Ingests extracted Concepts, TEACHES, and REQUIRES edges."""
        files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".json")]
        print(f"Ingesting concepts from {len(files)} extracted tutorial files...")

        with self.driver.session() as session:
            for fname in tqdm(files, desc="Building Graph Edges"):
                fpath = os.path.join(CACHE_DIR, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                tut_id = data.get("tutorial_id")
                concepts_taught = data.get("concepts_taught", [])
                prerequisites = data.get("prerequisites", [])

                # Ingest TEACHES relationships
                for concept in concepts_taught:
                    clean_concept = concept.strip()
                    if clean_concept:
                        session.run(
                            """
                            MATCH (t:Tutorial {id: $tut_id})
                            MERGE (cp:Concept {name: $concept_name})
                            MERGE (t)-[:TEACHES]->(cp)
                            """,
                            tut_id=tut_id,
                            concept_name=clean_concept,
                        )

                # Ingest REQUIRES relationships
                for prereq in prerequisites:
                    clean_prereq = prereq.strip()
                    if clean_prereq and "Basic Computer Literacy" not in clean_prereq:
                        session.run(
                            """
                            MATCH (t:Tutorial {id: $tut_id})
                            MERGE (cp:Concept {name: $prereq_name})
                            MERGE (t)-[:REQUIRES]->(cp)
                            """,
                            tut_id=tut_id,
                            prereq_name=clean_prereq,
                        )

        print("✓ All Concept nodes, TEACHES, and REQUIRES edges ingested successfully.")

    def print_graph_stats(self):
        """Prints high-level stats about the ingested graph."""
        with self.driver.session() as session:
            node_counts = session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY count DESC"
            )
            print("\n" + "=" * 40)
            print("       KNOWLEDGE GRAPH SUMMARY       ")
            print("=" * 40)
            print("NODE TYPES:")
            for record in node_counts:
                print(f"  • {record['label']:<15}: {record['count']}")

            rel_counts = session.run(
                "MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS count ORDER BY count DESC"
            )
            print("\nRELATIONSHIP TYPES:")
            for record in rel_counts:
                print(f"  • {record['rel']:<15}: {record['count']}")
            print("=" * 40 + "\n")


def main():
    kg = KnowledgeGraph()
    try:
        courses = load_all_target_courses()
        kg.create_constraints()
        kg.clear_database()
        kg.ingest_courses_and_tutorials(courses)
        kg.ingest_concepts_and_prerequisites()
        kg.print_graph_stats()
    finally:
        kg.close()


if __name__ == "__main__":
    main()
