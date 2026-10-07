import os
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
from fastembed import TextEmbedding
from neo4j import GraphDatabase
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD


class SemanticLinker:
    def __init__(self, threshold: float = 0.78):
        self.threshold = threshold
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
        self.embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def close(self):
        self.driver.close()

    def build_semantic_bridges(self):
        """Discovers semantic matches between REQUIRES and TEACHES concepts and creates graph edges."""
        print("\n" + "=" * 50)
        print("  STEP 4.5: SEMANTIC CONCEPT LINKING (EMBEDDINGS)  ")
        print("=" * 50)

        with self.driver.session() as session:
            # 1. Fetch concepts with their corresponding tutorials
            teaches_data = session.run(
                """
                MATCH (t:Tutorial)-[:TEACHES]->(c:Concept)
                RETURN t.id AS tut_id, t.title AS tut_title, t.course_name AS course, c.name AS concept
                """
            ).data()

            requires_data = session.run(
                """
                MATCH (t:Tutorial)-[:REQUIRES]->(c:Concept)
                RETURN t.id AS tut_id, t.title AS tut_title, t.course_name AS course, c.name AS concept
                """
            ).data()

            print(f"• Fetched {len(teaches_data)} TEACHES concept instances.")
            print(f"• Fetched {len(requires_data)} REQUIRES concept instances.")

            unique_teaches = list({d["concept"] for d in teaches_data})
            unique_requires = list({d["concept"] for d in requires_data})

            print(f"• Computing embeddings for {len(unique_teaches)} unique skills & {len(unique_requires)} prerequisites...")
            teaches_vecs = np.array(list(self.embed_model.embed(unique_teaches)))
            requires_vecs = np.array(list(self.embed_model.embed(unique_requires)))

            # 2. Cosine Similarity Matrix
            sim_matrix = np.dot(requires_vecs, teaches_vecs.T)

            # 3. Map high-similarity matches
            concept_bridge_map = {}
            for i, req_name in enumerate(unique_requires):
                for j, teach_name in enumerate(unique_teaches):
                    score = float(sim_matrix[i][j])
                    if score >= self.threshold:
                        if req_name not in concept_bridge_map or score > concept_bridge_map[req_name][1]:
                            concept_bridge_map[req_name] = (teach_name, score)

            print(f"• Identified {len(concept_bridge_map)} high-confidence concept bridges (similarity >= {self.threshold}).")

            # 4. Ingest RESOLVES_TO edges between Concept nodes
            print("• Ingesting [:RESOLVES_TO] edges in Neo4j Aura...")
            for req_name, (teach_name, score) in tqdm(concept_bridge_map.items(), desc="Linking Concepts"):
                session.run(
                    """
                    MATCH (c_req:Concept {name: $req_name})
                    MATCH (c_teach:Concept {name: $teach_name})
                    MERGE (c_req)-[r:RESOLVES_TO]->(c_teach)
                    SET r.score = $score
                    """,
                    req_name=req_name,
                    teach_name=teach_name,
                    score=score,
                )

            # 5. Ingest direct DEPENDS_ON edges between Tutorials
            print("• Ingesting direct [:DEPENDS_ON] prerequisite edges between Tutorials...")
            links_created = 0
            for r_item in requires_data:
                req_concept = r_item["concept"]
                if req_concept in concept_bridge_map:
                    teach_concept, score = concept_bridge_map[req_concept]
                    # Find which tutorials teach this concept
                    matching_teachers = [t for t in teaches_data if t["concept"] == teach_concept]
                    for m in matching_teachers:
                        # Don't let a tutorial depend on itself
                        if m["tut_id"] != r_item["tut_id"]:
                            session.run(
                                """
                                MATCH (t_consumer:Tutorial {id: $consumer_id})
                                MATCH (t_provider:Tutorial {id: $provider_id})
                                MERGE (t_consumer)-[r:DEPENDS_ON]->(t_provider)
                                SET r.reason = $reason, r.score = $score
                                """,
                                consumer_id=r_item["tut_id"],
                                provider_id=m["tut_id"],
                                reason=teach_concept,
                                score=score,
                            )
                            links_created += 1

            print(f"✓ Successfully created {len(concept_bridge_map)} [:RESOLVES_TO] edges.")
            print(f"✓ Successfully created {links_created} [:DEPENDS_ON] tutorial-to-tutorial prerequisite bridges!")

            # 6. Display a few prominent cross-course bridges
            print("\nSample Cross-Course Prerequisite Bridges:")
            sample_bridges = session.run(
                """
                MATCH (t1:Tutorial)-[r:DEPENDS_ON]->(t2:Tutorial)
                WHERE t1.course_name <> t2.course_name
                RETURN t1.course_name AS consumer_course, t1.title AS consumer_tut,
                       r.reason AS skill,
                       t2.course_name AS provider_course, t2.title AS provider_tut,
                       r.score AS score
                LIMIT 5
                """
            )
            for rec in sample_bridges:
                print(f"  • [{rec['consumer_course']}] '{rec['consumer_tut']}'")
                print(f"      ↳ DEPENDS ON: [{rec['provider_course']}] '{rec['provider_tut']}'")
                print(f"      ↳ Reason: {rec['skill']} (match: {rec['score']:.2f})\n")


def main():
    linker = SemanticLinker(threshold=0.78)
    try:
        linker.build_semantic_bridges()
    finally:
        linker.close()


if __name__ == "__main__":
    main()
