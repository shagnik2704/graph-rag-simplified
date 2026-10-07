import os
import sys

# Prevent OpenMP / ONNX Runtime multi-threaded mutex lock crash on macOS Apple Silicon
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
from typing import List
import numpy as np
from fastembed import TextEmbedding
from neo4j import GraphDatabase
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD


class VectorIndexer:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
        self.embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", threads=1)

    def close(self):
        self.driver.close()

    def create_vector_index(self):
        """Creates a native Neo4j vector index on (:Tutorial {embedding})."""
        print("• Creating Neo4j native vector index 'tutorial_index'...")
        query = """
        CREATE VECTOR INDEX tutorial_index IF NOT EXISTS
        FOR (t:Tutorial)
        ON (t.embedding)
        OPTIONS {indexConfig: {
          `vector.dimensions`: 384,
          `vector.similarity_function`: 'cosine'
        }}
        """
        with self.driver.session() as session:
            session.run(query)
        print("✓ Vector index 'tutorial_index' defined.")

    def embed_and_store_tutorials(self):
        """Generates embeddings for all tutorials and saves them onto the nodes in Neo4j."""
        print("• Fetching tutorials from Neo4j Aura...")
        with self.driver.session() as session:
            tutorials = session.run(
                """
                MATCH (t:Tutorial)
                RETURN t.id AS id, t.title AS title, t.course_name AS course, t.summary AS summary
                """
            ).data()

        print(f"• Generating embeddings for {len(tutorials)} tutorials...")
        texts = [f"{t['course']} - {t['title']}: {t['summary']}" for t in tutorials]
        embeddings = list(self.embed_model.embed(texts))

        print("• Storing 384-dimensional vector embeddings directly on (:Tutorial) nodes...")
        with self.driver.session() as session:
            for tut, emb in tqdm(zip(tutorials, embeddings), total=len(tutorials), desc="Writing Vectors to Neo4j"):
                session.run(
                    """
                    MATCH (t:Tutorial {id: $id})
                    SET t.embedding = $embedding
                    """,
                    id=tut["id"],
                    embedding=[float(val) for val in emb],
                )

        print("✓ All 67 tutorials now hold native vector embeddings in Neo4j Aura!")

    def verify_vector_search(self):
        """Tests the native Neo4j vector index with a sample query."""
        test_query = "Build web forms and input fields"
        print(f"\n• Testing native Neo4j vector search for: '{test_query}'...")

        query_vec = [float(val) for val in list(self.embed_model.embed([test_query]))[0]]

        cypher = """
        CALL db.index.vector.queryNodes('tutorial_index', 3, $query_vec)
        YIELD node, score
        RETURN node.course_name AS course, node.title AS title, node.level AS level, score
        """
        with self.driver.session() as session:
            results = session.run(cypher, query_vec=query_vec).data()

        print("\nTop 3 matches returned directly by Neo4j's C++ vector engine:")
        for r in results:
            print(f"  • [{r['course']}] {r['title']} (Score: {r['score']:.4f})")
        print()


def main():
    indexer = VectorIndexer()
    try:
        indexer.create_vector_index()
        indexer.embed_and_store_tutorials()
        indexer.verify_vector_search()
    finally:
        indexer.close()


if __name__ == "__main__":
    main()
