import os
import sys
import json
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Optional

os.environ["OMP_NUM_THREADS"] = "1"
from fastembed import TextEmbedding
from neo4j import GraphDatabase
import requests
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from src.config import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    SARVAM_API_KEY,
    SARVAM_API_URL,
    SARVAM_MODEL,
)

class AgentState(TypedDict):
    # Intake & Profiling
    raw_query: str                              # Raw user prompt
    extracted_goal: str                         # Disambiguated technical target
    learner_background: str                     # User's prior programming experience
    time_budget_hours: int                      # Available learning time in hours
    
    # Graph Retrieval
    seeds: List[Dict[str, Any]]                 # Destination tutorials from vector index
    candidate_tutorials: List[Dict[str, Any]]   # Target + prerequisite lessons
    dependency_edges: List[Dict[str, str]]      # Directed edges for topological sort
    prerequisite_concepts: List[Dict[str, Any]] # Key concepts taught in prerequisites
    
    # Diagnostic Pulse-Check
    quiz: List[Dict[str, Any]]                  # 2 targeted multiple-choice questions
    user_answers: Optional[str]                 # User's quiz responses
    mastered_ids: List[str]                     # Tutorial IDs to be pruned from graph
    
    # Final Output
    ordered_path: List[Dict[str, Any]]          # Pruned & Kahn's topologically sorted tutorials
    syllabus: Optional[str]                     # Final personalized, paced roadmap


class DiagnosticGraphAgent:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
        self.embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", threads=1)
        self.checkpointer = MemorySaver()
    def close(self):
        self.driver.close()
    def _call_sarvam(self, system_prompt: str, user_prompt: str) -> str:
        """Helper to call Sarvam conversational endpoint."""
        url = "https://api.sarvam.ai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "api-subscription-key": SARVAM_API_KEY,
        }
        payload = {
            "model": "sarvam-105b-conversations",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 1500,
            "temperature": 0.2,
        }
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=60)
            if res.status_code == 200:
                data = res.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[Warning] Sarvam call failed: {e}")
        return ""

    def node_extract_goal(self, state: AgentState) -> dict:
        """Distills raw user intent into a clean technical search target."""
        raw_query = state.get("raw_query", "")
        background = state.get("learner_background", "Beginner")
        time_budget = state.get("time_budget_hours", 5)
        system_prompt = (
            "You are an expert technical curriculum advisor. "
            "Your job is to convert a user's conversational learning goal into a "
            "concise, technically precise topic summary (1-2 sentences max) suitable for semantic search "
            "over web development course catalogs (HTML, CSS, Bootstrap, JavaScript). "
            "Output ONLY the concise topic summary, nothing else."
        )
        user_prompt = (
            f"User Goal: {raw_query}\n"
            f"Learner Background: {background}\n"
            f"Available Time: {time_budget} hours\n"
            "Distilled Technical Topic:"
        )
        extracted = self._call_sarvam(system_prompt, user_prompt).strip()
        if not extracted:
            extracted = raw_query
        print(f"\n[Agent: Node 1] Distilled Goal: '{extracted}'")
        return {"extracted_goal": extracted}

    def node_retrieve_seeds(self, state: AgentState) -> dict:
        """Finds target tutorials in Neo4j using the native vector index."""
        goal = state["extracted_goal"]
        goal_vec = [float(v) for v in list(self.embed_model.embed([goal]))[0]]
        # Adjust top_k based on time budget (tight time = 2 seeds, generous = 3 seeds)
        top_k = 2 if state.get("time_budget_hours", 5) <= 3 else 3
        query = """
        CALL db.index.vector.queryNodes('tutorial_index', $top_k, $goal_vec)
        YIELD node, score
        RETURN node.id AS id, node.title AS title, node.course_name AS course,
               node.summary AS summary, node.url AS url, node.level AS level,
               node.order_index AS order_index, score
        """
        with self.driver.session() as session:
            seeds = session.run(query, top_k=top_k, goal_vec=goal_vec).data()
        print(f"[Agent: Node 2] Found {len(seeds)} Target Lessons in Neo4j:")
        for s in seeds:
            print(f"      • [{s['course']}] {s['title']} (score: {s['score']:.3f})")
        return {"seeds": seeds}

    def node_retrieve_prerequisites(self, state: AgentState) -> dict:
        """
        Traverses backward along [:DEPENDS_ON] in Neo4j to gather prerequisites,
        and retrieves the (:Concept) nodes taught by those prerequisites.
        """
        seed_ids = [s["id"] for s in state["seeds"]]
        time_budget = state.get("time_budget_hours", 5)
        # Dynamic graph depth based on time constraint:
        # 3 hours or less: quick 2-hop sprint; more hours: deep 4-hop path
        max_hops = 2 if time_budget <= 3 else 4
        query = f"""
        MATCH path = (target:Tutorial)-[:DEPENDS_ON*0..{max_hops}]->(prereq:Tutorial)
        WHERE target.id IN $seed_ids
        WITH DISTINCT prereq
        OPTIONAL MATCH (prereq)-[:DEPENDS_ON]->(upstream:Tutorial)
        OPTIONAL MATCH (prereq)-[:TEACHES]->(c:Concept)
        RETURN prereq.id AS id, prereq.title AS title, prereq.course_name AS course,
               prereq.summary AS summary, prereq.url AS url, prereq.level AS level,
               prereq.order_index AS order_index,
               collect(DISTINCT upstream.id) AS dependencies,
               collect(DISTINCT c.name) AS concepts
        """
        with self.driver.session() as session:
            records = session.run(query, seed_ids=seed_ids).data()
        candidates = []
        edges = []
        prereq_concepts = []
        for r in records:
            candidates.append({
                "id": r["id"],
                "title": r["title"],
                "course": r["course"],
                "summary": r["summary"],
                "url": r["url"],
                "level": r["level"],
                "order_index": r["order_index"],
                "concepts": r["concepts"],
            })
            for dep_id in r["dependencies"]:
                if dep_id:
                    edges.append({"source": dep_id, "target": r["id"]})
            # If this is an upstream prerequisite (not one of our destination seeds),
            # save its concepts for the diagnostic quiz
            if r["id"] not in seed_ids and r["concepts"]:
                prereq_concepts.append({
                    "tutorial_id": r["id"],
                    "tutorial_title": r["title"],
                    "concepts": r["concepts"][:4],  # Top 4 concepts
                })
        print(f"[Agent: Node 3] Graph Traversal collected {len(candidates)} total lessons.")
        print(f"      Prerequisite lessons to test: {len(prereq_concepts)}")
        return {
            "candidate_tutorials": candidates,
            "dependency_edges": edges,
            "prerequisite_concepts": prereq_concepts,
        }

    def node_generate_quiz(self, state: AgentState) -> dict:
        """
        Generates 2 multiple-choice questions on prerequisite concepts,
        then interrupts graph execution to wait for user answers.
        """
        prereqs = state.get("prerequisite_concepts", [])
        if not prereqs:
            print("[Agent: Node 4] No upstream prerequisites to test. Skipping quiz.")
            return {"quiz": [], "user_answers": "skip"}
        # Format prerequisite concepts into a concise prompt for Sarvam
        concept_summary = ""
        for p in prereqs[:3]:
            concept_summary += f"- Lesson '{p['tutorial_title']}' (ID: {p['tutorial_id']}): {', '.join(p['concepts'])}\n"
        system_prompt = (
            "You are a technical diagnostic quiz generator for web developers. "
            "Generate exactly 2 multiple-choice diagnostic questions to test if a student understands "
            "the foundational concepts listed below. "
            "Each question MUST have 3 options: A), B), and C). "
            "Return your output as a VALID JSON array of objects with keys: "
            "'tutorial_id', 'question', 'options' (list of 3 strings), and 'correct_option' (A, B, or C). "
            "Do NOT include extra commentary or explanation."
        )
        user_prompt = (
            f"Generate a 2-question quiz testing these prerequisite lessons:\n"
            f"{concept_summary}\n\n"
            "Output JSON:"
        )
        raw_response = self._call_sarvam(system_prompt, user_prompt).strip()
        # Parse JSON output from Sarvam (strip markdown code fences if present)
        quiz = []
        try:
            cleaned = raw_response.replace("```json", "").replace("```", "").strip()
            quiz = json.loads(cleaned)
        except Exception:
            # Fallback if LLM returned plain text
            quiz = [{
                "tutorial_id": prereqs[0]["tutorial_id"],
                "question": f"Are you familiar with {prereqs[0]['tutorial_title']}?",
                "options": ["A) Yes, fully mastered", "B) Somewhat familiar", "C) Brand new to it"],
                "correct_option": "A"
            }]
        print(f"\n[Agent: Node 4] Generated {len(quiz)} Diagnostic Pulse-Check Questions.")
        # =====================================================================
        # LANGGRAPH INTERRUPT: Pauses graph execution for user input!
        # The return value of interrupt() is whatever the user provides when resuming.
        # =====================================================================
        user_response = interrupt({
            "type": "diagnostic_quiz",
            "quiz": quiz,
            "instructions": "Answer the questions (e.g. '1B, 2A') or type 'skip' if you are a beginner."
        })
        return {"quiz": quiz, "user_answers": str(user_response)}

    def _topological_sort(
        self, tutorials: List[Dict[str, Any]], edges: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Mathematically orders tutorials using Kahn's Algorithm with course prioritization."""
        from collections import defaultdict, deque
        course_priority = {"HTML": 1, "CSS": 2, "Bootstrap": 3, "JavaScript": 4}
        tuts = {t["id"]: t for t in tutorials}
        # Build in-degree map and adjacency list (prereq -> dependent)
        adj = defaultdict(list)
        in_degree = defaultdict(int)
        for tid in tuts:
            in_degree[tid] = 0
        for edge in edges:
            src, dst = edge["source"], edge["target"]
            if src in tuts and dst in tuts and src != dst:
                adj[src].append(dst)
                in_degree[dst] += 1
        # Kahn's Algorithm: start with tutorials having 0 unsatisfied prerequisites
        zero_in_degree = [tid for tid in tuts if in_degree[tid] == 0]
        zero_in_degree.sort(
            key=lambda tid: (
                course_priority.get(tuts[tid]["course"], 99),
                tuts[tid]["order_index"],
            )
        )
        queue = deque(zero_in_degree)
        sorted_path = []
        while queue:
            curr_id = queue.popleft()
            sorted_path.append(tuts[curr_id])
            newly_ready = []
            for neighbor_id in adj[curr_id]:
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    newly_ready.append(neighbor_id)
            newly_ready.sort(
                key=lambda tid: (
                    course_priority.get(tuts[tid]["course"], 99),
                    tuts[tid]["order_index"],
                )
            )
            queue.extend(newly_ready)
        return sorted_path
    # -------------------------------------------------------------------------
    # NODE 5: Answer Evaluation & Graph Pruning
    # -------------------------------------------------------------------------
    def node_evaluate_and_prune(self, state: AgentState) -> dict:
        """
        Evaluates user quiz answers, prunes mastered tutorials,
        and computes the topologically sorted path on the remaining graph.
        """
        quiz = state.get("quiz", [])
        answers = state.get("user_answers", "").strip()
        candidates = state.get("candidate_tutorials", [])
        edges = state.get("dependency_edges", [])
        mastered_ids = []
        # If user skipped or no quiz was given, keep all candidate tutorials
        if not answers or answers.lower() == "skip" or not quiz:
            print("\n[Agent: Node 5] No pruning requested. Retaining full candidate path.")
        else:
            system_prompt = (
                "You are an assessment evaluator for a programming course. "
                "Given a 2-question quiz with correct answers and the student's submitted response, "
                "determine which tutorial IDs the student demonstrated clear competence in. "
                "Output ONLY a valid JSON list of tutorial_id strings that should be SKIPPED/PRUNED. "
                "Example: [\"1071\", \"1072\"] or [] if they got questions wrong."
            )
            user_prompt = (
                f"Quiz Details:\n{json.dumps(quiz, indent=2)}\n\n"
                f"Student Answers: {answers}\n\n"
                "JSON list of mastered tutorial_ids to prune:"
            )
            eval_res = self._call_sarvam(system_prompt, user_prompt).strip()
            try:
                cleaned = eval_res.replace("```json", "").replace("```", "").strip()
                mastered_ids = json.loads(cleaned)
                if not isinstance(mastered_ids, list):
                    mastered_ids = []
            except Exception:
                mastered_ids = []
            print(f"\n[Agent: Node 5] Evaluation Complete:")
            print(f"      Pruned Mastered Lessons: {mastered_ids}")
        # Prune mastered tutorials from candidate set
        pruned_candidates = [t for t in candidates if t["id"] not in mastered_ids]
        # Mathematically sort the remaining tutorials using Kahn's algorithm
        ordered = self._topological_sort(pruned_candidates, edges)
        print(f"[Agent: Node 5] Final Pruned Path Length: {len(ordered)} lessons.")
        for idx, t in enumerate(ordered, 1):
            print(f"      {idx}. [{t['course']}] {t['title']}")
        return {
            "mastered_ids": mastered_ids,
            "ordered_path": ordered,
        }

    def node_synthesize_curriculum(self, state: AgentState) -> dict:
        """Calls Sarvam to synthesize the final personalized, time-paced roadmap."""
        ordered = state.get("ordered_path", [])
        goal = state.get("raw_query", "")
        background = state.get("learner_background", "Beginner")
        time_budget = state.get("time_budget_hours", 5)
        mastered_ids = state.get("mastered_ids", [])
        if not ordered:
            return {"syllabus": "No learning path could be generated."}
        # Build context list with authentic Spoken Tutorial URLs
        lessons_text = ""
        for i, t in enumerate(ordered, 1):
            lessons_text += (
                f"\nLesson {i}: [{t['course']}] {t['title']} (Level: {t['level']})\n"
                f"  Video URL: {t['url']}\n"
                f"  Summary: {t['summary']}\n"
            )
        system_prompt = (
            "You are an inspiring, pedagogical master educator at Spoken Tutorial. "
            "Write a beautifully formatted Markdown learning path tailored to the student's background and time budget. "
            "Follow these strict rules:\n"
            "1. Pace the plan specifically to fit their available time budget.\n"
            "2. If any prerequisite lessons were skipped due to the diagnostic quiz, celebrate this and briefly mention them.\n"
            "3. Include a Timeline Overview Table with estimated hours/days for each phase.\n"
            "4. For EACH lesson in the provided sequence, include its exact Video URL and 2-3 specific hands-on practice tasks.\n"
            "5. Conclude with a Capstone Project that connects all lessons together."
        )
        user_prompt = (
            f"Learner Goal: {goal}\n"
            f"Background: {background}\n"
            f"Available Time: {time_budget} hours\n"
            f"Skipped Lessons (Mastered via Diagnostic): {len(mastered_ids)} lessons\n\n"
            f"Ordered Lessons to Include in Syllabus:\n{lessons_text}\n\n"
            "Generate the complete, personalized Markdown roadmap:"
        )
        print("\n[Agent: Node 6] Synthesizing personalized roadmap with Sarvam AI...")
        roadmap = self._call_sarvam(system_prompt, user_prompt)
        return {"syllabus": roadmap}
        
    def build_graph(self):
        """Assembles and compiles the StateGraph with MemorySaver checkpointer."""
        builder = StateGraph(AgentState)
        # Register all nodes
        builder.add_node("extract_goal", self.node_extract_goal)
        builder.add_node("retrieve_seeds", self.node_retrieve_seeds)
        builder.add_node("retrieve_prereqs", self.node_retrieve_prerequisites)
        builder.add_node("generate_quiz", self.node_generate_quiz)
        builder.add_node("evaluate_and_prune", self.node_evaluate_and_prune)
        builder.add_node("synthesize_curriculum", self.node_synthesize_curriculum)
        # Wire linear flow with interrupt pause point inside generate_quiz
        builder.add_edge(START, "extract_goal")
        builder.add_edge("extract_goal", "retrieve_seeds")
        builder.add_edge("retrieve_seeds", "retrieve_prereqs")
        builder.add_edge("retrieve_prereqs", "generate_quiz")
        builder.add_edge("generate_quiz", "evaluate_and_prune")
        builder.add_edge("evaluate_and_prune", "synthesize_curriculum")
        builder.add_edge("synthesize_curriculum", END)
        return builder.compile(checkpointer=self.checkpointer)
# =============================================================================
# 4. INTERACTIVE MULTI-TURN CLI RUNNER
# =============================================================================
def main():
    agent = DiagnosticGraphAgent()
    app = agent.build_graph()
    try:
        print("=" * 65)
        print("     🧠 LANGGRAPH MULTI-TURN DIAGNOSTIC AGENT (GraphRAG)")
        print("=" * 65)
        # Intake turn: Goal, Background, Time Budget
        goal = input("\n🎯 What do you want to learn or build?\n> ").strip()
        if not goal:
            goal = "I want to build responsive grid layouts using Bootstrap"
        bg = input("\n🎓 What is your current programming background? (Press Enter if Beginner)\n> ").strip()
        if not bg:
            bg = "Beginner"
        hours_input = input("\n⏱️  How many total hours do you have to study? (Default: 5)\n> ").strip()
        try:
            hours = int(hours_input) if hours_input else 5
        except ValueError:
            hours = 5
        # Unique thread session ID for this conversation
        thread_id = "user_session_1"
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {
            "raw_query": goal,
            "learner_background": bg,
            "time_budget_hours": hours,
        }
        print("\n" + "-" * 65)
        print("🚀 [Turn 1] Agent is analyzing your goal and querying Neo4j...")
        print("-" * 65)
        # Kick off graph execution
        result = app.invoke(initial_state, config=config)
        # Check if the agent paused for human input (diagnostic quiz)
        state = app.get_state(config)
        if state.next:
            interrupts = [i.value for t in state.tasks for i in t.interrupts]
            if interrupts:
                quiz_data = interrupts[0]
                quiz = quiz_data.get("quiz", [])
                print("\n" + "=" * 65)
                print("           📝 60-SECOND PREREQUISITE DIAGNOSTIC CHECK")
                print("=" * 65)
                print("To skip lessons you already know, answer these quick questions:")
                print("(Or type 'skip' if you are new and want the complete foundational path)\n")
                for i, q in enumerate(quiz, 1):
                    print(f"Q{i}: {q['question']}")
                    for opt in q["options"]:
                        print(f"    {opt}")
                    print()
                user_ans = input("Your answers (e.g., '1B, 2A' or 'skip'):\n> ").strip()
                if not user_ans:
                    user_ans = "skip"
                print("\n" + "-" * 65)
                print("⚙️  [Turn 2] Resuming agent: Evaluating answers & pruning graph...")
                print("-" * 65)
                # Resume graph execution with user answers!
                result = app.invoke(Command(resume=user_ans), config=config)
        print("\n" + "=" * 65)
        print("                  🎉 YOUR PERSONALIZED ROADMAP")
        print("=" * 65 + "\n")
        print(result.get("syllabus", "No syllabus generated."))
    finally:
        agent.close()
if __name__ == "__main__":
    main()
