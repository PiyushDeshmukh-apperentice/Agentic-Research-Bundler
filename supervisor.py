import json
import os

from agents.planner_agent import run_planner
from agents.paper_agent import run_paper_agent
from agents.dataset_agent import run_dataset_agent
from agents.actionPlan_agent import run_action_plan_agent


def run_research_system(query: str):
    print("🔍 Running planner...")
    plan = run_planner(query)

    # Derive which agents to invoke from planner subtasks (planner returns 'subtasks')
    agents_invoked = [s.get("agent") for s in plan.get("subtasks", [])]
    # attach for transparency/backwards-compat
    if isinstance(plan, dict):
        plan["agents_invoked"] = agents_invoked

    results = {
        "query": query,
        "planner": plan,
        "papers": None,
        "datasets": None,
        "action_plan": None
    }

    if "paper_agent" in agents_invoked:
        print("📄 Running paper agent...")
        results["papers"] = run_paper_agent(query)
        # Prefer the saved agent output JSON (contains full fields) if available
        try:
            with open("outputs/paper_agent_output.json", "r") as f:
                results["papers"] = json.load(f)
        except Exception:
            pass

    if "dataset_agent" in agents_invoked:
        print("📊 Running dataset agent...")
        results["datasets"] = run_dataset_agent(query)
        try:
            with open("outputs/dataset_agent_output.json", "r") as f:
                results["datasets"] = json.load(f)
        except Exception:
            pass

    if (
        "action_plan_agent" in agents_invoked
        and results["papers"]
        and results["datasets"]
    ):
        print("🧭 Running action plan agent...")
        results["action_plan"] = run_action_plan_agent(
            query=query,
            papers=results["papers"],
            datasets=results["datasets"]
        )
        # If agent saved an output file, prefer that (it's more complete)
        try:
            with open("outputs/action_plan_agent_output.json", "r") as f:
                results["action_plan"] = json.load(f)
        except Exception:
            pass

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/research_bundle.json", "w") as f:
        json.dump(results, f, indent=4)

    print("✅ Research bundle created: outputs/research_bundle.json")
    return results


if __name__ == "__main__":
    query = "Deep learning methods for landslide detection using satellite imagery"
    run_research_system(query)
