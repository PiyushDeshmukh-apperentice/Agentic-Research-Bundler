import json
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    api_key=os.getenv("GROQ_API")
)

# ---------------------------
# Prompt
# ---------------------------
action_plan_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an Action Plan Agent.

You are given:
- A research query
- Retrieved research papers
- Retrieved datasets

Your job is to create a neutral, step-by-step research execution plan.

Rules:
- Do NOT identify research gaps
- Do NOT judge novelty
- Do NOT invent sources
- ONLY use provided inputs
- Output ONLY valid JSON

Your plan should reflect standard research workflow.
"""),
    ("human", """
Research Query:
{query}

Papers:
{papers}

Datasets:
{datasets}
""")
])

parser = JsonOutputParser()
action_plan_chain = action_plan_prompt | llm | parser


# ---------------------------
# Runner
# ---------------------------
def run_action_plan_agent(query: str, papers: dict, datasets: dict) -> dict:
    output = action_plan_chain.invoke({
        "query": query,
        "papers": json.dumps(papers, indent=2),
        "datasets": json.dumps(datasets, indent=2)
    })

    with open("outputs/action_plan_agent_output.json", "w") as f:
        json.dump(output, f, indent=4)

    return output


if __name__ == "__main__":
    print("Run via supervisor, not directly.")
