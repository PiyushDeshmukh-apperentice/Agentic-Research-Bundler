import subprocess
import json
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Load env
load_dotenv()

# Initialize LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
    api_key=os.getenv("GROQ_API")
)

# ---------------------------
# Step 1: Query Kaggle API
# ---------------------------
import csv
import subprocess
import io

def search_kaggle_datasets(query: str, max_results: int = 5):
    command = [
        "kaggle", "datasets", "list",
        "--search", query,
        "--csv"
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True
    )

    datasets = []
    csv_reader = csv.DictReader(io.StringIO(result.stdout))

    for row in csv_reader:
        if len(datasets) >= max_results:
            break

        dataset_ref = row["ref"]

        datasets.append({
            "name": row["title"],
            "ref": dataset_ref,
            "url": f"https://www.kaggle.com/datasets/{dataset_ref}"
        })

    return datasets


# ---------------------------
# Step 2: Dataset summarization prompt
# ---------------------------
dataset_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a Dataset Discovery Agent.

You are given REAL datasets retrieved from Kaggle.
Your job is to analyze their suitability.

Rules:
- ONLY use provided dataset info
- Do NOT invent datasets
- Do NOT suggest models
- Output ONLY valid JSON

JSON Schema:
{{
  "datasets": [
    {{
      "name": "",
      "source": "Kaggle",
      "task_type": "",
      "data_type": "",
      "labels": "",
      "url": ""
    }}
  ],
  "coverage_notes": []
}}
"""),
    ("human", """
Research Query:
{query}

Retrieved Datasets:
{datasets}
""")
])

parser = JsonOutputParser()
dataset_chain = dataset_prompt | llm | parser


# ---------------------------
# Step 3: Run Dataset Agent
# ---------------------------
def run_dataset_agent(query: str) -> dict:
    kaggle_datasets = search_kaggle_datasets(query)

    output = dataset_chain.invoke({
        "query": query,
        "datasets": json.dumps(kaggle_datasets, indent=2)
    })

    with open("outputs/dataset_agent_output.json", "w") as f:
        json.dump(output, f, indent=4)

    return output


# ---------------------------
# Entry Point
# ---------------------------
if __name__ == "__main__":
    query = "Deep learning landslide detection using satellite imagery"
    result = run_dataset_agent(query)
    print(json.dumps(result, indent=4))
