import os
os.environ["KAGGLE_CONFIG_DIR"] = os.path.join(os.getcwd(), "credentials")

import subprocess
import json
import csv
import io
import requests
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from kaggle.api.kaggle_api_extended import KaggleApi

# ---------------------------
# Load env
# ---------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API")

# ---------------------------
# Initialize LLM (FILTER ONLY)
# ---------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    api_key=GROQ_API_KEY
)

# ---------------------------
# Step 1A: Kaggle Search
# ---------------------------
def search_kaggle_datasets(query: str, max_results: int = 5):
    try:
        api = KaggleApi()
        api.authenticate()

        results = api.dataset_list(search=query)

        datasets = []
        for ds in results[:max_results]:
            datasets.append({
                "name": ds.title,
                "source": "kaggle",
                "url": f"https://www.kaggle.com/datasets/{ds.ref}",
                "description": ds.subtitle or "",
                "size": getattr(ds, "size", "unknown"),
                "last_updated": str(getattr(ds, "lastUpdated", "unknown")),
                "votes": getattr(ds, "voteCount", 0)
            })

        return datasets

    except Exception as e:
        print("[KAGGLE API ERROR]:", e)
        return []


# ---------------------------
# Step 1B: HuggingFace Datasets
# ---------------------------
def search_hf_datasets(query: str, max_results: int = 5):
    url = f"https://huggingface.co/api/datasets?search={query}"

    try:
        response = requests.get(url)
        data = response.json()

        datasets = []
        for item in data[:max_results]:
            datasets.append({
                "name": item.get("id"),
                "source": "huggingface",
                "url": f"https://huggingface.co/datasets/{item.get('id')}",
                "description": item.get("description", "")[:200]
            })

        return datasets

    except Exception as e:
        print("[HF ERROR]:", e)
        return []


# ---------------------------
# Step 1C: GitHub Dataset Search
# ---------------------------
def search_github_datasets(query: str, max_results: int = 5):
    url = f"https://api.github.com/search/repositories?q={query}+dataset"

    try:
        response = requests.get(url)
        data = response.json()

        datasets = []
        for repo in data.get("items", [])[:max_results]:
            datasets.append({
                "name": repo.get("name"),
                "source": "github",
                "url": repo.get("html_url"),
                "description": repo.get("description", "")
            })

        return datasets

    except Exception as e:
        print("[GitHub ERROR]:", e)
        return []


# ---------------------------
# Step 2: Merge + Deduplicate
# ---------------------------
def merge_and_deduplicate(*sources):
    combined = []
    for src in sources:
        combined.extend(src)

    seen = set()
    unique = []

    for d in combined:
        name = d.get("name", "").lower().strip()
        if name and name not in seen:
            unique.append(d)
            seen.add(name)

    return unique


# ---------------------------
# Step 3: Ranking
# ---------------------------
def rank_datasets(datasets):
    return sorted(
        datasets,
        key=lambda x: len(x.get("description", "")),
        reverse=True
    )


# ---------------------------
# Step 4: LLM FILTER (🔥 KEY)
# ---------------------------
parser = JsonOutputParser()
format_instructions = parser.get_format_instructions()

filter_prompt = ChatPromptTemplate.from_messages([
    ("system", f"""
You are a dataset filtering agent.

Return ONLY valid JSON.

STRICT FORMAT:
{{
  "datasets": [
    {{
      "name": "",
      "source": "",
      "url": "",
      "description": ""
    }}
  ]
}}

RULES:
- DO NOT return a list directly
- DO NOT return anything outside JSON
- ONLY select from given datasets
- DO NOT modify dataset fields

{format_instructions}
"""),
    ("human", """
Query:
{query}

Datasets:
{datasets}
""")
])

filter_chain = filter_prompt | llm | parser


def filter_datasets(query, datasets):
    if not datasets:
        return []

    try:
        response = filter_chain.invoke({
            "query": query,
            "datasets": json.dumps(datasets, indent=2)
        })

        # ✅ HANDLE MULTIPLE RESPONSE TYPES
        if isinstance(response, list):
            filtered = response

        elif isinstance(response, dict):
            filtered = response.get("datasets", [])

        else:
            print("⚠️ Unexpected LLM response type")
            return datasets[:3]

        # ✅ VALIDATE against original datasets
        original_names = set(d["name"] for d in datasets)

        filtered = [
            d for d in filtered
            if isinstance(d, dict) and d.get("name") in original_names
        ]

        if not filtered:
            print("⚠️ LLM returned empty → fallback")
            return datasets[:3]

        return filtered

    except Exception as e:
        print("[FILTER ERROR]:", e)
        return datasets


# ---------------------------
# Step 5: Limit
# ---------------------------
def limit_datasets(datasets, max_total=6):
    return datasets[:max_total]


# ---------------------------
# Step 6: Run Dataset Agent
# ---------------------------
def run_dataset_agent(query: str) -> dict:
    print("📊 Searching Kaggle...")
    kaggle_data = search_kaggle_datasets(query, 6)

    print("📊 Searching HuggingFace...")
    hf_data = search_hf_datasets(query, 6)

    print("📊 Searching GitHub...")
    github_data = search_github_datasets(query, 6)

    all_datasets = merge_and_deduplicate(
        kaggle_data,
        hf_data,
        github_data
    )

    if not all_datasets:
        return {"datasets": []}

    # Step 1: Rank
    all_datasets = rank_datasets(all_datasets)

    # Step 2: Reduce input to LLM
    all_datasets = all_datasets[:10]

    # Step 3: LLM Filter 🔥
    print("🧠 Filtering datasets...")
    filtered = filter_datasets(query, all_datasets)

    # Step 4: Final limit
    final_datasets = limit_datasets(filtered, 6)

    output = {
        "query": query,
        "total_datasets": len(final_datasets),
        "datasets": final_datasets
    }

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/dataset_agent_output.json", "w") as f:
        json.dump(output, f, indent=4)

    return output


# ---------------------------
# Entry Point
# ---------------------------
if __name__ == "__main__":
    query = "Plant Image Disease Prediction"
    result = run_dataset_agent(query)
    print(json.dumps(result, indent=4))