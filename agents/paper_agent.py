import arxiv
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
# Step 1: Fetch papers from arXiv
# ---------------------------
def fetch_arxiv_papers(query: str, max_results: int = 5):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []
    for result in search.results():
        papers.append({
            "title": result.title,
            "year": result.published.year,
            "summary": result.summary,
            "authors": [a.name for a in result.authors],
            "pdf_url": result.pdf_url
        })
    return papers


# ---------------------------
# Step 2: Paper summarization prompt
# ---------------------------
paper_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a Paper Review Agent.

You will be given REAL research papers retrieved from arXiv.
Your job is to summarize them.

Rules:
- ONLY use provided paper content
- Do NOT invent papers
- Do NOT analyze gaps
- Do NOT suggest datasets or plans
- Output ONLY valid JSON

JSON Schema:
{{
  "papers": [
    {{
      "title": "",
      "year": "",
      "methodology": "",
      "data_used": "",
      "key_contribution": ""
    }}
  ],
  "overall_trends": []
}}
"""),
    ("human", """
Research Query:
{query}

Retrieved Papers:
{papers}
""")
])

parser = JsonOutputParser()

paper_chain = paper_prompt | llm | parser


# ---------------------------
# Step 3: Run Paper Agent
# ---------------------------
def run_paper_agent(query: str) -> dict:
    arxiv_papers = fetch_arxiv_papers(query)

    output = paper_chain.invoke({
        "query": query,
        "papers": json.dumps(arxiv_papers, indent=2)
    })

    with open("outputs/paper_agent_output.json", "w") as f:
        json.dump(output, f, indent=4)

    return output


# ---------------------------
# Entry Point
# ---------------------------
if __name__ == "__main__":
    query = "Deep learning methods for landslide detection using satellite imagery"
    result = run_paper_agent(query)
    print(json.dumps(result, indent=4))
