import arxiv
import json
import os
import requests
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API")
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"

# ---------------------------
# Initialize LLM (Groq) -> llama-3.3-70b-versatile
# ---------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    api_key=GROQ_API_KEY
)

# ---------------------------
# Step 1A: Fetch papers from arXiv
# ---------------------------
def fetch_arxiv_papers(query: str, max_results: int = 5): # Increased harvest
    client = arxiv.Client(
        page_size=max_results,
        delay_seconds=3,
        num_retries=3
    )

    clean_query = query.strip().replace(":", "").replace('"', '')

    search = arxiv.Search(
        query=clean_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []
    try:
        results = client.results(search)
        for result in results:
            papers.append({
                "title": result.title,
                "year": result.published.year,
                "summary": result.summary[:500],
                "authors": [a.name for a in result.authors],
                "source": "arxiv",
                "link": result.pdf_url
            })
    except Exception as e:
        print(f"⚠️ arXiv API Error: {e}")
        return []

    return papers


# ---------------------------
# Step 1B: Fetch papers from Semantic Scholar
# ---------------------------
def fetch_semantic_scholar_papers(query: str, max_results: int = 5): # Increased harvest
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,year,abstract,authors,url,citationCount"
    }

    try:
        response = requests.get(SEMANTIC_SCHOLAR_API, params=params)
        data = response.json()

        papers = []
        for paper in data.get("data", []):
            papers.append({
                "title": paper.get("title"),
                "year": paper.get("year"),
                "summary": (paper.get("abstract") or "")[:500],
                "authors": [a.get("name") for a in paper.get("authors", [])],
                "source": "semantic_scholar",
                "link": paper.get("url"),
                "citations": paper.get("citationCount", 0)
            })
        return papers
    except Exception as e:
        print("Semantic Scholar Error:", e)
        return []


# ---------------------------
# Step 2: Merge + Deduplicate
# ---------------------------
def merge_and_deduplicate(arxiv_papers, semantic_papers):
    combined = arxiv_papers + semantic_papers
    seen_titles = set()
    unique_papers = []

    for paper in combined:
        title = paper["title"].lower().strip()
        if title not in seen_titles:
            unique_papers.append(paper)
            seen_titles.add(title)
    return unique_papers


# ---------------------------
# Step 3: Ranking
# ---------------------------
def rank_papers(papers):
    return sorted(
        papers,
        key=lambda x: (x.get("citations", 0), x.get("year", 0)),
        reverse=True
    )


# ---------------------------
# Step 4: LLM FILTER
# ---------------------------
parser = JsonOutputParser()
format_instructions = parser.get_format_instructions()

filter_prompt = ChatPromptTemplate.from_messages([
    ("system", f"""
You are a research paper filtering agent.
Your job is to identify and retain all papers that are relevant to the user's research query.

STRICT RULES:
- Aim to retain as many relevant papers as possible (be inclusive).
- DO NOT modify paper content or summarize.
- RETURN the results as a JSON object with a key "papers" containing the list.
- If the query is broad, keep more papers.

{format_instructions}
"""),
    ("human", "Query: {query}\n\nPapers:\n{papers}")
])

filter_chain = filter_prompt | llm | parser

def filter_relevant_papers(query, papers):
    if not papers:
        return []

    try:
        response = filter_chain.invoke({
            "query": query,
            "papers": json.dumps(papers, indent=2)
        })

        # --- FIX: ROBUST RESPONSE HANDLING ---
        if isinstance(response, list):
            filtered = response
        elif isinstance(response, dict):
            filtered = response.get("papers", [])
        else:
            return papers[:5] # Fallback if response is weird

        # Ensure returned papers are actually from our original list
        original_titles = {p["title"].lower().strip() for p in papers}
        validated_filtered = [
            p for p in filtered 
            if isinstance(p, dict) and p.get("title", "").lower().strip() in original_titles
        ]

        if not validated_filtered:
            return papers[:5] # Fallback to top 15 if LLM fails

        return validated_filtered

    except Exception as e:
        print(f"LLM Filter Error: {e}")
        return papers[:15] # High volume fallback


# ---------------------------
# Step 6: Run Paper Agent
# ---------------------------
def run_paper_agent(query: str) -> dict:
    # --- 1. Fetch from arXiv ---
    print("🔍 Fetching papers from arXiv...")
    arxiv_papers = fetch_arxiv_papers(query, max_results=5)
    
    # If arXiv hit a rate limit, arxiv_papers will be [] because of your try-except
    if not arxiv_papers:
        print("⚠️ arXiv unavailable (Rate Limit/Error). Continuing with remaining sources...")
    
    # Optional: Small sleep to prevent simultaneous "bursting" across APIs
    import time
    time.sleep(1) 

    # --- 2. Fetch from Semantic Scholar ---
    print("🔍 Fetching papers from Semantic Scholar...")
    semantic_papers = fetch_semantic_scholar_papers(query, max_results=20)
    
    if not semantic_papers:
        print("⚠️ Semantic Scholar unavailable. Continuing with remaining sources...")

    # --- 3. Merge results ---
    # merge_and_deduplicate will handle cases where one list is empty
    all_papers = merge_and_deduplicate(arxiv_papers, semantic_papers)

    # Only exit if BOTH failed
    if not all_papers:
        print("❌ Both sources failed to return results.")
        return {"papers": []}

    # --- 4. Rank and Filter ---
    all_papers = rank_papers(all_papers)
    
    # Feed the top available results to the LLM
    all_papers_chunk = all_papers[:5]

    print(f"🧠 Filtering {len(all_papers_chunk)} papers using LLM...")
    filtered_papers = filter_relevant_papers(query, all_papers_chunk)

    final_papers = filtered_papers[:5]

    output = {
        "query": query,
        "total_papers": len(final_papers),
        "papers": final_papers
    }

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/paper_agent_output.json", "w") as f:
        json.dump(output, f, indent=4)

    return output


if __name__ == "__main__":
    result = run_paper_agent("Machine Learning for Cybersecurity")
    print(json.dumps(result, indent=4))