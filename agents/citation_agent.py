import json
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class CitationAgent:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            api_key=os.getenv("GROQ_API")
        )

    def _format_authors_apa(self, authors):
        if not authors:
            return "Unknown Author"
        if len(authors) == 1:
            return authors[0]
        if len(authors) == 2:
            return f"{authors[0]} & {authors[1]}"
        return f"{authors[0]} et al."

    def generate_apa(self, paper):
        """Generates a standard APA citation."""
        authors = self._format_authors_apa(paper.get("authors", []))
        year = paper.get("year", "n.d.")
        title = paper.get("title", "Untitled")
        source = paper.get("source", "Web").replace("_", " ").title()
        return f"{authors} ({year}). {title}. {source}. Retrieved from {paper.get('link')}"

    def generate_ieee(self, index, paper):
        """Generates a standard IEEE citation."""
        authors = ", ".join(paper.get("authors", []))
        if len(paper.get("authors", [])) > 3:
            authors = f"{paper.get('authors')[0]} et al."
        
        title = f"\"{paper.get('title')}\""
        year = paper.get("year", "n.d.")
        return f"[{index}] {authors}, {title}, {year}. [Online]. Available: {paper.get('link')}"

    def generate_bibtex(self, paper):
        """Uses LLM to generate a valid BibTeX entry to handle complex author/title parsing."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a LaTeX and BibTeX expert. Convert the provided paper metadata into a single, valid @article or @misc BibTeX entry. Return ONLY the BibTeX code block."),
            ("human", "Paper Metadata: {paper_json}")
        ])
        
        chain = prompt | self.llm
        try:
            # We pass a cleaned version of the paper to save tokens
            clean_paper = {
                "title": paper.get("title"),
                "authors": paper.get("authors"),
                "year": paper.get("year"),
                "link": paper.get("link")
            }
            response = chain.invoke({"paper_json": json.dumps(clean_paper)})
            return response.content.strip().replace("```bibtex", "").replace("```", "")
        except Exception as e:
            return f"% Error generating BibTeX for {paper.get('title')}: {e}"

    def run(self, paper_data):
        """
        Input: Dictionary from Paper Agent (output['papers'])
        Output: Dictionary of formatted citations
        """
        papers = paper_data.get("papers", [])
        if not papers:
            return {"apa": [], "ieee": [], "bibtex": ""}

        results = {
            "apa": [self.generate_apa(p) for p in papers],
            "ieee": [self.generate_ieee(i+1, p) for i, p in enumerate(papers)],
            "bibtex": "\n\n".join([self.generate_bibtex(p) for p in papers])
        }

        # Save to output folder
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/citation_agent_output.json", "w") as f:
            json.dump(results, f, indent=4)

        return results

# ---------------------------
# Entry Point for Testing
# ---------------------------
if __name__ == "__main__":
    # Mock data mimicking Paper Agent output
    mock_papers = {
        "papers": [
            {
                "title": "Attention Is All You Need",
                "authors": ["Ashish Vaswani", "Noam Shazeer"],
                "year": 2017,
                "source": "arxiv",
                "link": "https://arxiv.org/pdf/1706.03762"
            }
        ]
    }
    
    agent = CitationAgent()
    output = agent.run(mock_papers)
    print("IEEE Format:\n", output["ieee"][0])
    print("\nBibTeX Format:\n", output["bibtex"])