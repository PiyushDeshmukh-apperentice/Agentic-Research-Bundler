import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class PlannerAgent:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3, # Slightly higher for creative planning
            api_key=os.getenv("GROQ_API")
        )
        self.parser = StrOutputParser()

    def create_plan(self, query, papers, datasets):
        """
        Input: 
            query: Original research topic
            papers: List from paper_agent
            datasets: List from dataset_agent
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are an expert AI Research Lead and Project Architect. 
Your task is to create a 4-week implementation roadmap based ONLY on the provided research papers and datasets.

STRUCTURE YOUR RESPONSE IN MARKDOWN:
1. Executive Summary: The core objective.
2. Literature-Based Architecture: Suggest a model architecture citing the specific papers provided.
3. Data Strategy: Explain how to use the provided datasets for training/validation.
4. 4-Week Sprint Plan:
   - Week 1: Setup & Preprocessing
   - Week 2: Model Development
   - Week 3: Training & Fine-tuning
   - Week 4: Evaluation & Deployment
5. Potential Pitfalls: Technical challenges specific to this project.

STRICT RULES:
- If a paper mentions a specific model (e.g., YOLOv8, Transformer), use that in the plan.
- If a dataset is from Kaggle, mention the specific Kaggle URL provided.
- Do NOT suggest tools or data that are not in the context.
"""),
            ("human", """
Research Topic: {query}

---
RELEVANT PAPERS:
{paper_context}

---
RELEVANT DATASETS:
{dataset_context}
""")
        ])

        # Prepare context strings
        paper_context = "\n".join([f"- {p['title']} ({p['year']}): {p['summary']}" for p in papers.get("papers", [])])
        dataset_context = "\n".join([f"- {d['name']} ({d['source']}): {d['description']}" for d in datasets.get("datasets", [])])

        chain = prompt | self.llm | self.parser

        try:
            print("🧠 Synthesizing research into an action plan...")
            plan = chain.invoke({
                "query": query,
                "paper_context": paper_context,
                "dataset_context": dataset_context
            })
            
            # Save output
            os.makedirs("outputs", exist_ok=True)
            with open("outputs/planner_agent_output.md", "w") as f:
                f.write(plan)
                
            return plan

        except Exception as e:
            return f"Error generating plan: {str(e)}"

if __name__ == "__main__":
    # Test logic
    agent = PlannerAgent()
    # Dummy data for testing
    p_data = {"papers": [{"title": "Attention is All you Need", "year": 2017, "summary": "Intro to Transformers"}]}
    d_data = {"datasets": [{"name": "IMDB Dataset", "source": "Kaggle", "description": "Movie reviews for sentiment"}]}
    
    print(agent.create_plan("NLP Sentiment Transformer", p_data, d_data))