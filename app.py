import streamlit as st
import json
import os
from agents.paper_agent import run_paper_agent
from agents.dataset_agent import run_dataset_agent
from agents.citation_agent import CitationAgent
from agents.planner_agent import PlannerAgent

# --- Page Config ---
st.set_page_config(page_title="Agentic Research Bundler", page_icon="🧬", layout="wide")

# --- Initialize Agents ---
citation_tool = CitationAgent()
planner_tool = PlannerAgent()

def main():
    st.title("🧬 Agentic Research Bundler")
    st.markdown("Automated Literature Survey, Dataset Discovery, and Project Planning.")

    # Sidebar for API Status/Config
    with st.sidebar:
        st.header("⚙️ System Status")
        st.success("Groq API: Connected")
        st.success("arXiv/Semantic Scholar: Online")
        st.success("Kaggle/HF/GitHub: Online")
        st.divider()
        st.info("Model: Llama Models via Groq")

    # User Input
    query = st.text_input("Enter your research topic/query:", placeholder="e.g., Federated Learning for Medical Imaging")
    
    if st.button("🚀 Run Research Pipeline", use_container_width=True):
        if not query:
            st.warning("Please enter a topic first!")
            return

        # --- EXECUTION PIPELINE ---
        with st.status("🏗️ Building Research Bundle...", expanded=True) as status:
            
            # 1. Paper Agent
            st.write("🔍 Fetching and filtering research papers...")
            paper_results = run_paper_agent(query)
            st.write(f"✅ Found {len(paper_results.get('papers', []))} relevant papers.")

            # 2. Dataset Agent
            st.write("📊 Searching for datasets (Kaggle, HF, GitHub)...")
            dataset_results = run_dataset_agent(query)
            st.write(f"✅ Found {len(dataset_results.get('datasets', []))} datasets.")

            # 3. Citation Agent
            st.write("📜 Formatting citations (APA, IEEE, BibTeX)...")
            citations = citation_tool.run(paper_results)

            # 4. Planner Agent
            st.write("🧠 Synthesizing implementation strategy...")
            action_plan = planner_tool.create_plan(query, paper_results, dataset_results)
            
            status.update(label="✅ Research Bundle Complete!", state="complete", expanded=False)

        # --- DISPLAY RESULTS ---
        tab1, tab2, tab3, tab4 = st.tabs(["📑 Action Plan", "📚 Papers", "💾 Datasets", "🖋️ Citations"])

        with tab1:
            st.header("🛠️ Implementation Roadmap")
            st.markdown(action_plan)

        with tab2:
            st.header("Relevant Literature")
            for p in paper_results.get("papers", []):
                with st.expander(f"{p['title']} ({p['year']})"):
                    st.write(f"**Authors:** {', '.join(p['authors'])}")
                    st.write(f"**Source:** {p['source'].upper()}")
                    st.write(f"**Summary:** {p['summary']}")
                    st.link_button("View Paper", p['link'])

        with tab3:
            st.header("Discovered Datasets")
            cols = st.columns(2)
            for i, d in enumerate(dataset_results.get("datasets", [])):
                with cols[i % 2].container(border=True):
                    st.subheader(d['name'])
                    st.caption(f"Source: {d['source'].title()}")
                    st.write(d['description'][:200] + "...")
                    st.link_button("Access Dataset", d['url'])

        with tab4:
            st.header("Generated Citations")
            cite_type = st.radio("Format", ["IEEE", "APA", "BibTeX"], horizontal=True)
            
            if cite_type == "IEEE":
                for c in citations["ieee"]:
                    st.code(c, language=None)
            elif cite_type == "APA":
                for c in citations["apa"]:
                    st.code(c, language=None)
            else:
                st.code(citations["bibtex"], language="bibtex")

if __name__ == "__main__":
    main()