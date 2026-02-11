import json
import os
import streamlit as st

st.set_page_config(page_title="Research Assistant", layout="wide")

st.title("Research Bundle Explorer")

query = st.text_input("Research query", value="Deep learning methods for landslide detection using satellite imagery")

run_button = st.button("Run research pipeline")

output_area = st.empty()


def load_bundle_from_file(path="outputs/research_bundle.json"):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            return {"error": str(e)}
    return None


if run_button:
    output_area.info("Running research pipeline. This may take a while...")
    try:
        from supervisor import run_research_system

        results = run_research_system(query)
    except Exception as e:
        st.warning(f"Could not run full pipeline here: {e}")
        results = load_bundle_from_file() or {"error": "No local bundle available"}

    if results is None:
        st.error("No results produced.")
    else:
        output_area.success("Research bundle obtained — displaying below")

        # Planner
        if results.get("planner"):
            st.subheader("Planner Analysis")
            planner = results["planner"]
            if isinstance(planner, dict) and planner.get("analysis"):
                analysis = planner["analysis"]
                cols = st.columns(3)
                with cols[0]:
                    st.markdown("**Research domain**")
                    st.write(analysis.get("research_domain"))
                    st.markdown("**Sub-domain**")
                    st.write(analysis.get("sub_domain"))
                with cols[1]:
                    st.markdown("**Problem type**")
                    st.write(analysis.get("problem_type"))
                    st.markdown("**Data modality**")
                    st.write(analysis.get("data_modality"))
                with cols[2]:
                    st.markdown("**Key techniques**")
                    st.write(analysis.get("key_techniques"))
                    st.markdown("**Expected outputs**")
                    st.write(analysis.get("expected_outputs"))

            if isinstance(planner, dict) and planner.get("subtasks"):
                st.markdown("**Subtasks**")
                for s in planner.get("subtasks", []):
                    st.write(f"- **{s.get('agent')}**: {s.get('goal')} — {s.get('rationale')}")

        # Papers
        if results.get("papers"):
            st.subheader("Papers")
            papers_obj = results["papers"]
            papers = papers_obj.get("papers") if isinstance(papers_obj, dict) else None
            if not papers:
                st.write(papers_obj)
            else:
                for p in papers:
                    st.markdown(f"**{p.get('title')}** — {p.get('year')}")
                    st.write(p.get("methodology"))
                    st.write("**Data used:** " + str(p.get("data_used")))
                    st.write("**Key contribution:** " + str(p.get("key_contribution")))
                    st.write("---")

        # Datasets
        if results.get("datasets"):
            st.subheader("Datasets")
            datasets_obj = results["datasets"]
            datasets = datasets_obj.get("datasets") if isinstance(datasets_obj, dict) else None
            if not datasets:
                st.write(datasets_obj)
            else:
                for d in datasets:
                    st.markdown(f"**{d.get('name')}** — {d.get('source', 'unknown')}")
                    st.write("Task: " + str(d.get("task_type")))
                    st.write("Data type: " + str(d.get("data_type")))
                    st.write("Labels: " + str(d.get("labels")))
                    if d.get("url"):
                        st.write(d.get("url"))
                    st.write("---")

        # Action plan
        if results.get("action_plan"):
            st.subheader("Action Plan")
            ap = results["action_plan"]
            # If nested under 'research_plan'
            if isinstance(ap, dict) and ap.get("research_plan"):
                rp = ap["research_plan"]
                st.markdown(f"**{rp.get('title', 'Research Plan')}**")
                st.write(rp.get("objective"))
                for step in rp.get("methodology", []):
                    st.markdown(f"- **{step.get('step')}**: {step.get('description')}")
                    if step.get("papers"):
                        st.write("  - Papers:")
                        for p in step.get("papers"):
                            st.write(f"    - {p.get('title')} ({p.get('year')})")
                    if step.get("datasets"):
                        st.write("  - Datasets:")
                        for d in step.get("datasets"):
                            st.write(f"    - {d.get('name')}")
            else:
                st.write(ap)

else:
    st.info("Enter a query and click 'Run research pipeline' — or open an existing `outputs/research_bundle.json` by running the pipeline separately.")
