# Research Bundle — Agentic AI

A small multi-agent research assistant that plans a research workflow, retrieves papers and datasets, and generates an action plan. This repository runs a planner, paper discovery, dataset discovery, and action-plan agent and aggregates results into `outputs/research_bundle.json`.

## Contents

- `supervisor.py` — Orchestrates agents and writes `outputs/research_bundle.json`.
- `agents/` — Agent implementations:
  - `planner_agent.py` — Produces a structured plan (analysis + subtasks).
  - `paper_agent.py` — Fetches papers (arXiv) and summarizes them.
  - `dataset_agent.py` — Searches Kaggle and summarizes datasets.
  - `actionPlan_agent.py` — Produces a neutral step-by-step research plan.
- `app.py` — Streamlit UI to run or view the research bundle.
- `requirements.txt` — Python dependencies (update as needed).
- `outputs/` — Generated outputs; main artifact is `outputs/research_bundle.json`.

## Quick summary of outputs

- `outputs/research_bundle.json` — Final combined bundle containing:
  - `query` — input query string
  - `planner` — planner analysis and subtasks
  - `papers` — paper summaries (prefer `paper_agent_output.json` if available)
  - `datasets` — dataset summaries (prefer `dataset_agent_output.json` if available)
  - `action_plan` — final action plan (prefer `outputs/action_plan_agent_output.json` if available)

Agent-specific files that may appear at repository root or `outputs/`:
- `paper_agent_output.json`
- `dataset_agent_output.json`
- `outputs/action_plan_agent_output.json`
- `planner_agent_output.json`

## Setup

Prerequisites:
- Python 3.10+ (project uses 3.12 in a virtualenv here; any 3.10+ should work)
- Git (optional)
- `kaggle` CLI configured if you want dataset discovery to work locally

Steps:

1. Create a virtual environment and activate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Environment variables and credentials:

- GROQ_API: required by the LLM wrapper `langchain_groq` used in agents. Export it if you have access:

```bash
export GROQ_API="your_groq_api_key"
```

- Kaggle credentials: place your Kaggle API JSON at `~/.kaggle/kaggle.json` (permission 600). The repository includes a `credentials/kaggle.json` sample — either copy it to `~/.kaggle/` or set `KAGGLE_CONFIG_DIR`:

```bash
mkdir -p ~/.kaggle
cp credentials/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

If you prefer, set `KAGGLE_CONFIG_DIR` to point to `credentials/`:

```bash
export KAGGLE_CONFIG_DIR=$(pwd)/credentials
```

4. (Optional) If you do not have `langchain_groq` or an API key, the Streamlit UI can still load and display an existing `outputs/research_bundle.json` created elsewhere.

## Running the pipeline

From the repository root:

```bash
python3 supervisor.py
```

This will:
- Run the planner agent to generate `planner_agent_output.json`.
- Conditionally run the paper and dataset agents according to planner subtasks.
- Conditionally run the action plan agent if papers and datasets are available.
- Write the combined `outputs/research_bundle.json`.

Note: If `langchain_groq` or other LLM dependencies are missing, the run will error early. In that case you can still view or edit `outputs/research_bundle.json` manually.

## Streamlit UI

Start the UI with:

```bash
streamlit run app.py
```

The app allows you to:
- Enter a research query and attempt to run the pipeline from the UI (requires the same environment and credentials as above).
- If running the pipeline from the UI fails, it will fall back to loading `outputs/research_bundle.json` and display the planner analysis, paper summaries, datasets, and action plan.

## File locations and expectations

- `outputs/research_bundle.json` — the canonical combined output. If you want to reuse the outputs between runs, commit or archive this file.
- Agent outputs may be saved at root (e.g., `paper_agent_output.json`, `dataset_agent_output.json`) or under `outputs/` (e.g., `outputs/action_plan_agent_output.json`). `supervisor.py` prefers agent output files when building the bundle.

## Troubleshooting

- Missing packages: install via `pip install -r requirements.txt`. If a specific module still errors (e.g., `langchain_groq`), check that the package name and versions in `requirements.txt` match pip package names on PyPI.
- LLM/API errors: ensure `GROQ_API` (or other keys) are set and valid.
- Kaggle errors: ensure the Kaggle CLI is installed and `kaggle.json` is accessible (`~/.kaggle/kaggle.json`), or set `KAGGLE_CONFIG_DIR` to the local `credentials/` folder.
- Permission errors writing to `outputs/`: ensure you have write permission in the repo directory.

## Development notes

- Planner output format: `planner_agent.py` produces a Pydantic `PlannerOutput` with `analysis` and `subtasks`. `supervisor.py` now derives `agents_invoked` from `planner['subtasks']`.
- If you want agents to always write their outputs to `outputs/`, modify the `with open(...)` paths inside each agent (e.g., change `paper_agent_output.json` to `outputs/paper_agent_output.json`).
- The Streamlit UI (`app.py`) is intentionally simple. You can extend it to show PDFs, download links, or to run longer jobs asynchronously.

## Example workflow

1. Ensure credentials and `GROQ_API` are set.
2. Run `python3 supervisor.py`.
3. Inspect `outputs/research_bundle.json`.
4. Optionally run `streamlit run app.py` to view results in a web UI.

## Contributing

- Fork and open a PR.
- Keep changes minimal and focused to the requested feature or fix.

## License

Add a license if you plan to publish this project.
