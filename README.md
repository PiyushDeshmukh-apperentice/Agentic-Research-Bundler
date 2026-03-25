# 🧬 Agentic Research Bundler

An automated, multi-agent research assistant that orchestrates specialized agents to perform literature surveys, discover datasets, generate citations, and draft technical implementation plans.

## 🧠 How It Works

The system utilizes a **Modular Orchestrator** pattern where a central Streamlit application coordinates four specialized agents.

1.  **Paper Agent**: Queries **arXiv** and **Semantic Scholar**. It merges results, ranks them by citations/recency, and uses **Llama 3.3-70b** to filter for semantic relevance.
2.  **Dataset Agent**: Scours **Kaggle**, **HuggingFace**, and **GitHub** for relevant data sources, applying LLM-based filtering to ensure the datasets match the research goal.
3.  **Citation Agent**: A hybrid agent that uses deterministic logic for **APA/IEEE** formatting and an LLM for valid **BibTeX** generation.
4.  **Planner Agent**: The system "Brain" that synthesizes the papers and datasets into a **4-week implementation roadmap**.

---

## 🛠️ Tech Stack

* **Language**: Python 3.9+
* **LLM Inference**: Groq (Llama-3.3-70b-versatile)
* **Framework**: LangChain
* **UI**: Streamlit
* **APIs**: arXiv, Semantic Scholar, Kaggle, HuggingFace, GitHub

---

## 🚀 Setup Instructions

### 1. Clone & Install
```bash
git clone <your-repo-url>
cd research-bundler
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory. This file stores your sensitive API keys.

```text
# .env
GROQ_API="your_groq_api_key_here"
```

### 3. Setup Kaggle Credentials
Kaggle requires a physical `kaggle.json` file.
1.  Go to [Kaggle.com](https://www.kaggle.com) -> Settings -> **Create New Legacy API Token**.
2.  Create a folder named `credentials` in your project root.
3.  Move your downloaded `kaggle.json` into the `credentials` folder.

---

## 🔑 Obtaining Credentials

| Tool | How to get API Key | Where to store |
| :--- | :--- | :--- |
| **Groq** | Sign up at [Groq Cloud](https://console.groq.com/) and create an API Key. | `.env` file |
| **Kaggle** | Settings > API > Create New Token. | `credentials/kaggle.json` |
| **arXiv** | Public API (No key required). | N/A |
| **Semantic Scholar** | Public API (No key required for basic tier). | N/A |
| **HuggingFace** | Settings > Access Tokens (Optional for search). | `.env` (if used) |
| **GitHub** | Settings > Developer Settings > Personal Access Tokens. | `.env` (Optional) |

---

## 📂 Project Structure

```text
.
├── app.py                # Main Streamlit UI & Orchestrator
├── paper_agent.py        # arXiv & Semantic Scholar logic
├── dataset_agent.py      # Kaggle, HF, & GitHub logic
├── citation_agent.py     # APA, IEEE, & BibTeX logic
├── planner_agent.py      # Implementation Roadmap logic
├── .env                  # API Keys (Git ignored)
├── credentials/          
│   └── kaggle.json       # Kaggle API credentials
└── outputs/              # Stores generated JSON & MD files
```

---

## 🛡️ Resilience & Rate Limiting

The system is designed with **Fail-Safe Orchestration**:
* **Backoff Strategy**: Includes delays between API calls to avoid `HTTP 429` rate limits.
* **Source Fallback**: If arXiv is rate-limited, the system automatically proceeds with Semantic Scholar data.
* **Session Persistence**: Streamlit `session_state` is used to prevent the pipeline from re-running when toggling UI elements.
