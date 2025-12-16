# 🩺 Salud Knowledge Base Agent

**Salud Knowledge Base Agent** is an AI-powered Streamlit application for ingesting, indexing, and querying **payer policy PDFs** (e.g., Medicaid and commercial payer policies).  
It combines **local embeddings**, **vector search**, and **Anthropic Claude** to deliver **grounded, citation-backed answers** and **policy change intelligence**.

---

## Key Features

### Intelligent Policy Ingestion
- Organizes policies by `run_date` and `payer`
- Extracts text page-by-page from PDFs
- Stores structured policy records with full metadata (file, page, run date)

### Semantic Search & RAG
- Sentence-Transformers (MiniLM) embeddings
- Chroma vector database for fast similarity search
- Metadata-aware filtering (payer, run_date)

### Policy-Aware AI Answers
- Powered by **Anthropic Claude**
- Two answer modes:
  - **Strict Mode** – answers only if explicitly supported by policy text
  - **Hybrid Mode** – policy-grounded answers with clearly labeled general context
- Inline citations to exact PDF pages

### Change Intelligence
- Per-payer “Key Changes” summaries for each run
- Diff against previous runs with strict guardrails
- Focus on actionable policy updates (no hallucinations)

### 🧾 Traceability & Auditing
- Download original source PDFs
- Inline preview of cited PDF page text
- End-to-end transparency from answer → source

---

## Project Structure

├── BigAgentFinal_Streamlit.py # Main Streamlit app
├── requirements.txt # Python dependencies
├── README.md
├── .streamlit/
│ └── config.toml # Salud green UI theme
└── Charlie Output/ # Local data (ignored in git)
└── Salud_main_1/
└── <run_date>/
└── <payer_id>/
└── *.pdf

## Running Locally

streamlit run BigAgentFinal_Streamlit.py
Deploying on Streamlit Community Cloud
Main file path: BigAgentFinal_Streamlit.py
Add secret in Manage app → Settings → Secrets:
ANTHROPIC_API_KEY = "sk-ant-..." <-- YOUR API KEY HERE
Deploy 


## Answer Modes

Strict Mode
  Uses only retrieved policy text
  No outside knowledge

Hybrid Mode
  Prioritizes policy context
  Adds clearly labeled general healthcare knowledge
  Never invents payer-specific rules

## Security Notes
No API keys stored in code
Secrets managed via Streamlit Secrets
Intended for policy analysis (no PHI/PII assumptions)

## Author
Sarthak Chandarana
Project: Salud Knowledge Base Agent
