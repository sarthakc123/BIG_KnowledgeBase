import os
from pathlib import Path
from typing import Optional, List
import base64  # 🔹 for inline PDF render

import streamlit as st
import pandas as pd
from pypdf import PdfReader

from sentence_transformers import SentenceTransformer

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from datetime import datetime

# =========================
# TEMP: API key here for local testing.
# For real use, set via env or Streamlit secrets.
import os
import streamlit as st

# Prefer Streamlit Cloud Secrets
if "ANTHROPIC_API_KEY" in st.secrets:
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]

# Stop early if still missing
if not os.environ.get("ANTHROPIC_API_KEY"):
    st.error("ANTHROPIC_API_KEY is not set. Add it in Streamlit Secrets.")
    st.stop()
# =========================

# =========================
# Config & paths
# =========================

DATA_ROOT = Path("Charlie Output") / "Salud_main_1"
VECTOR_DB_DIR = Path("chroma_store")


# =========================
# Embedding wrapper
# =========================

class STEmbedding(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()


@st.cache_resource(show_spinner=False)
def get_vectordb():
    embeddings = STEmbedding()
    vectordb = Chroma(
        collection_name="payer_policies",
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DB_DIR),
    )
    return vectordb


# =========================
# IngestionAgent
# =========================

class IngestionAgent:
    def __init__(self, data_root: Path, vectordb: Chroma):
        self.data_root = data_root
        self.vectordb = vectordb
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " "],
        )

    def _build_policy_table(self, run_date: str) -> pd.DataFrame:
        run_dir = self.data_root / run_date
        if not run_dir.exists():
            raise FileNotFoundError(f"Run folder not found: {run_dir}")

        rows = []

        for payer_dir in run_dir.iterdir():
            if not payer_dir.is_dir():
                continue

            payer_id = payer_dir.name.lower()
            payer_name = payer_id.upper()

            for pdf_path in payer_dir.glob("*.pdf"):
                try:
                    reader = PdfReader(str(pdf_path))
                except Exception as e:
                    st.warning(f"⚠️ Failed to read {pdf_path}: {e}")
                    continue

                for page_idx, page in enumerate(reader.pages, start=1):
                    try:
                        text = page.extract_text() or ""
                    except Exception as e:
                        st.warning(
                            f"⚠️ Failed to extract text from {pdf_path} p{page_idx}: {e}"
                        )
                        text = ""

                    rows.append(
                        {
                            "run_date": run_date,
                            "payer_id": payer_id,
                            "payer_name": payer_name,
                            "state": None,
                            "policy_file": pdf_path.name,
                            "policy_path": str(pdf_path),
                            "page_number": page_idx,
                            "content": text,
                            "ingested_at": datetime.utcnow().isoformat(),
                        }
                    )

        df = pd.DataFrame(rows)
        out_path = run_dir / "policies.parquet"
        df.to_parquet(out_path, index=False)
        st.success(f"✅ Saved {len(df)} rows to {out_path}")
        return df

    def _chunk_and_index(self, df: pd.DataFrame):
        docs_to_add: list[Document] = []

        for _, row in df.iterrows():
            chunks = self.splitter.split_text(row["content"])
            for i, chunk in enumerate(chunks):
                docs_to_add.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "run_date": row["run_date"],
                            "payer_id": row["payer_id"],
                            "payer_name": row["payer_name"],
                            "state": row["state"],
                            "policy_file": row["policy_file"],
                            "policy_path": row["policy_path"],
                            "page_number": row["page_number"],
                            "chunk_index": i,
                        },
                    )
                )

        if not docs_to_add:
            st.warning("⚠️ No rows to index.")
            return

        BATCH_SIZE = 2000
        total = len(docs_to_add)
        for start in range(0, total, BATCH_SIZE):
            end = start + BATCH_SIZE
            batch = docs_to_add[start:end]
            self.vectordb.add_documents(batch)
            st.write(f"➕ Indexed batch {start}–{min(end, total)} (size {len(batch)})")

        self.vectordb.persist()
        st.success(
            f"✅ Indexed {total} chunks for run_date={df['run_date'].iloc[0]}"
        )

    def run_batch(self, run_date: str) -> pd.DataFrame:
        st.info(f"🚀 Starting ingestion for run_date={run_date}")
        df = self._build_policy_table(run_date)
        self._chunk_and_index(df)
        st.success("🏁 Ingestion pipeline finished.")
        return df


# =========================
# AnswerAgent (with chat history + guardrails)
# =========================

class AnswerAgent:
    """
    Uses the vector store and Anthropic Claude to answer questions
    about payer policies, with citations.

    Supports two modes:
    - "strict": policy-grounded only (RAG only, no general knowledge)
    - "hybrid": use policy context when available, otherwise safe general knowledge
    """

    def __init__(
        self,
        vectordb: Chroma,
        model_name: str = "claude-sonnet-4-5",
        temperature: float = 0.2,
        max_tokens: int = 1000,
    ):
        self.vectordb = vectordb
        self.llm = ChatAnthropic(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # -------- per-run payer summary (current run only) --------
    def summarize_payer_changes(
        self,
        payer_id: str,
        run_date: str,
        k: int = 12,
    ) -> str:
        """
        Retrieve and summarize 'change' signals for a single payer for one run_date.
        """

        change_terms = [
            "updated", "revised", "effective", "change",
            "changes", "modification", "modified", "update",
            "replaces", "new requirement",
        ]

        # Multi-query retrieval
        all_docs = []
        for term in change_terms:
            docs = self.vectordb.similarity_search(
                term,
                k=k,
                filter={
                    "$and": [
                        {"payer_id": payer_id.lower()},
                        {"run_date": run_date},
                    ]
                },
            )
            all_docs.extend(docs)

        # Deduplicate by (file, page)
        seen = set()
        unique_docs = []
        for d in all_docs:
            key = (d.metadata.get("policy_file"), d.metadata.get("page_number"))
            if key not in seen:
                seen.add(key)
                unique_docs.append(d)

        if not unique_docs:
            return f"No explicit changes mentioned for **{payer_id}**."

        # Build context
        context_chunks = []
        for i, d in enumerate(unique_docs, start=1):
            m = d.metadata
            context_chunks.append(
                f"[{i}] payer={m.get('payer_id')}, file={m.get('policy_file')}, page={m.get('page_number')}\n"
                f"{d.page_content}\n"
            )
        context_text = "\n\n".join(context_chunks)

        prompt = f"""
You are a healthcare policy expert summarizing only explicit **changes** to payer rules.

Using the extracted policy text below, summarize the most important changes.
Focus on high-level items such as:
- policy revisions
- new requirements
- replaced rules
- effective date changes
- documentation updates
- claim submission / prior auth updates

Provide:
1) A 3–5 bullet list of changes.
2) Keep them concise and action-oriented.
3) Only summarize statements supported by the context.

Context:
{context_text}

Summary:
"""

        resp = self.llm.invoke(prompt)
        return resp.content

    # -------- diff vs previous run --------
    def summarize_payer_changes_vs_previous(
        self,
        payer_id: str,
        current_run_date: str,
        previous_run_date: str,
        k: int = 10,
    ) -> str:
        """
        Compare 'change-like' policy text for a payer between two run_dates
        and summarize differences.
        """

        change_terms = [
            "updated", "revised", "effective", "change",
            "changes", "modification", "modified", "update",
            "replaces", "new requirement",
        ]

        def retrieve_for_run(run_date: str):
            docs_all = []
            for term in change_terms:
                docs = self.vectordb.similarity_search(
                    term,
                    k=k,
                    filter={
                        "$and": [
                            {"payer_id": payer_id.lower()},
                            {"run_date": run_date},
                        ]
                    },
                )
                docs_all.extend(docs)

            # Deduplicate by (file, page)
            seen = set()
            unique = []
            for d in docs_all:
                key = (d.metadata.get("policy_file"), d.metadata.get("page_number"))
                if key not in seen:
                    seen.add(key)
                    unique.append(d)
            return unique

        current_docs = retrieve_for_run(current_run_date)
        previous_docs = retrieve_for_run(previous_run_date)

        if not current_docs and not previous_docs:
            return (
                f"For **{payer_id.upper()}**, I couldn't find any explicit change-like "
                f"language in either {previous_run_date} or {current_run_date}."
            )

        def build_context(docs, label: str):
            chunks = []
            for i, d in enumerate(docs, start=1):
                m = d.metadata
                chunks.append(
                    f"[{label} {i}] payer={m.get('payer_id')}, run_date={m.get('run_date')}, "
                    f"file={m.get('policy_file')}, page={m.get('page_number')}\n"
                    f"{d.page_content}\n"
                )
            return "\n\n".join(chunks) if chunks else "(no extracted text)"

        current_context = build_context(current_docs, "NEW")
        previous_context = build_context(previous_docs, "OLD")

        prompt = f"""
You are a cautious healthcare policy expert comparing two snapshots of payer policies
for the same payer: an older run and a newer run.

Guardrails:
- Only describe differences that are supported by the text below.
- Do NOT invent effective dates, numeric limits, or reasons for the changes.
- Do NOT assume why a change was made, only what appears to have changed.
- Focus on CLAIMS-RELATED changes and other clearly policy-impacting differences:
  - new requirements, removed requirements
  - changed wording that alters scope, documentation, or timing
  - changes to claim submission, processing, prior auth, or appeals processes.
- If you are not sure whether something truly changed, flag it as "possible change"
  and explain why.

Older policy excerpts (run_date={previous_run_date}):
{previous_context}

Newer policy excerpts (run_date={current_run_date}):
{current_context}

Provide your answer in three parts:

1) High-level summary (3–5 bullets):
   - What clearly changed for this payer between the two runs?
   - Group them into: "New rules", "Removed/relaxed rules", "Changed/clarified wording".

2) Detailed differences (bullets):
   - For each difference, quote or closely paraphrase the relevant OLD vs NEW text.
   - Make it clear which run_date each side comes from.
   - If you are not sure something changed (e.g., text is too partial), mark it as "possible".

3) Caveats:
   - Brief note on limitations of this comparison (partial context, need to check full PDFs).
"""
        resp = self.llm.invoke(prompt)
        return resp.content

    # -------- main QA --------
    def answer(
            self,
            question: str,
            payer_id: Optional[str] = None,
            run_date: Optional[str] = None,
            k: int = 6,
            debug: bool = False,
            mode: str = "hybrid",  # "strict" or "hybrid"
            history: Optional[List[dict]] = None,  # chat history for context
    ) -> dict:
        """
        Ask a question about payer policies.

        Returns:
            {
                "text": <answer string or debug text>,
                "sources": [
                    {
                        "payer_id": ...,
                        "run_date": ...,
                        "policy_file": ...,
                        "policy_path": ...,
                        "page_number": ...,
                        "index": i,  # index used in [i] citations
                    },
                    ...
                ]
            }
        """

        # Build conversation history text (last few turns)
        history_text = ""
        if history:
            trimmed = history[-6:]  # last 6 messages
            parts = []
            for msg in trimmed:
                role = "User" if msg["role"] == "user" else "Assistant"
                parts.append(f"{role}: {msg['content']}")
            history_text = "\n".join(parts)

        # -----------------------------
        # 1. Build metadata filter
        # -----------------------------
        filter_dict = {}
        if payer_id:
            filter_dict["payer_id"] = payer_id.lower()
        if run_date:
            filter_dict["run_date"] = run_date

        if filter_dict:
            if len(filter_dict) == 1:
                # Single field → plain filter, e.g. {"payer_id": "bcbsil"}
                metadata_filter = filter_dict
            else:
                # Multiple fields → $and of individual where expressions
                metadata_filter = {
                    "$and": [{k: v} for k, v in filter_dict.items()]
                }
        else:
            metadata_filter = None

        # -----------------------------
        # 2. Retrieve context
        # -----------------------------
        docs = self.vectordb.similarity_search(
            question,
            k=k,
            filter=metadata_filter,
        )

        # Prepare sources list from retrieved docs
        sources: List[dict] = []
        for i, d in enumerate(docs, start=1):
            m = d.metadata
            sources.append(
                {
                    "index": i,
                    "payer_id": m.get("payer_id"),
                    "run_date": m.get("run_date"),
                    "policy_file": m.get("policy_file"),
                    "policy_path": m.get("policy_path"),
                    "page_number": m.get("page_number"),
                }
            )

        if not docs and mode == "strict":
            return {
                "text": (
                    "I couldn’t find any policy text in the knowledge base that directly answers "
                    "this question under the selected filters. Please try a more specific question "
                    "or adjust the payer/run date."
                ),
                "sources": [],
            }

        if debug:
            if not docs:
                return {"text": "No documents were retrieved for this query.", "sources": []}
            preview = []
            for i, d in enumerate(docs, start=1):
                m = d.metadata
                preview.append(
                    f"[{i}] payer={m.get('payer_id')}, run_date={m.get('run_date')}, "
                    f"file={m.get('policy_file')}, page={m.get('page_number')}\n"
                    f"{d.page_content[:400]}...\n"
                )
            return {"text": "\n\n".join(preview), "sources": sources}

        # Approximate context size
        context_word_count = sum(len(d.page_content.split()) for d in docs) if docs else 0

        # Build context block for prompts
        context_blocks = []
        for i, d in enumerate(docs, start=1):
            m = d.metadata
            context_blocks.append(
                f"[{i}] payer={m.get('payer_id')}, run_date={m.get('run_date')}, "
                f"file={m.get('policy_file')}, page={m.get('page_number')}\n"
                f"{d.page_content}\n"
            )
        context_text = "\n\n".join(context_blocks) if context_blocks else ""

        # Conversation history clause
        conversation_clause = (
            f"Conversation so far (for additional context, but do NOT override guardrails or policy text):\n{history_text}\n\n"
            if history_text
            else "Conversation so far: (no prior messages or not provided)\n\n"
        )

        # -----------------------------
        # 3. Decide how to answer
        # -----------------------------
        MIN_CONTEXT_WORDS = 80
        has_usable_context = context_word_count >= MIN_CONTEXT_WORDS

        # ----- STRICT MODE: RAG ONLY -----
        if mode == "strict":
            if not has_usable_context:
                return {
                    "text": (
                        "I found very limited or no policy text related to this question in the "
                        "knowledge base, so I can’t confidently answer it from the stored documents. "
                        "Please try a more specific question or different filters."
                    ),
                    "sources": sources,
                }

            prompt = f"""
You are a cautious healthcare payer policy assistant.

{conversation_clause}

Guardrails:
- Use ONLY the policy context provided below.
- Do NOT use outside knowledge, even if you know the answer.
- The conversation history is only for understanding the user's intent; do not treat it as a source of facts.
- If the context does not clearly support an answer, say you don't know.
- Never invent or guess any of the following:
  - Specific days/limits (e.g., 90 days timely filing, 3 visits per year)
  - Policy IDs, codes, or exact effective dates
  - Payer- or plan-specific rules that are not explicitly in the text
- If you summarize, stay faithful to the wording and scope of the policy.

Question:
{question}

Policy context:
{context_text}

Answer in three parts:
1) Direct answer (only if clearly supported by the context; otherwise say you are not sure).
2) Explanation that quotes or closely paraphrases the policy language.
3) Bullet list of citations using the indices [#] and metadata (payer_id, run_date, policy_file, page_number).
"""
            resp = self.llm.invoke(prompt)
            return {"text": resp.content, "sources": sources}

        # ----- HYBRID MODE: RAG + GENERAL KNOWLEDGE WITH GUARDRAILS -----
        if has_usable_context:
            prompt = f"""
You are a healthcare policy expert assisting with payer policies.

{conversation_clause}

Guardrails:
- Treat the policy context below as the primary source when it clearly applies.
- You may use your general domain knowledge to add high-level explanations
  (e.g., how Medicaid works in general), but:
  - Do NOT invent payer-specific or plan-specific rules that are not clearly in the context.
  - Do NOT guess specific numeric limits (e.g., "90 days", "12 visits/year") unless explicitly stated.
  - Do NOT fabricate policy IDs, codes, or effective dates.
- If you must generalize beyond the text, make it clear that these are general principles
  and that specific details vary by state, payer, and plan.
- The conversation history is only for understanding the user's intent and follow-ups,
  not a source of authoritative facts.
- If the context conflicts with your general knowledge, prioritize the context.

Question:
{question}

Policy context (may or may not fully answer the question):
{context_text}

Provide your answer in 3 parts:
1) Direct answer:
   - Use the policy context when it clearly applies.
   - If you also use general knowledge, state explicitly that it is general and may vary.
2) Explanation:
   - Reference or quote key policy language when relevant.
   - Clearly separate what comes from the context vs. general knowledge.
3) Citations:
   - Bullet list of citations for any statements based on the context, using indices [#]
     and metadata (payer_id, run_date, policy_file, page_number).
   - If parts of your answer come only from general knowledge, say "general knowledge, no citation".
"""
            resp = self.llm.invoke(prompt)
            return {"text": resp.content, "sources": sources}

        # Case B: no useful context → safe general knowledge answer with strong guardrails
        general_prompt = f"""
You are a cautious healthcare policy explainer.

{conversation_clause}

The user asked:
{question}

There is no reliable policy text available from the knowledge base for this query.

Guardrails for your answer:
- Answer using general US healthcare / Medicaid knowledge only.
- Do NOT mention or speculate about any specific payer, plan, product, or policy ID.
- Do NOT guess specific numeric limits (e.g., "90 days timely filing", "3 visits/year").
- Speak at a high level (e.g., typical provider types, roles, general structures).
- Always acknowledge that exact rules vary by state, payer, and plan and must be verified
  against official policy documents.
- If the question clearly requires exact, policy-specific numbers or rules, say that those
  require checking the actual policy text.

Provide your answer in 2 parts:
1) General explanation that directly addresses the question in a high-level, non-specific way.
2) A short disclaimer reminding the user that details vary by state and payer and should be
   confirmed in the actual Medicaid or payer policy documentation.
"""
        resp = self.llm.invoke(general_prompt)
        return {"text": resp.content, "sources": sources}


# =========================
# Helpers
# =========================
@st.cache_data
def get_run_dates(data_root: Path) -> list[str]:
    if not data_root.exists():
        return []
    return sorted(
        [p.name for p in data_root.iterdir() if p.is_dir()]
    )

@st.cache_data
def get_payers_for_run(data_root: Path, run_date: str) -> list[str]:
    run_dir = data_root / run_date
    if not run_dir.exists():
        return []
    return sorted([p.name for p in run_dir.iterdir() if p.is_dir()])


def get_previous_run_date(current: str, all_run_dates: list[str]) -> Optional[str]:
    """Return the immediate previous run_date before `current`, or None if not found."""
    sorted_runs = sorted(all_run_dates)
    if current not in sorted_runs:
        return None
    idx = sorted_runs.index(current)
    if idx == 0:
        return None
    return sorted_runs[idx - 1]

from streamlit.components.v1 import html
import base64

from pypdf import PdfReader
import streamlit as st

def render_pdf_inline(path: str, page: int | None = None, height: int = 500):
    """
    'Inline preview' for a policy PDF: show the *text* of the cited page.

    This avoids browser PDF embedding issues and still lets the user
    quickly see what the chatbot is citing.
    """
    try:
        reader = PdfReader(path)
        total_pages = len(reader.pages)
        if total_pages == 0:
            st.warning("This PDF has no pages or could not be read.")
            return

        # Clamp page index safely
        page_idx = (page or 1) - 1
        page_idx = max(0, min(total_pages - 1, page_idx))

        p = reader.pages[page_idx]
        text = p.extract_text() or "(No extractable text on this page – it might be a scanned image.)"

        st.markdown(f"**Page {page_idx + 1} of {total_pages}**")
        st.text_area(
            "Page text preview",
            value=text,
            height=height,
        )
    except Exception as e:
        st.warning(f"Could not preview PDF page text: {e}")



# =========================
# Streamlit UI (chat style)
# =========================

def main():
    st.set_page_config(page_title="Salud Knowledge Base Agent", layout="wide")

    st.title("🩺 Salud Knowledge Base Agent")
    st.caption("Chat over payer PDFs with Anthropic + local embeddings")

    api_key_set = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not api_key_set:
        st.error(
            "ANTHROPIC_API_KEY is not set. Please set it in your environment or via Streamlit secrets."
        )
        st.stop()

    vectordb = get_vectordb()
    ingestion_agent = IngestionAgent(DATA_ROOT, vectordb)

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")

        run_dates = get_run_dates(DATA_ROOT)
        if not run_dates:
            st.error(f"No run_date folders found under: {DATA_ROOT}")
            st.stop()

        selected_run = st.selectbox("Run date", run_dates, index=len(run_dates) - 1)

        payers = get_payers_for_run(DATA_ROOT, selected_run)
        payer_choice = st.selectbox(
            "Payer (folder name)", ["(all)"] + payers, index=0
        )
        payer_id = None if payer_choice == "(all)" else payer_choice

        # Mode toggle
        mode_label = st.radio(
            "Answer mode",
            ["Hybrid (context + general knowledge)", "Strict (policy-grounded only)"],
            index=0,
        )
        mode = "hybrid" if mode_label.startswith("Hybrid") else "strict"

        st.markdown("---")
        st.subheader("Ingestion")
        if st.button("Run ingestion for selected run_date"):
            with st.spinner("Running ingestion pipeline..."):
                ingestion_agent.run_batch(selected_run)

        st.markdown("---")
        k = st.slider("Top-k chunks", min_value=3, max_value=12, value=6)
        debug_mode = st.checkbox("Debug mode (show retrieved chunks only)", value=False)

        st.markdown("---")
        if st.button("🧹 Clear chat history"):
            st.session_state.pop("messages", None)
            st.rerun()

    # AnswerAgent after sidebar (uses mode)
    answer_agent = AnswerAgent(vectordb)

    # Init chat + summary state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "payer_summaries" not in st.session_state:
        # key: (run_date, payer_id) -> summary_text
        st.session_state.payer_summaries = {}

    if "payer_diffs" not in st.session_state:
        # key: (prev_run, current_run, payer_id) -> diff_text
        st.session_state.payer_diffs = {}

    # ======== Provider-level Changes Summary (compute once & cache) ========
    st.subheader("📊 Key Changes by Payer")

    # Button to explicitly trigger summary generation
    generate_summaries = st.button(
        "⚡ Generate / refresh payer summaries for this run_date",
        key="generate_summaries",
    )

    if generate_summaries:
        with st.spinner("Summarizing changes for each payer..."):
            for payer in payers:
                key = (selected_run, payer)
                summary = answer_agent.summarize_payer_changes(
                    payer_id=payer,
                    run_date=selected_run,
                )
                st.session_state.payer_summaries[key] = summary

        st.session_state["summaries_last_run_date"] = selected_run

    prev_run = get_previous_run_date(selected_run, run_dates)

    # Display summaries
    for payer in payers:
        key = (selected_run, payer)
        summary = st.session_state.payer_summaries.get(key)

        label = f"🔹 {payer.upper()} — Key Changes"
        with st.expander(label):
            # ---- key changes summary ----
            if summary:
                st.markdown(summary)
            else:
                st.info(
                    "No summary generated yet for this payer. "
                    "Click the button above to create one."
                )

            # ---- Diff vs previous run_date ----
            if prev_run is None:
                st.caption("No previous run_date available for comparison.")
            else:
                diff_key = (prev_run, selected_run, payer)
                if diff_key in st.session_state.payer_diffs:
                    with st.expander(
                            f"🧾 Changes vs previous run ({prev_run} → {selected_run})"
                    ):
                        st.markdown(st.session_state.payer_diffs[diff_key])
                else:
                    if st.button(
                            f"🔍 Compare to previous run ({prev_run})",
                            key=f"diff_btn_{payer}",
                    ):
                        with st.spinner(
                                f"Computing differences for {payer.upper()} "
                                f"({prev_run} → {selected_run})..."
                        ):
                            diff_text = answer_agent.summarize_payer_changes_vs_previous(
                                payer_id=payer,
                                current_run_date=selected_run,
                                previous_run_date=prev_run,
                            )
                            st.session_state.payer_diffs[diff_key] = diff_text

                        with st.expander(
                                f"🧾 Changes vs previous run ({prev_run} → {selected_run})"
                        ):
                            st.markdown(diff_text)


    st.subheader("💬 Chat with the agent")

    # Render past messages
    for msg in st.session_state.messages:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input(
        "Ask a question about payer policies, Medicaid, prior auth, etc..."
    )

    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking with Claude..."):
                result = answer_agent.answer(
                    question=user_input,
                    payer_id=payer_id,
                    run_date=selected_run,
                    k=k,
                    debug=debug_mode,
                    mode=mode,
                    history=st.session_state.messages,
                )
                answer_text = result["text"]
                sources = result.get("sources", [])

                if debug_mode:
                    st.subheader("Retrieved chunks preview")
                    st.code(answer_text)
                else:
                    st.markdown(answer_text)

                    # 🔗 Source PDFs for this specific answer
                    if sources:
                        st.markdown("**📎 Source PDFs referenced in this answer**")

                        seen_paths = set()
                        for src in sources:
                            policy_path = src.get("policy_path")
                            policy_file = src.get("policy_file")
                            payer_src = src.get("payer_id")
                            page_num = src.get("page_number")

                            if not policy_path or policy_path in seen_paths:
                                continue

                            seen_paths.add(policy_path)
                            file_path = Path(policy_path)

                            if not file_path.exists():
                                st.caption(
                                    f"Source file not found: `{policy_file}` "
                                    f"(payer={payer_src}, page={page_num})"
                                )
                                continue

                            # Download button
                            try:
                                with open(file_path, "rb") as f:
                                    st.download_button(
                                        label=f"⬇️ {policy_file} "
                                              f"(payer={payer_src}, page {page_num})",
                                        data=f,
                                        file_name=policy_file,
                                        mime="application/pdf",
                                        key=f"dl_answer_{payer_src}_{policy_file}_{page_num}",
                                    )
                            except Exception as e:
                                st.caption(
                                    f"Could not open `{policy_file}` for download: {e}"
                                )

                            with st.expander(
                                    f"👀 Preview {policy_file} (payer={payer_src}, page {page_num})",
                                    expanded=False,
                            ):
                                render_pdf_inline(str(file_path), page=page_num or None, height=800)

        # Store assistant message (just the text in history)
        st.session_state.messages.append({"role": "assistant", "content": answer_text})


if __name__ == "__main__":
    main()
