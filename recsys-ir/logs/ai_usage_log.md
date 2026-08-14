# AI Usage Log

This is a **living document**.

## Log Format

Each entry should follow this template:

```
### YYYY-MM-DD HH:MM - [Brief title]

- **Tool / Model:** (e.g. Gemini 2.5 Pro, GitHub Copilot, ChatGPT-4o, …)
- **Prompt / Task:** (what you asked or what was auto-suggested)
- **Output Summary:** (what the tool produced)
- **Disposition:** Accepted as-is / Accepted with edits / Rejected
- **Edits Made:** (if accepted with edits, describe what you changed and why)
```

---

### 2026-08-13 15:23 - Repository scaffold generation

- **Tool / Model:** Gemini (Antigravity, Claude Opus 4.6 Thinking)
- **Prompt / Task:** Create the full `recsys-ir/` repository skeleton with directory structure, stub Python files, configs, Makefile, gitignore, README, and this AI usage log.
- **Output Summary:** Generated ~40 files: directory tree with `.gitkeep` placeholders, all Python stubs with module docstrings, `pyproject.toml` with pinned deps, YAML configs, `Makefile` with stub targets, `.gitignore`, `README.md`, notebook stubs, test stubs, and scripts.
- **Disposition:** Accepted as-is.
- **Edits Made:** No edits made.

---

### 2026-08-14 19:53 - Dataset Download & Exploratory Data Analysis (EDA)

- **Tool / Model:** Claude 4.6
- **Prompt / Task:** Download MIND (English, TSV) and EB-NeRD (Danish, Parquet) datasets, perform EDA on both (row counts, schemas, CTR, distributions), create EDA notebooks, and summarize key numbers. Include HF token usage for gated datasets.
- **Output Summary:** Created download scripts (`src/ingestion/download_mind.py`, `download_ebnerd.py`) integrating `python-dotenv` for HF tokens and robust retry logic. Generated Jupyter notebooks (`notebooks/00_explore_mind.ipynb`, `notebooks/00_explore_ebnerd.ipynb`) using programmatic EDA generation via python script. Created comprehensive `EDA_SUMMARY.md` tracking sizes, temporal splits, CTR (MIND ~4%, EB-NeRD ~8.5%), missingness, and verified Danish unicode requirements.
- **Disposition:** Accepted with edits
- **Edits Made:** User prompted to use `.env` file to store HF tokens instead of direct export, prompted to update `Makefile` to include specific `make download` targets, and instructed the model to retry/continue download processes that timed out due to unstable connections.
