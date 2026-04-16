# Insurance Industry Foresight Research System

**Current version: v0.2 — Search-Grounded Venture Scout**

A research automation tool built for a foresight collaboration between
**IEX Labs**, **IE University**, and **VIG Insurance Group**.

---

## New in v0.2

| # | Upgrade | Impact |
|---|---|---|
| 1 | **Real web search** (Tavily) | Scout finds ventures from live search results, not just LLM memory. Dramatically reduces hallucination. |
| 2 | **Three-layer verification** | Domain sanity check → HTTP check → content matching. Rejects aggregator URLs and wrong-domain results. |
| 3 | **Shared LLM client** (`llm_client.py`) | One place for all OpenAI calls, retries, and error handling. Removed ~120 lines of duplicated code. |
| 4 | **Evidence / source tracking** | `source_urls` on every `VentureRecord` traces which search results mentioned each company. |
| 5 | **Timestamped output files** | Each run writes uniquely named files (`verified_ventures_20260416_143022.csv`). No more overwriting. |

---

## What is this project?

We are researching:

> *What is the mitigating effect of new ventures and innovations on climate
> change's impact on the insurance sector?*

Phase 1 was done manually — reading reports, tagging ventures, building
spreadsheets. This system automates parts of that workflow in a careful,
auditable, beginner-friendly way.

**This is NOT a replacement for human judgment.**
It is a research assistant that reduces repetitive work and makes the process
reproducible.

---

## What does Version 1 do?

Version 1 implements a single **Venture Scout Agent** that runs this pipeline:

```
User query
   │
   ▼
[Scout]       Generate candidate venture names using an LLM
   │
   ▼
[Verifier]    For each candidate: find and confirm an official website
              → Reject if no website can be confirmed
   │
   ▼
[Extractor]   Extract structured fields (name, location, stage, funding, ...)
   │
   ▼
[Classifier]  Classify into D1–D11 drivers + Direct/Indirect category
   │
   ▼
[Saver]       Write results to CSV, JSON, and a rejected log
```

**Core rule:** If no official website can be confirmed, the venture is rejected.
Accuracy is more important than recall.

---

## Repository structure

```
insurance-foresight-agent/
├── README.md
├── .env.example            ← copy this to .env and fill in your keys
├── .gitignore
├── requirements.txt
├── pyproject.toml
│
├── src/
│   ├── __init__.py
│   ├── main.py             ← CLI entry point (run from here)
│   ├── config.py           ← loads settings from environment variables
│   ├── logger.py           ← logging setup
│   ├── models.py           ← Pydantic data models
│   ├── prompts.py          ← all LLM prompts live here
│   ├── utils.py            ← small shared helpers
│   ├── llm_client.py       ← shared OpenAI client with retries (v0.2)
│   ├── search.py           ← Tavily web search integration (v0.2)
│   ├── scout.py            ← Step 1: candidate discovery
│   ├── verifier.py         ← Step 2: website verification (3-layer, v0.2)
│   ├── extractor.py        ← Step 3: field extraction
│   ├── classifier.py       ← Step 4: D1–D11 classification
│   └── saver.py            ← Step 5: save results (timestamped, v0.2)
│
├── data/
│   └── seed_queries.txt    ← example queries you can run
│
├── output/                 ← CSV and JSON results land here
│   └── .gitkeep
│
├── logs/                   ← run logs land here
│   └── .gitkeep
│
└── tests/
    ├── test_models.py
    ├── test_classifier.py
    ├── test_utils.py
    ├── test_search.py      ← search module tests, mocked (v0.2)
    └── test_verifier.py    ← verification logic tests, mocked (v0.2)
```

---

## Setup

### 1. Clone and enter the repo

```bash
git clone <your-repo-url>
cd insurance-foresight-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Open .env and fill in at minimum: OPENAI_API_KEY
# Optionally add: TAVILY_API_KEY (strongly recommended — see below)
```

### 5. (Recommended) Add a Tavily API key for search-grounded discovery

Without a Tavily key, the Scout generates candidate names from the LLM's training
memory only. This works but can produce outdated or hallucinated companies.

With a Tavily key, the Scout first searches the web for your query and then extracts
companies from real search results. This is the main quality improvement in v0.2.

**To set it up:**
1. Sign up free at [https://tavily.com](https://tavily.com)
2. Copy your API key (`tvly-...`)
3. Add it to your `.env` file: `TAVILY_API_KEY=tvly-...`

The system automatically detects whether a key is present and selects the right mode.
The terminal output will tell you which mode is active on each run.

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | Your OpenAI API key |
| `OPENAI_MODEL` | No | `gpt-4o` | Model to use for all LLM calls |
| `OUTPUT_DIR` | No | `output` | Where to write CSV / JSON results |
| `MAX_CANDIDATES` | No | `10` | How many candidates to generate per query |
| `TAVILY_API_KEY` | No | — | Tavily search key — enables search-grounded mode |

---

## How to run

### Run the Venture Scout

```bash
python -m src.main scout "Find flood-risk ventures in Europe relevant to insurance"
```

Other example queries (see `data/seed_queries.txt` for more):

```bash
python -m src.main scout "Find cyber startups relevant to insurers"
python -m src.main scout "Find climate analytics companies that could help insurers price risk"
python -m src.main scout "Find non-obvious ventures that may mitigate catastrophe losses"
```

### Options

```bash
python -m src.main scout --help

# Specify how many candidates to generate
python -m src.main scout "Find flood ventures" --max-candidates 15

# Specify a custom output directory
python -m src.main scout "Find flood ventures" --output-dir my_results
```

### Output files

After a run you will find:

| File | Contents |
|---|---|
| `output/verified_ventures.csv` | All verified ventures with fields |
| `output/verified_ventures.json` | Same data in JSON format |
| `output/rejected_candidates.csv` | Rejected ventures with rejection reasons |
| `logs/run.log` | Full run log |

---

## Driver framework (D1–D11)

Each venture is tagged with one or more of these drivers:

| Tag | Driver |
|---|---|
| D1 | Climate Change & Catastrophe Losses |
| D2 | AI & Digital Transformation |
| D3 | Investment Returns & Financial Markets |
| D4 | Systemic Cyber Risk |
| D5 | Demographic Shifts & Longevity Risk |
| D6 | Geopolitical Fragmentation & Trade Disruption |
| D7 | Social Inflation & Litigation Cost Escalation |
| D8 | Protection Gap & Underinsurance |
| D9 | Regulatory Evolution & ESG Compliance |
| D10 | Distribution Disruption & Embedded Insurance |
| D11 | Sovereign & Corporate Debt Crisis |

And one category:
- **Direct** — clearly insurance-specific (insurtech, underwriting, cyber insurance, ...)
- **Indirect** — not primarily insurtech, but materially affects insurance outcomes

---

## Future roadmap

The code is structured so the following can be added later:

- [ ] Real-time web search to improve candidate discovery
- [ ] Browser automation / scraping to verify websites more robustly
- [ ] Signal monitoring (track ventures over time)
- [ ] Investment flow tracking (Crunchbase / PitchBook integration)
- [ ] Multiple agents (Monitor Agent, Synthesis Agent, ...)
- [ ] Database storage (SQLite or PostgreSQL)
- [ ] Notion API integration to push results to the research workspace
- [ ] Dashboard for exploring results visually
- [ ] Async processing for faster runs on large query batches
- [ ] Scheduler to run queries on a recurring basis

---

## Research context

This project is part of a foresight study examining how macro trends and
emerging ventures will shape the insurance industry over the next 10–20 years.

Collaborating institutions: IEX Labs · IE University · VIG Insurance Group
