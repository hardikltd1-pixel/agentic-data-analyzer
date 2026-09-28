# Agentic Data Analyzer

A command-line AI agent that answers natural-language questions about **any CSV file**, and **remembers useful facts across sessions** (metric definitions, corrections, preferences, named filters).

You load a CSV, ask questions in plain English, and the agent decides which analysis tool to run, executes it with pandas on the real data, and explains the result. If you tell it something reusable ("active customers means status = active only"), it saves that to disk and applies it in every future question, even after a restart.

---

## Quick start

**Requirements:** Python 3.10+ (developed and tested on 3.12) and a free Groq API key from https://console.groq.com.

```bash
# 1. (Recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key: create a file named .env in the project root
GROQ_API_KEY=your_key_here

# 4. Run
python main.py
```

When prompted, enter the path to a CSV file (for example `data/office.csv`), then start asking questions.

**Optional:** `python check_models.py` lists the models your Groq key can access. I used it to pick a model that is actually available on my account.

---
## Project structure

```
agentic-data-analyzer/
├── main.py            # Entry point: model config, agent, instructions, interactive loop
├── tools.py           # Analysis tools + persistent memory (fact store)
├── check_models.py    # Helper: lists models available to your Groq API key
├── memory.json        # Persistent memory (created/updated automatically)
├── data/              # Sample CSV files
├── requirements.txt
└── .env               # Your GROQ_API_KEY (not committed)
```

---

## How it works

```
 You ask a question
        │
        ▼
 main.py loads remembered facts from memory.json
 and puts them in front of your question
        │
        ▼
 Agent (smolagents CodeAgent + LLM on Groq)
 reads the question, decides which tool(s) to use,
 writes a short Python snippet that calls them
        │
        ▼
 Tools (tools.py) run real pandas code on your CSV
 and return small, structured results
        │
        ▼
 Agent turns the result into a plain-English answer
 (and calls remember_fact if you stated a reusable fact)
```

The important part: **the LLM never sees your whole dataset and never does the arithmetic.** It only chooses tools and explains results. All numbers come from pandas.

---

## The tools

The agent has six tools it can call. They work on whichever CSV is currently loaded, so nothing is hard-coded to one dataset.

| Tool | What it does | Notes |
|---|---|---|
| `dataset_info` | Returns row count and column names | Used for "how big is this data / what columns exist" |
| `summary_statistics` | Count, mean, std, min, median, max for numerical columns, rounded to 2 decimals | Skips ID-like columns; optional `columns` argument to focus on specific columns |
| `filter_data` | Finds rows where a column equals a value | Always returns an accurate `match_count`, plus only a capped sample of rows (default 20) so large results cannot overflow the model's context |
| `group_and_aggregate` | Average of one numeric column grouped by another (e.g. average spend per plan type) | Validates that both columns exist and returns clear errors |
| `correlation_analysis` | Correlations between numerical columns | Numeric columns only |
| `remember_fact` | Saves a reusable fact to persistent memory | See the memory section below |

`load_dataset` (also in `tools.py`) is called by `main.py` when you enter a CSV path; it stores the DataFrame that every other tool reads from.

Every tool validates its inputs (for example, an unknown column name returns a message listing the available columns), so the agent can recover from mistakes instead of crashing.

---

## Persistent memory design

### The goal
Without memory, the agent starts from zero every session: it guesses at ambiguous business terms and makes you repeat the same corrections. Memory turns a one-time correction into a lasting improvement.

### What is stored (and what is not)
Only **small, structured facts**, never chat logs and never dataset contents. Each fact has four fields:

```json
{
  "key": "active_customers",
  "category": "definition",
  "statement": "Active customers means status is active only, not trial.",
  "timestamp": 1758987123.45
}
```

Categories: `definition` (what a term means), `preference` (how you like results shown), `filter` (a reusable named filter). Corrections are handled as updates to an existing definition.

### How it works

1. **Deciding what is worth keeping.** The agent has a `remember_fact` tool, and its instructions say to call it *only* when the user's message states a reusable definition, correction, preference or named filter, and never for ordinary data questions. So "how many customers are in Delhi?" saves nothing, while "treat active customers as status active only" does.
2. **Persistence.** Facts are written to `memory.json` on disk, so they survive restarts.
3. **Updating stale or wrong facts.** Each fact has a `key`. Saving a fact with an existing key **overwrites** the old one (most recent statement wins), so a correction replaces the outdated definition instead of creating a conflicting duplicate.
4. **Retrieval.** Before every question, `main.py` reads `memory.json` and puts all remembered facts at the start of the prompt ("Known facts from earlier sessions: ..."). The agent's instructions tell it to apply them when relevant. Because the file is re-read on every question, a fact saved now applies immediately, even in the same session.
5. **Affecting later work.** Since the facts sit in the prompt, they change how the agent interprets questions. Example: after "treat active customers as status active only, not trial", the question "how many active customers are there" filters on `status == "active"` without being told again, in this session or after a restart.

### Example

```
Ask a question: treat active customers as status active only, not trial
  -> agent calls remember_fact(category="definition", key="active_customers", ...)

Ask a question: memory
  [active_customers] (definition): status active only, not trial

(restart the program, reload the CSV)

Ask a question: how many active customers are there
  -> agent applies the stored definition and answers, with no re-explaining

Ask a question: forget active_customers
```

### Limitations 
- **The AI decides what is worth saving.** It can occasionally miss a fact. The `memory` command shows what was saved, so the user can simply say it again.
- **Memory is shared across all datasets.** A fact saved for one CSV is also shown when a different CSV is loaded. The `forget <key>` command removes it, and per-dataset memory would be a natural next step.
- **Facts are saved without asking for confirmation.** A "save this? yes/no" step would give the user more control.

---

## Dataset

The app works with **any CSV**. For testing I used a customer dataset with 14 columns:

`customer_id, name, city, country, age, signup_date, status, plan_type, monthly_spend, tenure_months, support_tickets, satisfaction_score, last_login_days_ago, referred_by`

`status` takes the values `active`, `cancelled` and `trial`, which makes it a good test for memory: "active customers" is genuinely ambiguous (do trials count?), so a stored definition visibly changes the answer.

### Questions to try

| Feature | Try asking |
|---|---|
| Structure | `how many rows and columns does this dataset have` |
| Statistics | `give me summary statistics for this dataset` |
| Filtering | `how many active customers are there` |
| Grouping | `what's the average monthly spend by plan type` |
| Correlation | `what correlates with satisfaction score` |
| Memory: define | `treat active customers as status active only, not trial` |
| Memory: apply | `how many active customers are there` (again) |
| Memory: inspect | `memory` |
| Memory: preference | `I prefer spend broken down by city, not overall totals` |
| Memory: control | `forget active_customers` |

To check persistence across sessions, save a fact, exit, restart, reload the CSV and ask a question that depends on it.

---

## A note on generic wrappers

This is not a chat box forwarding your question to an LLM. The differences:

- **The model does not answer from its imagination.** It has to call tools, and every number in an answer comes from pandas running on your actual CSV. The instructions forbid guessing values.
- **It is an agent, not a single prompt.** It plans, calls tools, reads results and decides its next step (up to 8 steps per question). It uses smolagents' `CodeAgent`, so it writes a short Python snippet that can combine several tools in one go.
- **It is engineered for real data.** `filter_data` caps how many rows it returns, `summary_statistics` skips ID columns and rounds values, all tools validate their inputs, requests have a timeout, and an error on one question never ends the session.
- **Memory is a designed subsystem.** It has a schema, selective writes, keyed updates for corrections, injection at question time, and user-facing inspect/delete controls, and it deliberately does not store raw chats or data.

An LLM is still what plans and explains, since that is the point of an agent. The value of this project is in the tools, the guardrails and the memory around it.

---

## Design decisions and what I learned

- **Agent type: `CodeAgent`.** It has the model write short Python snippets that calls tools as functions, which was more forgiving than forcing strict JSON tool calls on every step.
- **Model: `qwen/qwen3.8-27b` on Groq**, chosen after testing. I tried `openai/gpt-oss-20b` with both `CodeAgent` and `ToolCallingAgent`, and both failed with HTTP 400 "tool choice" errors from the API, so I did not use it. Qwen ran the full flow reliably. `reasoning_effort="none"` turns off its reasoning narration so responses stay in the required `<code>` format.
- **Timeout of 30s.** Hosted models can occasionally be slow or queued. A timeout turns a silent freeze into a clear error, and the session continues.
- **Bug found and fixed while testing:** the first version of `filter_data` returned every matching row. On a large CSV that overflowed the model's context and the request failed. It now returns an accurate count plus a capped sample, and the instructions tell the agent to use `match_count` for "how many" questions.
- **Secrets:** the API key lives in `.env`, which is git-ignored.

---

## Known limitations

- Response speed depends on the hosted model and its current load.