"""
main.py
--------
Entry point for the agentic data-analysis assistant.
"""

import os
import sys

from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIModel

from tools import (
    load_dataset,
    dataset_info,
    summary_statistics,
    filter_data,
    group_and_aggregate,
    correlation_analysis,
    remember_fact,
    get_all_facts,
    forget_fact_by_key,
)


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print(
        "ERROR: GROQ_API_KEY not found in your .env file.\n"
        "Add a line like:\n"
        "  GROQ_API_KEY=your_key_here\n"
        "Get a free key at https://console.groq.com"
    )
    sys.exit(1)

model = OpenAIModel(
    model_id="qwen/qwen3.8-27b",
    api_key=GROQ_API_KEY,
    api_base="https://api.groq.com/openai/v1",
    temperature=0.1,
    reasoning_effort="none",  # instruct mode -- no reasoning-trace narration
    timeout=30,
)


# ---------------------------------------------------------------------------
# 2. Agent instructions
# ---------------------------------------------------------------------------

instructions = """
You are a data analysis agent. Answer questions about the currently loaded
CSV dataset by writing short Python code that calls the tools available to
you. Never invent or guess dataset values -- always get real data from a
tool first.

Tool selection guide:
- dataset_info: dataset size, columns, or structure.
- summary_statistics: numerical statistics (mean, std, min, max, etc).
- filter_data: rows where a specific column equals a specific value.
  It returns match_count (always accurate) plus a capped sample of
  records (default 20). For "how many..." questions, use match_count
  directly -- do not count the records list, since it may be truncated.
- group_and_aggregate: an average of one column grouped by another
  (e.g. "average salary per city").
- correlation_analysis: relationships between numerical columns.
- remember_fact: call this ONLY when the user's question itself states a
  reusable definition, correction, preference, or named filter (e.g.
  "treat active customers as non-cancelled accounts", "I prefer weekly
  summaries, not daily"). Do not call it for ordinary data questions.
  Still answer the user's actual question afterward, using the other
  tools as needed.

Any previously remembered facts will be given to you at the start of the
question, prefixed with "Known facts from earlier sessions:". Apply them
when relevant instead of re-deriving or re-asking.

IMPORTANT -- every response you give, including your final answer, MUST be
valid Python code wrapped exactly like this:

Thoughts: <your brief reasoning>
<code>
<your python code here>
</code>

Once you have the real result from a tool, finish with a final_answer call
in that same format, for example:

Thoughts: I have the data I need, so I will summarize it.
<code>
final_answer("The average salary in Delhi is 32,333 and in Mumbai is 42,500.")
</code>
"""


# ---------------------------------------------------------------------------
# 3. Build the agent
# ---------------------------------------------------------------------------

agent = CodeAgent(
    tools=[
        dataset_info,
        summary_statistics,
        filter_data,
        group_and_aggregate,
        correlation_analysis,
        remember_fact,
    ],
    model=model,
    instructions=instructions,
    max_steps=8,
)


# ---------------------------------------------------------------------------
# 4. Dataset loading (any CSV -- not hard-coded to one example)
# ---------------------------------------------------------------------------

def load_csv_with_retry() -> None:
    """Keep asking for a CSV path until one loads successfully."""
    while True:
        file_path = input("Enter the path to your CSV file: ").strip()
        try:
            result = load_dataset(file_path)
            print(
                f"\nDataset loaded: {result['rows']} rows, "
                f"{len(result['columns'])} columns -> {result['columns']}\n"
            )
            return
        except FileNotFoundError:
            print(f"File not found: {file_path}. Try again.\n")
        except Exception as e:
            print(f"Could not load that file ({e}). Try again.\n")


# ---------------------------------------------------------------------------
# 5. Interactive Q&A loop
# ---------------------------------------------------------------------------

def main() -> None:
    load_csv_with_retry()

    print("Dataset ready. Ask a question about it.")
    print("Type 'load' to switch to a different CSV, 'memory' to see what's")
    print("remembered, 'forget <key>' to remove a fact, or 'exit' to quit.\n")

    while True:
        question = input("Ask a question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if question.lower() == "load":
            load_csv_with_retry()
            continue

        if question.lower() == "memory":
            facts = get_all_facts()
            if not facts:
                print("\nNo facts remembered yet.\n")
            else:
                print()
                for f in facts:
                    print(f"[{f['key']}] ({f['category']}): {f['statement']}")
                print()
            continue

        if question.lower().startswith("forget "):
            key = question.split(" ", 1)[1].strip()
            removed = forget_fact_by_key(key)
            print("\nForgotten.\n" if removed else "\nNo fact with that key.\n")
            continue

        if not question:
            continue

        try:
            facts = get_all_facts()
            if facts:
                known = "\n".join(f"- {f['statement']}" for f in facts)
                prompt = f"Known facts from earlier sessions:\n{known}\n\nQuestion: {question}"
            else:
                prompt = question

            answer = agent.run(prompt)
            print(f"\nANSWER: {answer}\n")
        except Exception as e:
            # Keep the session alive even if one question fails -- important
            # for live demos with unfamiliar datasets/questions.
            print(f"\nSomething went wrong answering that question: {e}\n")


if __name__ == "__main__":
    main()