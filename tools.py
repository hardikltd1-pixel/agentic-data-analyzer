import json
import time
from pathlib import Path

import pandas as pd
from smolagents import tool


# The currently loaded dataset
df: pd.DataFrame | None = None


def _get_dataset() -> pd.DataFrame:
    """Return the loaded dataset or raise an error if none is loaded."""
    if df is None:
        raise ValueError("No dataset has been loaded.")
    return df


@tool
def load_dataset(file_path: str) -> dict:
    """
    Loads a CSV dataset from a file path.

    Args:
        file_path: The path to the CSV file.

    Returns:
        A dictionary containing information about the loaded dataset.
    """
    global df

    df = pd.read_csv(file_path)

    return {
        "message": "Dataset loaded successfully.",
        "rows": len(df),
        "columns": list(df.columns)
    }


@tool
def dataset_info() -> dict:
    """
    Returns basic information about the currently loaded dataset.

    Returns:
        A dictionary containing the number of rows and column names."""
    data = _get_dataset()

    return {
        "rows": len(data),
        "columns": list(data.columns)
    }


@tool
def summary_statistics(columns: str = "") -> dict:
    """
    Returns compact statistics (count, mean, std, min, median, max) for
    numerical columns, rounded to 2 decimals. ID-like columns are skipped.

    Args:
        columns: Optional comma-separated column names to restrict the
            output (e.g. "monthly_spend,age"). Leave empty to summarize
            all numerical columns.

    Returns:
        A dictionary mapping each column to its statistics.
    """
    data = _get_dataset()

    numeric = data.select_dtypes(include="number")

    # Skip ID-like columns that carry no useful statistics
    skip = [c for c in numeric.columns
            if c.lower().endswith("_id") or c.lower() in {"id", "referred_by"}]
    numeric = numeric.drop(columns=skip, errors="ignore")

    if columns.strip():
        wanted = [c.strip() for c in columns.split(",")]
        missing = [c for c in wanted if c not in numeric.columns]
        if missing:
            raise ValueError(
                f"Not numerical or not found: {missing}. "
                f"Available: {list(numeric.columns)}"
            )
        numeric = numeric[wanted]

    stats = numeric.describe().loc[["count", "mean", "std", "min", "50%", "max"]]
    stats = stats.rename(index={"50%": "median"}).round(2)

    return stats.to_dict()


@tool
def filter_data(column: str, value: str, limit: int = 20) -> dict:
    """
    Filters the dataset where a column equals a given value.

    Args:
        column: The name of the column to filter.
        value: The value that the column should equal.
        limit: Maximum number of matching rows to return in full detail
            (default 20). Keeps responses small on large datasets. The
            total match count is always accurate regardless of this limit.

    Returns:
        A dictionary with:
          - match_count: total number of matching rows (always accurate,
            even if more rows exist than `limit`)
          - returned_count: how many rows are actually included below
          - records: up to `limit` matching rows as full records
    """
    data = _get_dataset()

    if column not in data.columns:
        raise ValueError(
            f"Column '{column}' does not exist. "
            f"Available columns: {list(data.columns)}"
        )

    matches = data[data[column].astype(str) == value]

    return {
        "match_count": len(matches),
        "returned_count": min(limit, len(matches)),
        "records": matches.head(limit).to_dict(orient="records"),
    }


@tool
def group_and_aggregate(
    group_column: str,
    value_column: str
) -> dict:
    """
    Groups the dataset and calculates the average of a numerical column.

    Args:
        group_column: The column used to create groups.
        value_column: The numerical column whose average is calculated.

    Returns:
        A dictionary containing the average for each group.
    """
    data = _get_dataset()

    if group_column not in data.columns:
        raise ValueError(f"Column '{group_column}' does not exist.")

    if value_column not in data.columns:
        raise ValueError(f"Column '{value_column}' does not exist.")

    return data.groupby(group_column)[value_column].mean().to_dict()


@tool
def correlation_analysis() -> dict:
    """
    Calculates correlations between numerical columns.

    Returns:
        A dictionary containing correlation values.
    """
    data = _get_dataset()

    return data.corr(numeric_only=True).to_dict()


# ---------------------------------------------------------------------------
# Persistent memory
# ---------------------------------------------------------------------------
# Stores small, structured facts (definitions, corrections, preferences,
# named filters) across sessions -- never raw chat logs or dataset dumps.

MEMORY_FILE = Path("memory.json")


def _load_memory() -> list[dict]:
    if not MEMORY_FILE.exists():
        return []
    try:
        return json.loads(MEMORY_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _save_memory(facts: list[dict]) -> None:
    MEMORY_FILE.write_text(json.dumps(facts, indent=2))


@tool
def remember_fact(category: str, key: str, statement: str) -> dict:
    """
    Saves a reusable fact -- a metric definition, a correction, a
    reporting preference, or a named filter -- so it can be reused in
    future questions, even after the program restarts. Only call this
    when the user's question itself states such a fact (e.g. "treat
    active customers as non-cancelled accounts", "I prefer weekly
    summaries"). Do not call this for plain data questions.

    Args:
        category: One of "definition", "preference", or "filter".
        key: A short, unique, snake_case identifier for this fact
            (e.g. "active_customers"). Using the same key again
            overwrites the old fact -- that's how corrections work.
        statement: The fact itself, in plain English.

    Returns:
        A dictionary confirming what was remembered.
    """
    facts = [f for f in _load_memory() if f["key"] != key]
    facts.append({
        "key": key,
        "category": category,
        "statement": statement,
        "timestamp": time.time(),
    })
    _save_memory(facts)

    return {"message": f"Remembered ({category}): {statement}"}


def get_all_facts() -> list[dict]:
    """Plain helper (not an agent tool): returns every stored fact.
    Used by main.py to inject context and to power the 'memory' command."""
    return _load_memory()


def forget_fact_by_key(key: str) -> bool:
    """Plain helper (not an agent tool): removes a fact by key.
    Used by main.py to power the 'forget <key>' command."""
    facts = _load_memory()
    remaining = [f for f in facts if f["key"] != key]
    _save_memory(remaining)
    return len(remaining) < len(facts)