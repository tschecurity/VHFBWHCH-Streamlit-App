# app.py
import json
import streamlit as st
import pandas as pd
import plotly.express as px
from datasets import load_dataset

st.set_page_config(page_title="SQuAD v2 Answerability Dashboard", layout="wide")

# Sidebar controls
st.sidebar.title("Settings")
debug = st.sidebar.checkbox("Debug mode (show diagnostics)", value=False)
st.sidebar.markdown("When **Debug mode** is on the app will show raw dataset diagnostics.")

def normalize_answers(cell):
    """Normalize various possible representations of `answers` into a list of text strings."""
    if cell is None:
        return []
    # dict with 'text' key (HF typical)
    if isinstance(cell, dict) and "text" in cell:
        texts = cell["text"]
        if isinstance(texts, (list, tuple)):
            return [str(x) for x in texts if x is not None]
        return [str(texts)] if texts is not None else []
    # list of dicts or strings
    if isinstance(cell, (list, tuple)):
        out = []
        for item in cell:
            if isinstance(item, dict) and "text" in item:
                t = item["text"]
                if isinstance(t, (list, tuple)):
                    out.extend([str(x) for x in t if x is not None])
                else:
                    out.append(str(t))
            else:
                out.append(str(item))
        return [x for x in out if x not in ("", "None", "[]")]
    # string: maybe JSON-encoded or plain
    if isinstance(cell, str):
        s = cell.strip()
        if s == "" or s == "[]":
            return []
        if (s.startswith("{") or s.startswith("[")):
            try:
                parsed = json.loads(s)
                return normalize_answers(parsed)
            except Exception:
                return [s]
        return [s]
    # fallback
    return [str(cell)]

def compute_is_answerable_from_raw(example):
    """Robust check for raw HF example dict."""
    a = example.get("answers", {})
    texts = a.get("text") if isinstance(a, dict) else a
    if texts is None:
        return False
    if isinstance(texts, (list, tuple)):
        return any(bool(t) for t in texts)
    if isinstance(texts, str):
        return bool(texts.strip())
    return False

def load_and_prepare(debug=False):
    """
    Load SQuAD v2, convert to pandas, normalize answers, compute is_answerable,
    and return combined dataframe and summary.
    """
    # Load dataset (may download on first run)
    try:
        ds = load_dataset("squad_v2")
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        raise

    if debug:
        st.write("datasets version:", __import__("datasets").__version__)
        st.write("pandas version:", pd.__version__)
        st.write("train length:", len(ds["train"]))
        st.write("validation length:", len(ds["validation"]))
        st.write("Raw sample ds['train'][0]:")
        st.json(ds["train"][0])

    # Convert splits to pandas DataFrames using to_pandas for stability
    try:
        train_df = ds["train"].to_pandas()
        valid_df = ds["validation"].to_pandas()
    except Exception:
        train_df = pd.DataFrame(ds["train"])
        valid_df = pd.DataFrame(ds["validation"])

    # Ensure columns are clean
    train_df.columns = [c.strip() for c in train_df.columns]
    valid_df.columns = [c.strip() for c in valid_df.columns]

    # Compute is_answerable robustly on DataFrame
    def df_is_answerable_cell(cell):
        texts = normalize_answers(cell)
        return any(bool(t) for t in texts)

    if "answers" in train_df.columns:
        train_df["is_answerable"] = train_df["answers"].apply(df_is_answerable_cell)
        valid_df["is_answerable"] = valid_df["answers"].apply(df_is_answerable_cell)
    else:
        # If answers column missing, try to detect alternative column names or set False
        train_df["is_answerable"] = False
        valid_df["is_answerable"] = False
        if debug:
            st.warning("No 'answers' column found in converted DataFrame; columns: " + ", ".join(train_df.columns))

    # Combine splits
    combined = pd.concat([train_df, valid_df], ignore_index=True)

    # Summary counts
    summary = combined["is_answerable"].value_counts().rename_axis("is_answerable").reset_index(name="count")
    summary["label"] = summary["is_answerable"].map({True: "answerable", False: "unanswerable"})

    if debug:
        st.write("Converted train columns:", list(train_df.columns))
        st.write("Converted combined shape:", combined.shape)
        st.write("Converted sample row (head):", combined.head(3).to_dict(orient="records"))

        # Compare raw HF counts vs converted DF counts for quick sanity
        raw_train_ans = sum(1 for ex in ds["train"] if compute_is_answerable_from_raw(ex))
        raw_train_total = len(ds["train"])
        raw_valid_ans = sum(1 for ex in ds["validation"] if compute_is_answerable_from_raw(ex))
        raw_valid_total = len(ds["validation"])
        st.write("Raw HF answerable (train):", raw_train_ans, "/", raw_train_total)
        st.write("Raw HF answerable (validation):", raw_valid_ans, "/", raw_valid_total)
        st.write("Converted DF answerable (train+valid):", int(combined["is_answerable"].sum()), "/", combined.shape[0])

        # Show first few disagreements (if any) between raw and converted for train split
        disagreements = []
        N = min(200, len(ds["train"]))
        for i, ex in enumerate(ds["train"][:N]):
            raw_flag = compute_is_answerable_from_raw(ex)
            # try match by id if present
            df_flag = None
            if "id" in train_df.columns:
                row = train_df[train_df["id"] == ex.get("id")]
                if not row.empty:
                    df_flag = bool(row["is_answerable"].iloc[0])
            if df_flag is None:
                if i < len(train_df):
                    df_flag = bool(train_df["is_answerable"].iloc[i])
            if df_flag is not None and raw_flag != df_flag:
                disagreements.append({
                    "index": i,
                    "id": ex.get("id"),
                    "raw_answers": ex.get("answers"),
                    "raw_flag": raw_flag,
                    "df_answers_repr": repr(train_df["answers"].iloc[i]) if i < len(train_df) else None,
                    "df_flag": df_flag
                })
        st.write("Number of disagreements in first", N, "train rows:", len(disagreements))
        if disagreements:
            st.json(disagreements[:5])

    return combined, summary

# Cached wrapper for production loads (debug=False)
@st.cache_data
def cached_load():
    return load_and_prepare(debug=False)

# Load data (use cache in production, direct load in debug)
if debug:
    data, summary = load_and_prepare(debug=True)
else:
    data, summary = cached_load()

# UI
st.title("SQuAD v2 Answerability Dashboard")
st.markdown("This dashboard shows the proportion of answerable vs unanswerable examples in SQuAD v2.")

st.subheader("Overall proportions")
fig_pie = px.pie(summary, names="label", values="count",
                 title="Answerable vs Unanswerable",
                 color_discrete_sequence=px.colors.qualitative.Set2)
fig_pie.update_traces(textposition="inside", textinfo="percent+label")
st.plotly_chart(fig_pie, use_container_width=True)

st.subheader("Counts by label")
st.table(summary[["label", "count"]].set_index("label"))

st.subheader("Sample unanswerable questions")
if "is_answerable" in data.columns:
    try:
        sample_unans = data[data["is_answerable"] == False].sample(10, random_state=42)[["id", "context", "question"]]
        st.table(sample_unans.reset_index(drop=True))
    except Exception:
        st.write("No unanswerable samples available to display.")
else:
    st.write("No is_answerable column present in data")

st.subheader("Download combined dataset")
csv = data.to_csv(index=False).encode("utf-8")
st.download_button(label="Download CSV", data=csv, file_name="squad_v2_combined.csv", mime="text/csv")

st.sidebar.markdown("---")
st.sidebar.markdown("If you enabled Debug mode, the app shows raw dataset diagnostics to help identify mismatches between environments.")

