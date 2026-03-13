# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from datasets import load_dataset

st.set_page_config(page_title="SQuAD v2 Answerability Dashboard", layout="wide")

# Sidebar: debug toggle
st.sidebar.title("Settings")
debug = st.sidebar.checkbox("Debug mode (show diagnostics)", value=False)

def raw_is_answerable(example):
    """
    Compute answerability directly from the raw HF example dict.
    This mirrors the typical Colab approach: check example['answers']['text'].
    """
    a = example.get("answers", {})
    # HF canonical structure: {'text': [...], 'answer_start': [...]}
    if isinstance(a, dict) and "text" in a:
        texts = a["text"]
        if isinstance(texts, (list, tuple)):
            return any(bool(t) for t in texts)
        return bool(str(texts).strip())
    # fallback: if answers is list of dicts or strings
    if isinstance(a, (list, tuple)):
        if len(a) == 0:
            return False
        first = a[0]
        if isinstance(first, dict) and "text" in first:
            # list of dicts with 'text' keys
            for item in a:
                t = item.get("text")
                if isinstance(t, (list, tuple)):
                    if any(bool(x) for x in t):
                        return True
                elif bool(str(t).strip()):
                    return True
            return False
        # list of strings
        return any(bool(str(x).strip()) for x in a)
    return False

@st.cache_data
def load_and_prepare_cached():
    # wrapper to cache the heavy load in production
    return load_and_prepare(debug=False)

def load_and_prepare(debug=False):
    # Load dataset (same call as in Colab)
    ds = load_dataset("squad_v2")

    if debug:
        st.write("datasets version:", __import__("datasets").__version__)
        st.write("pandas version:", pd.__version__)
        st.write("train length:", len(ds["train"]))
        st.write("validation length:", len(ds["validation"]))
        st.write("Raw sample ds['train'][0]:")
        st.json(ds["train"][0])

    # Compute is_answerable from raw examples and build DataFrame from raw dicts
    def examples_to_df(split):
        rows = []
        for ex in split:
            # compute answerable using raw structure (this mirrors Colab)
            is_ans = raw_is_answerable(ex)
            # copy fields we need; keep original nested fields intact
            row = {
                "id": ex.get("id"),
                "title": ex.get("title"),
                "context": ex.get("context"),
                "question": ex.get("question"),
                "answers": ex.get("answers"),
                "is_answerable": is_ans
            }
            rows.append(row)
        return pd.DataFrame(rows)

    train_df = examples_to_df(ds["train"])
    valid_df = examples_to_df(ds["validation"])
    combined = pd.concat([train_df, valid_df], ignore_index=True)

    # Summary counts (stable)
    summary = combined["is_answerable"].value_counts().rename_axis("is_answerable").reset_index(name="count")
    summary["label"] = summary["is_answerable"].map({True: "answerable", False: "unanswerable"})

    if debug:
        st.write("Converted combined shape:", combined.shape)
        st.write("Converted sample row (head):", combined.head(3).to_dict(orient="records"))
        st.write("Raw HF answerable (train):", sum(1 for ex in ds["train"] if raw_is_answerable(ex)), "/", len(ds["train"]))
        st.write("Raw HF answerable (validation):", sum(1 for ex in ds["validation"] if raw_is_answerable(ex)), "/", len(ds["validation"]))
        st.write("Combined answerable count:", int(combined["is_answerable"].sum()), "/", combined.shape[0])

    return combined, summary

# Load data (use cache when not debugging)
if debug:
    data, summary = load_and_prepare(debug=True)
else:
    data, summary = load_and_prepare_cached()

# UI: visualizations and tables
st.title("SQuAD v2 Answerability Dashboard")
st.markdown("Proportion of answerable vs unanswerable examples (computed from raw HF examples).")

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
    unans = data[data["is_answerable"] == False]
    if not unans.empty:
        sample_unans = unans.sample(min(10, len(unans)), random_state=42)[["id", "context", "question"]]
        st.table(sample_unans.reset_index(drop=True))
    else:
        st.write("No unanswerable samples found.")
else:
    st.write("No is_answerable column present in data")

st.subheader("Download combined dataset")
csv = data.to_csv(index=False).encode("utf-8")
st.download_button(label="Download CSV", data=csv, file_name="squad_v2_combined.csv", mime="text/csv")
