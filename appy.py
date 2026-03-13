
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
st.sidebar.markdown("Toggle caching for faster loads in production below.")
use_cache = st.sidebar.checkbox("Use cache for dataset load", value=True)

# -------------------------
# Utility functions
# -------------------------
def raw_is_answerable(example):
    """
    Compute answerability directly from the raw HF example dict.
    Mirrors the Colab logic: check example['answers']['text'].
    """
    a = example.get("answers", {})
    if isinstance(a, dict) and "text" in a:
        texts = a["text"]
        if isinstance(texts, (list, tuple)):
            return any(bool(t) for t in texts)
        return bool(str(texts).strip())
    if isinstance(a, (list, tuple)):
        if len(a) == 0:
            return False
        first = a[0]
        if isinstance(first, dict) and "text" in first:
            for item in a:
                t = item.get("text")
                if isinstance(t, (list, tuple)):
                    if any(bool(x) for x in t):
                        return True
                elif bool(str(t).strip()):
                    return True
            return False
        return any(bool(str(x).strip()) for x in a)
    return False

def normalize_answers(cell):
    """Normalize various possible representations of `answers` into a list of text strings."""
    if cell is None:
        return []
    if isinstance(cell, dict) and "text" in cell:
        texts = cell["text"]
        if isinstance(texts, (list, tuple)):
            return [str(x) for x in texts if x is not None and str(x).strip() != ""]
        return [str(texts)] if texts is not None and str(texts).strip() != "" else []
    if isinstance(cell, (list, tuple)):
        out = []
        for item in cell:
            if isinstance(item, dict) and "text" in item:
                t = item["text"]
                if isinstance(t, (list, tuple)):
                    out.extend([str(x) for x in t if x is not None and str(x).strip() != ""])
                else:
                    if t is not None and str(t).strip() != "":
                        out.append(str(t))
            else:
                if item is not None and str(item).strip() != "":
                    out.append(str(item))
        return out
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
    return [str(cell)]

# -------------------------
# Core loader (raw-based, mirrors Colab)
# -------------------------
def load_and_prepare_raw(debug=False):
    """
    Load SQuAD v2 from Hugging Face, compute is_answerable from raw examples,
    and return combined DataFrame and summary DataFrame.
    """
    ds = load_dataset("squad_v2")

    if debug:
        st.write("datasets version:", __import__("datasets").__version__)
        st.write("pandas version:", pd.__version__)
        st.write("train length:", len(ds["train"]))
        st.write("validation length:", len(ds["validation"]))
        st.write("Raw sample ds['train'][0]:")
        st.json(ds["train"][0])

    # Build rows from raw examples (compute is_answerable using raw_is_answerable)
    def examples_to_rows(split):
        rows = []
        for ex in split:
            is_ans = raw_is_answerable(ex)
            rows.append({
                "id": ex.get("id"),
                "title": ex.get("title"),
                "context": ex.get("context"),
                "question": ex.get("question"),
                "answers": ex.get("answers"),
                "is_answerable": is_ans
            })
        return rows

    train_rows = examples_to_rows(ds["train"])
    valid_rows = examples_to_rows(ds["validation"])

    # Create DataFrames (fast and predictable)
    train_df = pd.DataFrame(train_rows)
    valid_df = pd.DataFrame(valid_rows)

    # Combine
    combined = pd.concat([train_df, valid_df], ignore_index=True)

    # Summary
    summary = combined["is_answerable"].value_counts().rename_axis("is_answerable").reset_index(name="count")
    summary["label"] = summary["is_answerable"].map({True: "answerable", False: "unanswerable"})

    if debug:
        st.write("Converted combined shape:", combined.shape)
        st.write("Converted sample row (head):", combined.head(3).to_dict(orient="records"))
        st.write("Raw HF answerable (train):", sum(1 for ex in ds["train"] if raw_is_answerable(ex)), "/", len(ds["train"]))
        st.write("Raw HF answerable (validation):", sum(1 for ex in ds["validation"] if raw_is_answerable(ex)), "/", len(ds["validation"]))
        st.write("Combined answerable count:", int(combined["is_answerable"].sum()), "/", combined.shape[0])

    return combined, summary

# Cached wrapper for production loads
if use_cache:
    @st.cache_data
    def cached_load():
        return load_and_prepare_raw(debug=False)
else:
    def cached_load():
        return load_and_prepare_raw(debug=False)

# Load data (debug uses direct call to show diagnostics)
if debug:
    data, summary = load_and_prepare_raw(debug=True)
else:
    data, summary = cached_load()

# -------------------------
# Robust plotting and UI
# -------------------------
st.title("SQuAD v2 Answerability Dashboard")
st.markdown("Proportion of answerable vs unanswerable examples (computed from raw HF examples).")

# Defensive summary display for debugging
if debug:
    st.subheader("Summary (debug)")
    st.write(summary)

st.subheader("Overall proportions")

# Defensive checks and robust plotting
if summary is None or summary.empty:
    st.warning("Summary is empty — cannot draw pie chart.")
else:
    if not {"label", "count"}.issubset(set(summary.columns)):
        st.warning("Summary missing required columns 'label' and 'count'. Showing table instead.")
        st.table(summary)
    else:
        summary_plot = summary.copy()
        summary_plot["count"] = pd.to_numeric(summary_plot["count"], errors="coerce").fillna(0).astype(int)
        summary_plot = summary_plot[summary_plot["count"] > 0]
        if summary_plot.empty:
            st.warning("No nonzero counts to plot. Showing table instead.")
            st.table(summary_plot)
        else:
                # Robust pie chart (replace existing pie code with this)
import plotly.graph_objects as go

if summary is None or summary.empty:
    st.warning("Summary is empty — cannot draw pie chart.")
else:
    if not {"label", "count"}.issubset(set(summary.columns)):
        st.warning("Summary missing required columns 'label' and 'count'. Showing table instead.")
        st.table(summary)
    else:
        summary_plot = summary.copy()
        summary_plot["count"] = pd.to_numeric(summary_plot["count"], errors="coerce").fillna(0).astype(int)
        # DEBUG lines — remove after confirming
        st.write("DEBUG summary_plot:", summary_plot)
        labels = summary_plot["label"].astype(str).tolist()
        values = summary_plot["count"].astype(int).tolist()
        st.write("DEBUG labels:", labels)
        st.write("DEBUG values:", values)
        st.write("DEBUG sum(values):", sum(values))
        if sum(values) == 0:
            st.warning("All counts are zero — nothing to plot.")
            st.table(summary_plot)
        else:
            try:
                fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.0, sort=False,
                                       marker=dict(colors=px.colors.qualitative.Set2)))
                fig.update_traces(textposition="inside", textinfo="percent+label")
                fig.update_layout(title_text="Answerable vs Unanswerable")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error("Failed to render pie chart: " + str(e))
                try:
                    fig_bar = px.bar(summary_plot, x="label", y="count", title="Answerable vs Unanswerable (bar)")
                    st.plotly_chart(fig_bar, use_container_width=True)
                except Exception:
                    st.table(summary_plot)


st.subheader("Counts by label")
try:
    st.table(summary[["label", "count"]].set_index("label"))
except Exception:
    st.write(summary)

st.subheader("Sample unanswerable questions")
if "is_answerable" in data.columns:
    unans = data[data["is_answerable"] == False]
    if not unans.empty:
        try:
            sample_unans = unans.sample(min(10, len(unans)), random_state=42)[["id", "context", "question"]]
            st.table(sample_unans.reset_index(drop=True))
        except Exception:
            # fallback: show first rows
            st.table(unans.head(10)[["id", "context", "question"]].reset_index(drop=True))
    else:
        st.write("No unanswerable samples found.")
else:
    st.write("No is_answerable column present in data")

st.subheader("Download combined dataset")
try:
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button(label="Download CSV", data=csv, file_name="squad_v2_combined.csv", mime="text/csv")
except Exception:
    st.write("Download not available.")

st.sidebar.markdown("---")
st.sidebar.markdown("If you enabled Debug mode, the app shows raw dataset diagnostics to help identify mismatches between environments.")

