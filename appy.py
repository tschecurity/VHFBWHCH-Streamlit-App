# app.py
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datasets import load_dataset

st.set_page_config(page_title="SQuAD v2 Answerability Dashboard", layout="wide")

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.title("Settings")
debug = st.sidebar.checkbox("Debug mode (show diagnostics)", value=False)
st.sidebar.markdown("When **Debug mode** is on the app will show raw dataset diagnostics.")
st.sidebar.markdown("Toggle caching and manual load to reduce startup lag.")
use_cache = st.sidebar.checkbox("Use cache for dataset load", value=True)
manual_load = st.sidebar.checkbox("Manual dataset load (click button to load)", value=False)

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

def safe_load_dataset(name):
    """
    Load dataset with a spinner and catch exceptions so the app doesn't crash on startup.
    Returns a dict-like object with 'train' and 'validation' keys even on failure.
    """
    try:
        with st.spinner(f"Loading dataset {name} (may take a minute)..."):
            ds = load_dataset(name)
        return ds
    except Exception as e:
        st.error(f"Dataset load failed: {e}")
        # return minimal empty structure so app can continue
        return {"train": [], "validation": []}

# -------------------------
# Core loader (raw-based, mirrors Colab)
# -------------------------
def load_and_prepare_raw(ds, debug=False):
    """
    Build DataFrame from raw HF examples (compute is_answerable from raw examples),
    and return combined DataFrame and summary DataFrame.
    """
    # If ds is empty structure, return empty frames
    if not ds or ("train" not in ds and "validation" not in ds):
        empty_df = pd.DataFrame(columns=["id", "title", "context", "question", "answers", "is_answerable"])
        summary = pd.DataFrame(columns=["is_answerable", "count", "label"])
        return empty_df, summary

    if debug:
        try:
            st.write("datasets version:", __import__("datasets").__version__)
        except Exception:
            pass
        st.write("pandas version:", pd.__version__)
        st.write("train length:", len(ds["train"]))
        st.write("validation length:", len(ds["validation"]))
        if len(ds["train"]) > 0:
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

    # Create DataFrames
    train_df = pd.DataFrame(train_rows)
    valid_df = pd.DataFrame(valid_rows)

    # Combine
    combined = pd.concat([train_df, valid_df], ignore_index=True)

    # Summary
    if "is_answerable" in combined.columns and not combined["is_answerable"].empty:
        summary = combined["is_answerable"].value_counts().rename_axis("is_answerable").reset_index(name="count")
        summary["label"] = summary["is_answerable"].map({True: "answerable", False: "unanswerable"})
    else:
        summary = pd.DataFrame(columns=["is_answerable", "count", "label"])

    if debug:
        st.write("Converted combined shape:", combined.shape)
        st.write("Converted sample row (head):", combined.head(3).to_dict(orient="records"))
        st.write("Raw HF answerable (train):", sum(1 for ex in ds["train"] if raw_is_answerable(ex)), "/", len(ds["train"]))
        st.write("Raw HF answerable (validation):", sum(1 for ex in ds["validation"] if raw_is_answerable(ex)), "/", len(ds["validation"]))
        st.write("Combined answerable count:", int(combined["is_answerable"].sum()) if "is_answerable" in combined.columns else 0, "/", combined.shape[0])

    return combined, summary

# -------------------------
# Cached wrapper for production loads
# -------------------------
if use_cache:
    @st.cache_data
    def cached_load_and_prepare(name="squad_v2"):
        ds = safe_load_dataset(name)
        return load_and_prepare_raw(ds, debug=False)
else:
    def cached_load_and_prepare(name="squad_v2"):
        ds = safe_load_dataset(name)
        return load_and_prepare_raw(ds, debug=False)

# -------------------------
# Load data (manual or automatic)
# -------------------------
data = pd.DataFrame()
summary = pd.DataFrame()

if manual_load:
    if st.sidebar.button("Load dataset now"):
        ds = safe_load_dataset("squad_v2")
        data, summary = load_and_prepare_raw(ds, debug=debug)
    else:
        st.sidebar.info("Click 'Load dataset now' to download SQuAD v2.")
else:
    # automatic load (cached or not)
    data, summary = cached_load_and_prepare("squad_v2")

# -------------------------
# UI and robust plotting
# -------------------------
st.title("SQuAD v2 Answerability Dashboard")
st.markdown("Proportion of answerable vs unanswerable examples (computed from raw HF examples).")

# Defensive summary display for debugging
if debug:
    st.subheader("Summary (debug)")
    st.write(summary)

st.subheader("Overall proportions")

# --- Dual pie charts: demo 50/50 and real chart (paste in place of existing plotting block) ---
import plotly.graph_objects as go

# Sidebar toggle to show demo chart
show_demo = st.sidebar.checkbox("Show forced 50/50 demo pie", value=True)

# Defensive checks
if summary is None or summary.empty:
    st.warning("Summary is empty — cannot draw pie chart.")
else:
    if not {"label", "count"}.issubset(set(summary.columns)):
        st.warning("Summary missing required columns 'label' and 'count'. Showing table instead.")
        st.table(summary)
    else:
        # Prepare real summary for plotting
        summary_plot = summary.copy()
        summary_plot["count"] = pd.to_numeric(summary_plot["count"], errors="coerce").fillna(0).astype(int)
        summary_plot = summary_plot[summary_plot["count"] > 0]

        # DEBUG prints for the real chart
        if debug:
            st.write("DEBUG summary_plot (real):", summary_plot)
        real_labels = summary_plot["label"].astype(str).tolist()
        real_values = summary_plot["count"].astype(int).tolist()
        if debug:
            st.write("DEBUG real_labels:", real_labels)
            st.write("DEBUG real_values:", real_values)
            st.write("DEBUG sum(real_values):", sum(real_values))

        # Render demo chart if toggled on
        if show_demo:
            st.markdown("**DEBUG demo: forced 50/50 pie chart**")
            fake_labels = ["answerable", "unanswerable"]
            fake_values = [1, 1]  # equal weights -> 50/50
            if debug:
                st.write("DEBUG fake_labels:", fake_labels)
                st.write("DEBUG fake_values:", fake_values)
                st.write("DEBUG sum(fake_values):", sum(fake_values))
            try:
                fig_demo = go.Figure(
                    go.Pie(labels=fake_labels, values=fake_values, sort=False,
                           marker=dict(colors=px.colors.qualitative.Set2))
                )
                fig_demo.update_traces(textposition="inside", textinfo="percent+label")
                fig_demo.update_layout(title_text="Forced 50/50 Pie Chart Debug Demo")
                st.plotly_chart(fig_demo, use_container_width=True)
            except Exception as e:
                st.error("Demo pie failed: " + str(e))
                st.table(summary_plot)

        # Render the real chart below the demo (or alone if demo is off)
        st.markdown("**Real Answerable vs Unanswerable**")
        if sum(real_values) == 0:
            st.warning("All counts are zero — nothing to plot for the real chart.")
            st.table(summary_plot)
        else:
            try:
                fig_real = go.Figure(
                    go.Pie(labels=real_labels, values=real_values, sort=False,
                           marker=dict(colors=px.colors.qualitative.Set2))
                )
                fig_real.update_traces(textposition="inside", textinfo="percent+label")
                fig_real.update_layout(title_text="Answerable vs Unanswerable (Real Data)")
                st.plotly_chart(fig_real, use_container_width=True)
            except Exception as e:
                st.error("Real pie failed: " + str(e))
                # fallback to bar or table
                try:
                    fig_bar = px.bar(summary_plot, x="label", y="count", title="Answerable vs Unanswerable (bar)")
                    st.plotly_chart(fig_bar, use_container_width=True)
                except Exception:
                    st.table(summary_plot)
# --- end dual pie charts block ---

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


