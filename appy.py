# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from datasets import load_dataset

def load_and_prepare(debug=False):
    # 1) load dataset
    ds = load_dataset("squad_v2")

    # Diagnostics: show dataset metadata
    if debug:
        st.write("datasets version:", __import__("datasets").__version__)
        st.write("pandas version:", pd.__version__)
        st.write("train length:", len(ds["train"]))
        st.write("validation length:", len(ds["validation"]))
        st.write("train sample element (raw):")
        st.json(ds["train"][0])

    # 2) convert using Dataset.to_pandas() for consistent conversion
    def df_from_split(split):
        try:
            df = split.to_pandas()
        except Exception:
            df = pd.DataFrame(split)
        df.columns = [c.strip() for c in df.columns]

        # robust is_answerable computation
        def is_answerable_cell(cell):
            try:
                if isinstance(cell, dict) and "text" in cell:
                    return len(cell["text"]) > 0
                if isinstance(cell, (list, tuple)):
                    if len(cell) == 0:
                        return False
                    first = cell[0]
                    if isinstance(first, dict) and "text" in first:
                        return len(first["text"]) > 0
                    return any(bool(x) for x in cell)
                if isinstance(cell, str):
                    return len(cell.strip()) > 0
            except Exception:
                pass
            return False

        if "answers" in df.columns:
            df["is_answerable"] = df["answers"].apply(is_answerable_cell)
        else:
            st.warning("No 'answers' column found in converted DataFrame; columns: " + ", ".join(df.columns))
            df["is_answerable"] = False

        return df

    train_df = df_from_split(ds["train"])
    valid_df = df_from_split(ds["validation"])
    combined = pd.concat([train_df, valid_df], ignore_index=True)

    # stable summary
    summary = combined["is_answerable"].value_counts().rename_axis("is_answerable").reset_index(name="count")
    summary["label"] = summary["is_answerable"].map({True: "answerable", False: "unanswerable"})

    if debug:
        st.write("Converted combined shape:", combined.shape)
        st.write("Combined columns:", list(combined.columns))
        st.write("Sample converted row:", combined.head(1).to_dict(orient="records"))

    return combined, summary


st.title("SQuAD v2 Answerability Dashboard")
data, summary = load_and_prepare()

st.subheader("Overall proportions")
fig_pie = px.pie(summary, names="label", values="count",
                 title="Answerable vs Unanswerable",
                 color_discrete_sequence=px.colors.qualitative.Set2)
fig_pie.update_traces(textposition="inside", textinfo="percent+label")
st.plotly_chart(fig_pie, use_container_width=True)

st.subheader("Counts by split")
st.write(summary[["label", "count"]].set_index("label"))

st.subheader("Sample unanswerable questions")
sample_unans = data[data["is_answerable"]==False].sample(10, random_state=42)[["id","context","question"]]
st.table(sample_unans.reset_index(drop=True))

st.subheader("Download data")
st.markdown("You can download the combined dataset as CSV for offline analysis.")
csv = data.to_csv(index=False).encode("utf-8")
st.download_button(label="Download CSV", data=csv, file_name="squad_v2_combined.csv", mime="text/csv")
