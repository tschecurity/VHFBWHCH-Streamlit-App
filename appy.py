# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from datasets import load_dataset

@st.cache_data
def load_and_prepare():
    ds = load_dataset("squad_v2")
    def df_from_split(split):
        df = pd.DataFrame(split)
        df["is_answerable"] = df["answers"].apply(lambda a: len(a["text"]) > 0)
        return df
    train_df = df_from_split(ds["train"])
    valid_df = df_from_split(ds["validation"])
    combined = pd.concat([train_df, valid_df], ignore_index=True)
    summary = combined["is_answerable"].value_counts().rename_axis("is_answerable").reset_index(name="count")
    summary["label"] = summary["is_answerable"].map({True: "answerable", False: "unanswerable"})
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
