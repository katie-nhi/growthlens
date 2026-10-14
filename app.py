
import streamlit as st
import pandas as pd

# Set up the website
st.set_page_config(
    page_title="GrowthLens",
    page_icon="📊",
    layout="wide"
)

# Website heading
st.title("GrowthLens")
st.subheader("Customer Analytics Workspace")

st.write(
    "Turn customer data into business insights."
)

# CSV uploader
uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("File uploaded successfully!")

    st.subheader("Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Rows", len(df))

    with col2:
        st.metric("Total Columns", len(df.columns))

    st.subheader("Data Preview")
    st.dataframe(df.head(20))
