"""GrowthLens - main entry point.

A two-mode customer analytics workspace:

* Analyse My Data        - upload a CSV/Excel file and run real reports
                           (overview, visitor funnel, cohort retention,
                           segmentation) plus optional personal inquiry.
* Build an Analysis Blueprint - describe your schema, get a plan (no upload).

Navigation is session-state driven, so users can always return Home.
"""

from __future__ import annotations

import streamlit as st

from blueprint import render_blueprint
from data_analysis import render_data_analysis

PAGE_HOME = "home"
PAGE_DATA = "data"
PAGE_BLUEPRINT = "blueprint"


def _go(page):
    st.session_state.page = page
    st.rerun()


def _init_state():
    if "page" not in st.session_state:
        st.session_state.page = PAGE_HOME


def render_home():
    st.markdown(
        "<h1 style='text-align:center;margin-bottom:0;'>From Data to Decisions</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;font-size:1.1rem;color:gray;'>"
        "GrowthLens is a customer analytics workspace.</p>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.subheader("How would you like to start?")

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown("### Analyse My Data")
            st.write(
                "For people who are permitted to upload CSV or Excel files. "
                "Inspect your dataset instantly."
            )
            st.caption("Upload required.")
            if st.button("Analyse My Data", type="primary", width="stretch"):
                _go(PAGE_DATA)

    with right:
        with st.container(border=True):
            st.markdown("### Build an Analysis Blueprint")
            st.write(
                "For people who cannot upload their data. Describe your columns "
                "and get a plan: analyses, Excel formulas and dashboard layout."
            )
            st.caption("No upload required.")
            if st.button("Build an Analysis Blueprint", width="stretch"):
                _go(PAGE_BLUEPRINT)


def render_sidebar():
    with st.sidebar:
        st.markdown("## GrowthLens")
        st.caption("Customer Analytics Workspace")
        if st.button("Home", width="stretch"):
            _go(PAGE_HOME)
        if st.button("Analyse My Data", width="stretch"):
            _go(PAGE_DATA)
        if st.button("Analysis Blueprint", width="stretch"):
            _go(PAGE_BLUEPRINT)
        st.divider()
        st.caption("Upload a CSV/Excel file to run real reports: executive "
                   "overview, visitor funnel, cohort retention and segmentation. "
                   "Or build a rule-based blueprint without uploading.")


def main():
    st.set_page_config(
        page_title="GrowthLens",
        page_icon="\U0001F4CA",
        layout="wide",
    )
    _init_state()
    render_sidebar()

    page = st.session_state.page
    if page == PAGE_DATA:
        render_data_analysis()
    elif page == PAGE_BLUEPRINT:
        render_blueprint()
    else:
        render_home()


if __name__ == "__main__":
    main()
