"""GrowthLens - Data Analysis mode.

Preserves the original behaviour: upload a CSV or Excel file, see row/column
counts and a 20-row preview, plus a real data-quality chart. On top of that it
renders the real standard reports (Executive Overview, visitor-level funnel,
weekly cohort retention and browsing segmentation) computed on the Streamlit
server with pandas, and the "Personal Inquiry Analysis" section.

No results are fabricated: every number comes from the uploaded file.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

import analytics_schema
import inquiry
import standard_reports

PRIVACY_WARNING = (
    "Privacy: do not upload confidential company data without authorization. Your "
    "uploaded file is processed on the GrowthLens Streamlit server, not on your own "
    "device, so it leaves your machine when you upload it. The standard reports run "
    "on that server and never forward raw records to any external AI or LLM API. "
    "Only the optional Personal Inquiry section can send a sanitized schema and "
    "aggregated results to an AI provider, and only after you give consent. For "
    "sensitive datasets, run GrowthLens locally instead of a public deployment."
)


def _read_uploaded_file(uploaded_file):
    """Read a CSV or Excel upload into a DataFrame. Returns (df, error)."""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file), None
        if name.endswith((".xlsx", ".xlsm")):
            excel = pd.ExcelFile(uploaded_file, engine="openpyxl")
            sheet = excel.sheet_names[0]
            if len(excel.sheet_names) > 1:
                sheet = st.selectbox("Select sheet", excel.sheet_names)
            frame = excel.parse(sheet)
            st.caption("Loaded sheet: " + sheet)
            return frame, None
        return None, "Unsupported file type. Please upload a .csv or .xlsx file."
    except Exception as exc:  # noqa: BLE001 - surface any read error to the user
        return None, "Could not read the file: " + str(exc)


def render_upload_section():
    """Privacy warning + uploader. Returns the DataFrame or None."""
    st.warning(PRIVACY_WARNING, icon="\u26a0\ufe0f")

    uploaded_file = st.file_uploader(
        "Upload your CSV or Excel file",
        type=["csv", "xlsx"],
        help="Supported formats: .csv and .xlsx (first sheet, or choose a sheet).",
    )
    if uploaded_file is None:
        return None

    df, error = _read_uploaded_file(uploaded_file)
    if error:
        st.error(error)
        return None

    st.success("File uploaded successfully!")
    return df


def render_overview(df):
    """Row/column counts - kept from the original app."""
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))


def render_preview(df):
    """First 20 rows - kept from the original app."""
    st.subheader("Data Preview")
    st.dataframe(df.head(20))


def render_data_quality_chart(df):
    """A real, computed chart: non-null counts per column.

    This uses Plotly and only visualises the uploaded data - nothing is invented.
    """
    if df.empty:
        return
    st.subheader("Data Quality Snapshot")
    non_null = df.notna().sum().reset_index()
    non_null.columns = ["column", "non_null_rows"]
    fig = px.bar(
        non_null,
        x="column",
        y="non_null_rows",
        title="Non-null values per column",
        labels={"non_null_rows": "Non-null rows"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Computed directly from your uploaded file.")


def _fig_from_spec(spec, table):
    """Build an interactive Plotly figure from a report's chart spec."""
    kind = spec.get("kind")
    try:
        if kind == "bar":
            return px.bar(table, x=spec["x"], y=spec["y"], title=spec.get("title"))
        if kind == "funnel":
            return px.funnel(table, x=spec["y"], y=spec["x"], title=spec.get("title"))
        if kind == "grouped_bar":
            return px.bar(table, x=spec["x"], y=spec["series"], barmode="group", title=spec.get("title"))
        if kind == "heatmap":
            week_cols = [c for c in table.columns if isinstance(c, str) and c.startswith("W") and c[1:].isdigit()]
            # Use categorical labels taken from the same cohort_start values as the
            # table, and force a category axis so Plotly/JS never parses them as
            # dates (which would shift Monday -> Sunday in negative UTC offsets).
            labels = standard_reports.cohort_heatmap_labels(table)
            matrix = table[week_cols].apply(pd.to_numeric, errors="coerce")
            fig = px.imshow(matrix, x=week_cols, y=labels, text_auto=True, aspect="auto",
                            title=spec.get("title"), color_continuous_scale="Blues")
            fig.update_yaxes(type="category")
            return fig
    except Exception:  # noqa: BLE001 - a chart failure must not break the page
        return None
    return None


def _render_report(report, download_name):
    if not report.get("available"):
        st.info("Not available for this dataset: " + (report.get("reason") or "insufficient columns."))
        return
    metrics = report.get("metrics") or []
    if metrics:
        cols = st.columns(min(len(metrics), 4))
        for i, (label, value) in enumerate(metrics):
            cols[i % len(cols)].metric(label, value)
    table = report.get("table")
    if table is not None and not table.empty:
        fig = _fig_from_spec(report.get("chart") or {}, table)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button(
            "Download table (CSV)",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name=download_name,
            mime="text/csv",
            key="dl_" + download_name,
        )
    for note in report.get("notes") or []:
        st.caption("- " + note)


def render_standard_reports(df):
    """The real advanced reports. Nothing is shown if a report cannot be computed."""
    st.subheader("Advanced Reports")
    reports = standard_reports.run_all(df)

    tabs = st.tabs(["Executive Overview", "Funnel", "Cohort Retention", "Segmentation"])
    with tabs[0]:
        _render_report(reports["overview"], "growthlens_executive_overview.csv")
    with tabs[1]:
        _render_report(reports["funnel"], "growthlens_funnel.csv")
    with tabs[2]:
        _render_report(reports["cohort"], "growthlens_cohort_retention.csv")
    with tabs[3]:
        _render_report(reports["segmentation"], "growthlens_segmentation.csv")


def render_data_analysis():
    """Entry point used by app.py."""
    st.header("Analyse My Data")
    st.write(
        "Upload a CSV or Excel export to inspect it. Standard reports appear first, "
        "then the advanced reports, then you can ask your own custom questions."
    )

    df = render_upload_section()
    if df is not None:
        render_overview(df)
        with st.expander("Inspect Raw Data (preview and non-null counts)"):
            render_preview(df)
            render_data_quality_chart(df)
        render_standard_reports(df)
        inquiry.render_inquiry(df)
    else:
        st.divider()
        st.subheader("Advanced Reports")
        st.info("Upload a CSV or Excel file above to compute the executive overview, "
                "funnel, cohort retention and segmentation reports.")
        st.divider()
        st.subheader("Personal Inquiry Analysis - Ask GrowthLens AI")
        st.info("Upload a CSV or Excel file above to ask custom questions about your data.")
