# GrowthLens

Turn customer data into business insights. GrowthLens is a Streamlit customer
analytics workspace with two modes.

## Modes

### 1. Analyse My Data (requires upload)
For people who are permitted to upload their data.

- Upload a `.csv` or `.xlsx` file, then see total rows/columns. The row preview
  and non-null chart are tucked under an **Inspect Raw Data** expander so the
  reports are reachable immediately.
- **Real standard reports** computed on the Streamlit server with pandas:
  - **Executive Overview** - total events, unique visitors, unique items, date
    span, events per visitor and an event-type breakdown.
  - **Visitor-level funnel** - first `view` -> earliest `cart` *strictly after*
    that view -> earliest `transaction` *strictly after* that cart, per visitor.
  - **Weekly cohort retention** - Monday-start UTC weeks, unique visitor-week
    records, first observed activity as the cohort, and blank (N/A) cells for
    periods a cohort could not have observed yet.
  - **Browsing segmentation** - visitors bucketed by view count (1 / 2-4 / 5+)
    with the ordered-funnel outcomes (reached cart / transaction) per bucket.
- Each report has an interactive Plotly chart and a downloadable CSV table.

A report is only shown when it can actually be computed from the uploaded data;
otherwise the section says so instead of inventing results.

Underneath the reports is **Personal Inquiry Analysis - Ask GrowthLens AI**. It
lets you ask custom business questions about the uploaded dataset instead of
being limited to the fixed reports:

- GrowthLens infers the schema (identifiers, timestamps, events, dimensions,
  measures, currency) and lets you correct the row grain and column roles.
- Example questions adapt to the columns actually present, and are careful:
  identifier columns such as `visitorid`/`itemid` are only ever requested as a
  **distinct count of visitors** (never summed or averaged); average order value
  only appears when a real amount column exists and uses non-null transaction
  amounts; and a channel field that varies within a visitor is described as an
  **event-level** field, not as customer acquisition attribution.
- An optional LLM interprets your question into a **strict JSON plan**; the plan
  is validated against the dataset and then executed **locally with pandas**.
- You get the interpreted question, the proposed metric definitions, the real
  computed table, an appropriate Plotly chart, a business interpretation,
  limitations, and a CSV download.

This is a real analysis workflow, **not a chatbot**: the model never invents
results, never runs code, and never sees raw rows. The AI connection is
currently **not configured**, so this section stays disabled and states that
plainly - a rule-based result is never presented as AI.

Supported operations: grouping, filtering, distinct counts, sums/averages/
medians, event-stage comparisons, ordered funnels, cohort retention, and
association (correlation) - always computed with pandas on your uploaded data.

### Analytical rules enforced
- Event rows are distinguished from unique visitors.
- Sequential funnels require a user ID, timestamps and ordered stages; a stage
  only counts users who reached it *after* the previous stage.
- Identifier columns are never summed or averaged.
- Cohort retention uses first observed activity, Monday-start weeks and blanks
  incomplete follow-up periods (N/A) instead of inventing them.
- Associations are never reported as causation.
- Revenue is never calculated without a numeric amount/currency column.
- If a question cannot be answered, GrowthLens explains why and lists the
  missing fields.
- No LLM-generated code, SQL or shell is ever executed.

### 2. Build an Analysis Blueprint (no upload)
For people who cannot upload their data. Describe your schema instead:

- Dataset name (optional), business objective (required) and row granularity (required).
- Desired time granularity (auto / daily / weekly / monthly) for cohorts and trends.
- Optional funnel stage labels (e.g. `view, cart, purchase`).
- An editable column table: name, data type, unit, role and business meaning
  (dropdowns with explanations; a replaceable example schema is provided).

GrowthLens then returns a short **Analysis Summary**, then four sections
(**Analysis Plan**, **Excel & PivotTable Guide**, **Dashboard Preview**,
**Setup & Checklist**) with transparent, **rule-based** (not AI) recommendations:

- A short, ranked list of relevant analyses (based on your objective and schema).
- Metric definitions (numerator, denominator, required columns).
- Excel 365 formulas using your column names, plus Power Query guidance where a
  simple formula would be misleading.
- A PivotTable configuration (Rows, Columns, Values, Filters, aggregation).
- A blank dashboard blueprint with KPI and chart placeholders, a schematic funnel
  and a blank W0-W8 cohort heatmap (no invented figures).
- Local execution instructions and a validation checklist.

### Analytical correctness
- **Timestamp units and timezone** are honoured: Unix milliseconds/seconds are
  interpreted as UTC; tz-aware timestamps are converted to UTC before grouping.
- **Ordered funnels**: each stage must occur *after* the previous stage for a
  given user; participation ratios are labelled separately from conversion.
- **Cohorts**: Monday-start UTC calendar weeks (W0, W1, ... for weekly),
  deduplicated visitor-week rows, the cohort size as the denominator, and N/A
  (not 0) for periods a recent cohort could not have observed yet.

## Privacy

- Do not upload confidential company data without authorization.
- Your uploaded file is processed on the GrowthLens **Streamlit server**, not on
  your own device, so it leaves your machine when you upload it. Run the app
  locally for sensitive data.
- The **standard reports** run on the Streamlit server and never forward raw
  records to any external AI or LLM API.
- The **Ask GrowthLens AI** section sends nothing to an external model unless
  you tick the consent checkbox. When you do, it sends only your question, a
  **sanitized schema** (column names, types, roles and low-cardinality category
  labels - never raw rows, never values from identifier-like or sensitive
  columns) and the **aggregated results** used for the explanation.
- Without a configured provider, the section shows "AI connection required" and
  runs nothing.
- API keys live in Streamlit Secrets or environment variables, never in source
  files or GitHub.
- The rule-based Blueprint outputs remain explicitly labelled rule-based / non-AI.

## AI configuration (optional)

The personal-inquiry feature needs an OpenAI-compatible chat-completions
endpoint. It is **disabled** until you configure a key. Set these in
`.streamlit/secrets.toml` (preferred) or as environment variables:

```toml
# .streamlit/secrets.toml  - add this file to .gitignore, never commit it
GROWTHLENS_LLM_API_KEY = "sk-..."
GROWTHLENS_LLM_PROVIDER = "openai"          # label only, shown in the UI
GROWTHLENS_LLM_MODEL = "gpt-4o-mini"        # any chat model
GROWTHLENS_LLM_BASE_URL = "https://api.openai.com/v1/chat/completions"
```

Only `GROWTHLENS_LLM_API_KEY` is required; the rest have sensible defaults for
OpenAI-compatible providers. Keys are read at runtime and never written to disk
by GrowthLens.

## Run locally

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Tests

The Blueprint engine and the inquiry engine use only the standard library and
pandas, so they run without Streamlit or a network:

```
python tests/test_blueprint_engine.py      # blueprint rules
python tests/test_analytics_schema.py      # schema inference
python tests/test_analytics_engine.py      # plan validation + pandas execution
python tests/test_standard_reports.py      # overview, funnel, cohort, segmentation
python tests/test_llm.py                   # optional LLM client (mocked, no network)
```

All tests use synthetic data only - including repeated and out-of-order events,
missing columns and epoch timestamps.

## Project layout

| File | Purpose |
| --- | --- |
| `app.py` | Homepage and navigation between modes |
| `data_analysis.py` | Data Analysis mode (upload, preview, reports, inquiry) |
| `standard_reports.py` | Executive Overview, funnel, cohort, segmentation (pandas only) |
| `analytics_schema.py` | Schema/role inference + sanitized LLM payload |
| `analytics_engine.py` | Validates structured plans and computes metrics with pandas |
| `inquiry.py` | "Ask GrowthLens AI" UI section (examples, consent, results) |
| `llm.py` | Optional LLM client (interpret question, explain results) |
| `blueprint.py` | Analysis Blueprint mode (form + results UI) |
| `blueprint_engine.py` | Rule-based Blueprint recommendation engine (stdlib only) |
| `tests/test_blueprint_engine.py` | Unit tests for the Blueprint engine |
| `tests/test_analytics_schema.py` | Unit tests for schema inference |
| `tests/test_analytics_engine.py` | Unit tests for plan validation/execution |
| `tests/test_standard_reports.py` | Unit tests for the standard reports |
| `tests/test_llm.py` | Unit tests for the optional LLM client |
