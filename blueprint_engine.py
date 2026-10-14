"""GrowthLens Analysis Blueprint engine.

Given ONLY a declared schema (column names, data types, units, roles and
meanings) plus a business objective and row granularity, this module produces
transparent, rule-based recommendations: analyses, metric definitions, Excel 365
formulas, PivotTable configurations and a blank dashboard layout.

Design decisions
----------------
* No fabrication: every recommendation is derived from the declared metadata.
  Where required inputs are missing, the blueprint says which inputs are needed
  instead of inventing results.
* No LLM: all logic here is deterministic and rule-based. It is never labelled
  as AI. ``llm.py`` is the (unused) hook for a later phase.
* Standard library only: this file imports nothing beyond Python itself, so it
  can be unit-tested even when pandas/streamlit are not installed.
* Timestamp units matter: Unix milliseconds/seconds are converted to genuine
  Excel Date/Time values before any TEXT/grouping/date comparison.
* Time granularity matters: weekly retention produces relative W0, W1, W2
  cohort steps, monthly retention produces M0, M1, ...
* Ordered funnels matter: each stage must occur AFTER the previous stage for a
  given user; taking each event's global minimum timestamp is wrong.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Controlled vocabularies (kept in one place so the UI and tests agree)
# ---------------------------------------------------------------------------

DATA_TYPES = [
    "identifier",
    "datetime",
    "categorical",
    "numeric",
    "currency",
    "percentage",
    "boolean",
    "text",
    "url",
    "email",
]

ROLES = [
    "identifier",
    "timestamp",
    "event",
    "dimension",
    "measure",
    "group",
    "ignore",
]

ROLE_LABELS = {
    "identifier": "Identifier (unique key)",
    "timestamp": "Timestamp / Date",
    "event": "Event / Stage",
    "dimension": "Dimension (segment)",
    "measure": "Measure (number)",
    "group": "Group / Variant (A/B)",
    "ignore": "Ignore",
}

TIME_GRANULARITIES = ["auto", "daily", "weekly", "monthly"]

EXAMPLE_SCHEMA = [
    {
        "name": "visitorid",
        "data_type": "identifier",
        "unit": "none",
        "role": "identifier",
        "meaning": "Unique visitor / user id",
    },
    {
        "name": "timestamp",
        "data_type": "datetime",
        "unit": "Unix ms",
        "role": "timestamp",
        "meaning": "When the event happened (Unix milliseconds)",
    },
    {
        "name": "event",
        "data_type": "categorical",
        "unit": "none",
        "role": "event",
        "meaning": "Event name, e.g. page_view, signup, purchase",
    },
]


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def normalize_schema(columns):
    """Return a clean list of column dicts with known keys only."""
    clean = []
    for raw in columns or []:
        name = str(raw.get("name", "") or "").strip()
        if not name:
            continue  # skip blank rows created by the editable table
        clean.append({
            "name": name,
            "data_type": str(raw.get("data_type", "text") or "text").strip().lower(),
            "unit": str(raw.get("unit", "") or "").strip(),
            "role": str(raw.get("role", "ignore") or "ignore").strip().lower(),
            "meaning": str(raw.get("meaning", "") or "").strip(),
        })
    return clean


def parse_stages(value):
    """Accept a comma-separated string or a list; return cleaned stage labels."""
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        parts = value
    else:
        parts = str(value).split(",")
    return [str(p).strip() for p in parts if str(p).strip()]


def _first_by_role(schema, role):
    for col in schema:
        if col["role"] == role:
            return col["name"]
    return None


def _first_by_type(schema, data_type):
    for col in schema:
        if col["data_type"] == data_type:
            return col["name"]
    return None


def column_by_name(schema, name):
    for col in schema:
        if col["name"] == name:
            return col
    return None


def id_col(schema):
    return _first_by_role(schema, "identifier") or _first_by_type(schema, "identifier")


def ts_col(schema):
    return _first_by_role(schema, "timestamp") or _first_by_type(schema, "datetime")


def event_col(schema):
    return _first_by_role(schema, "event")


def dim_col(schema):
    return _first_by_role(schema, "dimension")


def measure_col(schema):
    return _first_by_role(schema, "measure")


def group_col(schema):
    return _first_by_role(schema, "group")


def currency_col(schema):
    by_role = _first_by_role(schema, "measure")
    if by_role:
        for col in schema:
            if col["name"] == by_role and col["data_type"] == "currency":
                return by_role
    return _first_by_type(schema, "currency")


# Markers used to detect the declared timestamp unit (order matters: ms first).
_UNIX_MS_MARKERS = (
    "unix ms", "unix_ms", "unixmillis", "unix millis", "epoch ms", "epoch_ms",
    "millisecond", "milliseconds", "ms epoch", "ms since", "millis",
)
_UNIX_S_MARKERS = (
    "unix s", "unix_s", "unix second", "unix seconds", "epoch s", "epoch_s",
    "epoch second", "epoch seconds", "unix timestamp", "unix epoch", "posix",
    "unix time", "seconds since",
)


def timestamp_kind(schema):
    """Classify the declared timestamp unit: unix_ms | unix_s | datetime."""
    ts = ts_col(schema)
    col = column_by_name(schema, ts) if ts else None
    unit = (col["unit"] if col else "").lower()
    if any(marker in unit for marker in _UNIX_MS_MARKERS):
        return "unix_ms"
    if any(marker in unit for marker in _UNIX_S_MARKERS):
        return "unix_s"
    return "datetime"


def timestamp_descriptor(schema):
    """Describe how to turn the declared timestamp into a genuine Excel date.

    Returns a dict with the *Excel-side* column to use in formulas
    (``excel_column``), an optional helper-column formula and Power Query text.
    """
    ts = ts_col(schema)
    kind = timestamp_kind(schema)
    if not ts:
        return {
            "column": None, "excel_column": None, "kind": "none",
            "helper": None, "pq": None,
            "note": "No timestamp column is declared.",
        }
    if kind == "unix_ms":
        converted = ts + "_UTC"
        return {
            "column": ts,
            "excel_column": converted,
            "kind": kind,
            "helper": {
                "label": "Convert Unix ms to Date/Time (helper column)",
                "formula": "=[@%s]/86400000 + DATE(1970,1,1)" % ts,
                "note": (
                    "Add a calculated column named '%s' and set its format to "
                    "Date/Time. Excel dates are UTC (no timezone shift applied)."
                    % converted
                ),
            },
            "pq": (
                "Convert %s (declared Unix ms): add a custom column with "
                "#datetime(1970,1,1,0,0,0) + #duration(0,0,0,[%s]/1000), then set "
                "its type to Date/Time." % (ts, ts)
            ),
            "note": (
                "Declared unit is Unix milliseconds, so convert to a real "
                "Date/Time value before any TEXT/grouping/date comparison."
            ),
        }
    if kind == "unix_s":
        converted = ts + "_UTC"
        return {
            "column": ts,
            "excel_column": converted,
            "kind": kind,
            "helper": {
                "label": "Convert Unix seconds to Date/Time (helper column)",
                "formula": "=[@%s]/86400 + DATE(1970,1,1)" % ts,
                "note": (
                    "Add a calculated column named '%s' and set its format to "
                    "Date/Time." % converted
                ),
            },
            "pq": (
                "Convert %s (declared Unix seconds): add a custom column with "
                "#datetime(1970,1,1,0,0,0) + #duration(0,0,0,[%s]), then set its "
                "type to Date/Time." % (ts, ts)
            ),
            "note": (
                "Declared unit is Unix seconds, so convert to a real Date/Time "
                "value before grouping or formatting."
            ),
        }
    return {
        "column": ts,
        "excel_column": ts,
        "kind": "datetime",
        "helper": None,
        "pq": "Set the data type of [%s] to Date/Time." % ts,
        "note": "Timestamp is already declared as Date/Time.",
    }


def analyze_capabilities(schema):
    """Turn a schema into the boolean capabilities the rules depend on."""
    caps = {
        "identifier": False,
        "timestamp": False,
        "event": False,
        "dimension": False,
        "measure": False,
        "group": False,
        "currency": False,
    }
    for col in schema:
        if col["role"] == "identifier" or col["data_type"] == "identifier":
            caps["identifier"] = True
        if col["role"] == "timestamp" or col["data_type"] == "datetime":
            caps["timestamp"] = True
        if col["role"] == "event":
            caps["event"] = True
        if col["role"] == "dimension":
            caps["dimension"] = True
        if col["role"] == "measure":
            caps["measure"] = True
        if col["role"] == "group":
            caps["group"] = True
        if col["role"] == "measure" and col["data_type"] == "currency":
            caps["currency"] = True
        if col["data_type"] == "currency":
            caps["currency"] = True
    return caps


# ---------------------------------------------------------------------------
# Time granularity helpers
# ---------------------------------------------------------------------------

def resolve_time_granularity(objective, requested="auto"):
    """Resolve the desired period. 'auto' infers from the objective wording."""
    req = (requested or "auto").strip().lower()
    if req in ("daily", "weekly", "monthly"):
        return req
    text = (objective or "").lower()
    if "week" in text:
        return "weekly"
    if "day" in text or "daily" in text:
        return "daily"
    if "month" in text:
        return "monthly"
    return "monthly"


def _granularity_terms(granularity):
    """Return (unit_word, letter, singular)."""
    return {
        "daily": ("day", "D", "day"),
        "weekly": ("week", "W", "week"),
        "monthly": ("month", "M", "month"),
    }.get(granularity, ("month", "M", "month"))


# ---------------------------------------------------------------------------
# Metric + content helpers
# ---------------------------------------------------------------------------

def _metric(name, definition, numerator, denominator, required_columns):
    return {
        "name": name,
        "definition": definition,
        "numerator": numerator,
        "denominator": denominator,
        "required_columns": [c for c in required_columns if c],
    }


TABLE_NOTE = (
    "Format your data range as an Excel Table named 'Data' "
    "(select the range then Ctrl+T, then rename it 'Data' on the Table Design tab)."
)


# ---------------------------------------------------------------------------
# Individual analysis builders
# Each returns a dict consumed by the UI. Builders are only called when their
# requirements are met. ``ctx`` carries optional UI choices (granularity, stages).
# ---------------------------------------------------------------------------

def build_funnel(schema, ctx=None):
    ctx = ctx or {}
    ident = id_col(schema)
    event = event_col(schema)
    desc = timestamp_descriptor(schema)
    tsx = desc["excel_column"]

    stages = parse_stages(ctx.get("funnel_stages"))
    if len(stages) < 2:
        stages = ["<Step1Event>", "<Step2Event>", "<Step3Event>"]

    # One isolated "time" column per stage, so ordered MINIFS can compare them.
    time_cols = ["step%d_time" % (i + 1) for i in range(len(stages))]
    formulas = []
    if desc.get("helper"):
        formulas.append(desc["helper"])
    formulas.append({
        "label": "Event participation ratio per step (NOT conversion)",
        "formula": '=COUNTA(UNIQUE(FILTER(Data[%s], Data[%s]="<StepEvent>")))' % (ident, event),
        "note": (
            "Distinct users who ever performed the event. Ignores order - use the "
            "ordered steps below for a real funnel."
        ),
    })
    for i, stage in enumerate(stages):
        literal = '"%s"' % stage
        formulas.append({
            "label": "Stage %d time column (isolate %s)" % (i + 1, stage),
            "formula": "=IF([@%s]=%s, [@%s], BLANK())" % (event, literal, tsx),
            "note": (
                "Add a calculated column named '%s'. It holds the timestamp only "
                "for rows of this stage, blank otherwise." % time_cols[i]
            ),
        })

    # Ordered per-user summary sheet (one row per user).
    formulas.append({
        "label": "Unique users (summary sheet, column A)",
        "formula": "=UNIQUE(Data[%s])" % ident,
        "note": "Paste into a new sheet; one row per user, header in A1.",
    })
    prev_cells = []
    for i in range(len(stages)):
        cell = chr(ord("B") + i)  # B, C, D, ...
        if i == 0:
            formula = (
                '=IF(MINIFS(Data[{t}], Data[{id}], $A2)=0,"",'
                'MINIFS(Data[{t}], Data[{id}], $A2))'
            ).format(t=time_cols[i], id=ident)
            note = (
                "Column %s. Users who ever performed '%s' (their first such time)."
                % (cell, stages[i])
            )
        else:
            prev = prev_cells[-1]
            formula = (
                '=IF(MINIFS(Data[{t}], Data[{id}], $A2, Data[{t}], ">"&{pc})=0,"",'
                'MINIFS(Data[{t}], Data[{id}], $A2, Data[{t}], ">"&{pc}))'
            ).format(t=time_cols[i], id=ident, pc=prev + "2")
            note = (
                "Column %s. First '%s' that occurs AFTER step %d (cell %s). "
                "This is the ordered step, not the global minimum." % (
                    cell, stages[i], i, prev + "2"
                )
            )
        prev_cells.append(cell)
        formulas.append({
            "label": "Ordered step %d - first %s after previous step" % (i + 1, stages[i]),
            "formula": formula,
            "note": note,
        })

    # Conversion counts on the summary sheet.
    count_formulas = []
    for i in range(len(stages)):
        cell = chr(ord("B") + i)
        count_formulas.append({
            "label": "Users reaching ordered step %d" % (i + 1),
            "formula": "=COUNT(%s:%s)" % (cell, cell),
            "note": "COUNT ignores the blank cells left by users who did not reach this step.",
        })
    if len(stages) >= 2:
        count_formulas.append({
            "label": "Step 2 -> step 1 conversion (ordered)",
            "formula": "=IFERROR(COUNT(C:C)/COUNT(B:B),0)",
            "note": "Format as a percentage. Both counts are ordered steps.",
        })
        last = chr(ord("B") + len(stages) - 1)
        count_formulas.append({
            "label": "Overall ordered funnel conversion",
            "formula": "=IFERROR(COUNT(%s:%s)/COUNT(B:B),0)" % (last, last),
            "note": "Reached final ordered step / reached first step.",
        })
    formulas += count_formulas

    return {
        "key": "funnel",
        "title": "Event Funnel (sequential conversion)",
        "why": (
            "Your schema declares a user identifier, an event/stage column and a "
            "timestamp, so a TRUE ordered funnel is possible. A step only counts "
            "when it happens AFTER the previous step for the same user."
        ),
        "metrics": [
            _metric(
                "Event participation ratio (NOT conversion)",
                "Share of users who triggered a given event at any time. This does "
                "NOT require the events to happen in order and is not a funnel step.",
                "COUNT(DISTINCT users who performed the event)",
                "COUNT(DISTINCT users in the data)",
                [ident, event],
            ),
            _metric(
                "Ordered step conversion",
                "Share of users who reached step N AFTER completing step N-1 earlier.",
                "users with a step-N time later than their step-(N-1) time",
                "users who reached step N-1 in order",
                [ident, event, desc["column"]],
            ),
            _metric(
                "Overall ordered funnel conversion",
                "Share of first-step users who reached the final step in order.",
                "users reaching the final ordered step",
                "users reaching the first step",
                [ident, event, desc["column"]],
            ),
        ],
        "formulas": formulas,
        "power_query": [
            "Load the data as a table: Data > Get & Transform Data > From Table/Range (name it Data).",
            desc.get("pq") or "Set [%s] to Date/Time." % desc["column"],
            "Sort by [%s] ascending, then by [%s] ascending." % (ident, desc["column"]),
            "Group by [%s] with 'All Rows'." % ident,
            "For each user compute the first '%s' time." % stages[0],
            "Then compute the first '%s' time that is GREATER than the first-step time (per user)." % stages[1],
        ] + ([
            "Continue: each later stage uses the previous ordered step's time as its floor.",
        ] if len(stages) > 2 else []) + [
            "Users whose later stage has no time after the previous step are NOT counted as converted.",
            "Close & Load and chart the ordered step counts.",
        ],
        "pivot": {
            "rows": ["Ordered step (from the summary sheet)"],
            "columns": [],
            "values": ["Count of users reaching each ordered step"],
            "filters": [],
            "aggregation": "Count of non-blank ordered-step cells per stage",
        },
        "notes": [
            "Participation ratio counts any user who ever triggered the event; it does NOT enforce order.",
            "Never take each event's GLOBAL minimum timestamp: each stage must occur AFTER the previous stage for that user.",
            "MINIFS returns 0 when no row matches; the IF(...=0,\"\") guard treats that as 'step not reached'.",
            "Ordering is only meaningful when the timestamp is a real Date/Time value (see the conversion helper).",
        ],
        "stages": stages,
    }


def build_cohort(schema, ctx=None):
    ctx = ctx or {}
    ident = id_col(schema)
    desc = timestamp_descriptor(schema)
    tsx = desc["excel_column"]
    granularity = resolve_time_granularity(ctx.get("objective", ""), ctx.get("time_granularity", "auto"))
    unit_word, letter, _ = _granularity_terms(granularity)
    index_field = {"weekly": "WeekIndex", "monthly": "MonthIndex", "daily": "DayIndex"}[granularity]

    formulas = []
    if desc.get("helper"):
        formulas.append(desc["helper"])
    formulas.append({
        "label": "First observed activity per user (calculated column)",
        "formula": "=MINIFS(Data[%s], Data[%s], [@%s])" % (tsx, ident, ident),
        "note": "Earliest timestamp IN THIS DATASET for the row's user. Not necessarily a true signup.",
    })

    if granularity == "weekly":
        formulas += [
            {
                "label": "Cohort week start (calculated column)",
                "formula": "=[@FirstObserved]-WEEKDAY([@FirstObserved],2)+1",
                "note": "Rounds the first observed date back to Monday (WEEKDAY type 2). Name it CohortWeek.",
            },
            {
                "label": "Activity week start (calculated column)",
                "formula": "=[@%s]-WEEKDAY([@%s],2)+1" % (tsx, tsx),
                "note": "Rounds the activity date back to Monday. Name it ActivityWeek.",
            },
            {
                "label": "Relative week index W0, W1, W2 ... (calculated column)",
                "formula": "=INT(([@%s]-[@FirstObserved])/7)" % tsx,
                "note": "0 = the user's first week (W0), 1 = the following week (W1), and so on.",
            },
        ]
        cohort_field = "CohortWeek (week start)"
        relative_field = "WeekIndex"
    elif granularity == "daily":
        formulas += [
            {
                "label": "Cohort day (calculated column)",
                "formula": "=[@FirstObserved]",
                "note": "The user's first observed day. Name it CohortDay.",
            },
            {
                "label": "Relative day index D0, D1, ... (calculated column)",
                "formula": "=[@%s]-[@FirstObserved]" % tsx,
                "note": "Whole days since first observation.",
            },
        ]
        cohort_field = "CohortDay"
        relative_field = "DayIndex"
    else:  # monthly
        formulas += [
            {
                "label": "Cohort month (calculated column)",
                "formula": '=TEXT([@FirstObserved], "yyyy-mm")',
                "note": "Groups each user by their first observed month. Name it CohortMonth.",
            },
            {
                "label": "Activity month (calculated column)",
                "formula": '=TEXT([@%s], "yyyy-mm")' % tsx,
                "note": "The month of the current activity. Name it ActivityMonth.",
            },
            {
                "label": "Relative month index M0, M1, ... (calculated column)",
                "formula": "=(YEAR([@%s])*12+MONTH([@%s]))-(YEAR([@FirstObserved])*12+MONTH([@FirstObserved]))" % (tsx, tsx),
                "note": "0 = the first month (M0), 1 = the next month (M1), and so on.",
            },
        ]
        cohort_field = "CohortMonth"
        relative_field = "MonthIndex"

    return {
        "key": "cohort",
        "title": "Cohort Retention (%s)" % granularity,
        "why": (
            "Your schema has a stable user identifier and timestamps, so users can "
            "be grouped by first observed activity and tracked in relative %s "
            "steps (W0/W1/... for weekly)." % unit_word
        ),
        "metrics": [
            _metric(
                "Cohort size (the denominator)",
                "Distinct users whose first observed period is this cohort's period "
                "(index 0). This is the base for every retention percentage.",
                "COUNT(DISTINCT users with %s = 0 in the cohort)" % index_field,
                "n/a (it IS the denominator)",
                [ident, desc["column"]],
            ),
            _metric(
                "Retained users in %s N" % unit_word,
                "Distinct users from the cohort active again N %ss after their first period." % unit_word,
                "COUNT(DISTINCT users with %s = N) for that cohort" % index_field,
                "cohort size",
                [ident, desc["column"]],
            ),
            _metric(
                "Retention rate %s N" % unit_word,
                "Retention percentage for relative %s N." % unit_word,
                "retained users in %s N" % unit_word,
                "cohort size",
                [ident, desc["column"]],
            ),
        ],
        "formulas": formulas,
        "power_query": [
            desc.get("pq") or "Set [%s] to Date/Time." % desc["column"],
            "Group by [%s] and take the minimum [%s] to get each user's first observed date." % (ident, desc["column"]),
            "Merge that back so every row carries the user's first observed date.",
            "Add the relative %s index (%s) to every row." % (unit_word, index_field),
            "DEDUPLICATE user-%s records: remove duplicates on (%s, %s) so a user counts once per %s even with many events." % (unit_word, ident, index_field, unit_word),
            "Pivot: rows = cohort period, columns = %s (0..8 = W0..W8), values = count of users." % index_field,
        ],
        "pivot": {
            "rows": [cohort_field],
            "columns": ["%s (0..8, shown as %s0-%s8)" % (relative_field, letter, letter)],
            "values": ["Count of distinct " + ident],
            "filters": [],
            "aggregation": "Distinct Count (requires the Data Model) of DEDUPLICATED user-%s rows" % unit_word,
        },
        "notes": [
            "Deduplicate to one row per user per %s before counting, otherwise heavy users inflate retention." % unit_word,
            "The denominator is the cohort size (distinct users whose first period index = 0), not total users across all cohorts.",
            "A cohort built from this data uses first OBSERVED activity; if the data starts mid-history this is not a proven first-ever signup.",
            "Incomplete observation: recent cohorts have fewer observable %ss, so far-right cells are blank (right-censored), NOT zero." % unit_word,
        ],
    }


def build_segmentation(schema, ctx=None):
    dim = dim_col(schema)
    measure = measure_col(schema)
    values = ["Count of rows"]
    if measure:
        values = ["Count of rows", "Sum of " + measure, "Average of " + measure]
    formulas = [
        {
            "label": "Count rows per segment",
            "formula": '=COUNTIFS(Data[%s], "<SegmentValue>")' % dim,
            "note": "Replace <SegmentValue> with each segment label.",
        },
    ]
    if measure:
        formulas += [
            {
                "label": "Sum measure per segment",
                "formula": '=SUMIFS(Data[%s], Data[%s], "<SegmentValue>")' % (measure, dim),
                "note": "Total of the measure for one segment.",
            },
            {
                "label": "Average measure per segment",
                "formula": '=AVERAGEIFS(Data[%s], Data[%s], "<SegmentValue>")' % (measure, dim),
                "note": "Average of the measure for one segment.",
            },
        ]
    return {
        "key": "segmentation",
        "title": "Segmentation Breakdown",
        "why": (
            "Your schema declares a dimension (a way to split the data) "
            + ("and a numeric measure, so segments can be compared." if measure
               else "so rows can be split into segments.")
        ),
        "metrics": [
            _metric(
                "Segment size",
                "Number of rows (or users) in each segment.",
                "COUNT of rows in the segment",
                "total rows",
                [dim],
            ),
            (
                _metric(
                    "Segment measure",
                    "Aggregated numeric measure per segment.",
                    "SUM of " + measure,
                    "COUNT of rows in the segment (for averages)",
                    [dim, measure],
                )
                if measure else None
            ),
        ],
        "formulas": formulas,
        "power_query": [],
        "pivot": {
            "rows": [dim],
            "columns": [],
            "values": values,
            "filters": [],
            "aggregation": "Sum / Count / Average (choose per value field)",
        },
        "notes": [
            "Segmentation compares groups; it does not by itself prove WHY a segment behaves differently.",
        ],
    }


def build_timeseries(schema, ctx=None):
    ctx = ctx or {}
    desc = timestamp_descriptor(schema)
    tsx = desc["excel_column"]
    measure = measure_col(schema)
    granularity = resolve_time_granularity(ctx.get("objective", ""), ctx.get("time_granularity", "auto"))
    unit_word, _letter, _ = _granularity_terms(granularity)
    values = ["Count of rows"] + (["Sum of " + measure] if measure else [])
    formulas = []
    if desc.get("helper"):
        formulas.append(desc["helper"])
    formulas.append({
        "label": "Rows in a month (example: Jan 2024)",
        "formula": "=SUMPRODUCT((Data[%s]>=DATE(2024,1,1))*(Data[%s]<DATE(2024,2,1)))" % (tsx, tsx),
        "note": "Change the dates for other periods. A PivotTable (below) is easier to maintain.",
    })
    return {
        "key": "timeseries",
        "title": "Time Series / Trend",
        "why": "Your schema includes a timestamp, so activity can be tracked over time.",
        "metrics": [
            _metric(
                "Period volume",
                "Number of rows per %s." % unit_word,
                "COUNT of rows in the period",
                "n/a",
                [desc["column"]],
            ),
        ],
        "formulas": formulas,
        "power_query": [
            "A PivotTable with dates grouped by %s is usually enough; no Power Query needed." % unit_word.capitalize(),
        ],
        "pivot": {
            "rows": [desc["column"]],
            "columns": [],
            "values": values,
            "filters": [],
            "aggregation": "Right-click a date in Rows > Group, choose %ss / Years" % unit_word.capitalize(),
        },
        "notes": [
            "Make sure the timestamp column is a real Date/Time type, not text.",
        ],
    }


def build_conversion(schema, ctx=None):
    ident = id_col(schema)
    event = event_col(schema)
    return {
        "key": "conversion",
        "title": "Conversion Rate (two stages)",
        "why": (
            "Your schema has a user identifier and an event/stage column, so a "
            "conversion rate between two stages can be computed on distinct users."
        ),
        "metrics": [
            _metric(
                "Two-stage conversion rate",
                "Share of users who reached the target stage. This is a "
                "participation ratio unless a timestamp is used to enforce order.",
                "COUNT(DISTINCT users at target stage)",
                "COUNT(DISTINCT users at the starting stage)",
                [ident, event],
            ),
        ],
        "formulas": [
            {
                "label": "Distinct users at the starting stage",
                "formula": '=COUNTA(UNIQUE(FILTER(Data[%s], Data[%s]="<StartEvent>")))' % (ident, event),
                "note": "Replace <StartEvent>.",
            },
            {
                "label": "Distinct users at the target stage",
                "formula": '=COUNTA(UNIQUE(FILTER(Data[%s], Data[%s]="<TargetEvent>")))' % (ident, event),
                "note": "Replace <TargetEvent>. This ignores order between the two stages.",
            },
            {
                "label": "Two-stage conversion rate",
                "formula": (
                    '=COUNTA(UNIQUE(FILTER(Data[{id}], Data[{ev}]="<TargetEvent>")))'
                    ' / COUNTA(UNIQUE(FILTER(Data[{id}], Data[{ev}]="<StartEvent>")))'
                ).format(id=ident, ev=event),
                "note": "Format the cell as a percentage.",
            },
        ],
        "power_query": [],
        "pivot": {
            "rows": [event],
            "columns": [],
            "values": ["Count of distinct " + ident],
            "filters": [],
            "aggregation": "Distinct Count (requires the Data Model)",
        },
        "notes": [
            "This is a two-stage ratio. It does NOT enforce that the target happened after the start.",
            "For ordered, multi-step conversion use the Event Funnel analysis instead.",
        ],
    }


def build_ab_test(schema, ctx=None):
    group = group_col(schema)
    measure = measure_col(schema)
    return {
        "key": "ab_test",
        "title": "A/B Test Comparison",
        "why": (
            "Your schema declares a group/variant column and a numeric measure, so "
            "the average outcome of each variant can be compared."
        ),
        "metrics": [
            _metric(
                "Variant average",
                "Average measure value per variant.",
                "SUM of " + measure + " within the variant",
                "COUNT of rows within the variant",
                [group, measure],
            ),
            _metric(
                "Relative lift",
                "How much better one variant is versus the other.",
                "(average B - average A)",
                "average A",
                [group, measure],
            ),
        ],
        "formulas": [
            {
                "label": "Average for variant A",
                "formula": '=AVERAGEIFS(Data[%s], Data[%s], "A")' % (measure, group),
                "note": "Replace \"A\" with the real variant label.",
            },
            {
                "label": "Average for variant B",
                "formula": '=AVERAGEIFS(Data[%s], Data[%s], "B")' % (measure, group),
                "note": "Replace \"B\" with the real variant label.",
            },
            {
                "label": "Relative lift (B vs A)",
                "formula": '=(AVERAGEIFS(Data[%s], Data[%s], "B")-AVERAGEIFS(Data[%s], Data[%s], "A"))/AVERAGEIFS(Data[%s], Data[%s], "A")'
                % (measure, group, measure, group, measure, group),
                "note": "Format as a percentage. A single lift number is not proof of significance.",
            },
        ],
        "power_query": [],
        "pivot": {
            "rows": [group],
            "columns": [],
            "values": ["Average of " + measure, "Count of rows"],
            "filters": [],
            "aggregation": "Average / Count",
        },
        "notes": [
            "Comparing averages does not establish statistical significance or causality.",
        ],
    }


def build_revenue(schema, ctx=None):
    currency = currency_col(schema)
    ident = id_col(schema)
    desc = timestamp_descriptor(schema)
    metrics = [
        _metric(
            "Total revenue",
            "Sum of the monetary measure.",
            "SUM of " + currency,
            "n/a",
            [currency],
        ),
    ]
    formulas = [
        {
            "label": "Total revenue",
            "formula": "=SUM(Data[%s])" % currency,
            "note": "Format as currency.",
        },
    ]
    if ident:
        metrics.append(_metric(
            "ARPU",
            "Average revenue per user.",
            "SUM of " + currency,
            "COUNT(DISTINCT users)",
            [currency, ident],
        ))
        formulas.append({
            "label": "ARPU",
            "formula": "=SUM(Data[%s])/COUNTA(UNIQUE(Data[%s]))" % (currency, ident),
            "note": "Format as currency.",
        })
    return {
        "key": "revenue",
        "title": "Monetary Metrics (Revenue / ARPU)",
        "why": "Your schema declares a currency measure, so monetary metrics are valid.",
        "metrics": metrics,
        "formulas": formulas,
        "power_query": [],
        "pivot": {
            "rows": [desc["column"]] if desc["column"] else [],
            "columns": [],
            "values": ["Sum of " + currency, "Count of rows"],
            "filters": [],
            "aggregation": "Sum / Count",
        },
        "notes": [
            "Only monetary metrics requested here are included. ROAS additionally "
            "needs a separate 'spend' or 'cost' column, which is not declared.",
        ],
    }


def build_topn(schema, ctx=None):
    dim = dim_col(schema)
    measure = measure_col(schema)
    return {
        "key": "topn",
        "title": "Top-N Ranking",
        "why": "Your schema has a dimension and a measure, so items can be ranked by the measure.",
        "metrics": [
            _metric(
                "Ranked list",
                "Dimension values ordered by total measure.",
                "SUM of " + measure + " per dimension value",
                "n/a",
                [dim, measure],
            ),
        ],
        "formulas": [
            {
                "label": "Total measure for one item",
                "formula": '=SUMIFS(Data[%s], Data[%s], "<Item>")' % (measure, dim),
                "note": "Then use a PivotTable sorted descending for the full ranking.",
            },
        ],
        "power_query": [],
        "pivot": {
            "rows": [dim],
            "columns": [],
            "values": ["Sum of " + measure],
            "filters": [],
            "aggregation": "Sum, then sort the row labels descending",
        },
        "notes": [],
    }


def build_active_users(schema, ctx=None):
    ctx = ctx or {}
    ident = id_col(schema)
    desc = timestamp_descriptor(schema)
    tsx = desc["excel_column"]
    granularity = resolve_time_granularity(ctx.get("objective", ""), ctx.get("time_granularity", "auto"))
    unit_word, _letter, _ = _granularity_terms(granularity)
    formulas = []
    if desc.get("helper"):
        formulas.append(desc["helper"])
    formulas.append({
        "label": "Distinct active users since a date (example)",
        "formula": '=COUNTA(UNIQUE(FILTER(Data[%s], Data[%s]>=DATE(2024,1,1))))' % (ident, tsx),
        "note": "Adjust the date filter. PivotTables are easier for many periods.",
    })
    return {
        "key": "active_users",
        "title": "Active Users (DAU / MAU)",
        "why": (
            "Your schema has a user identifier and timestamps, so distinct active "
            "users per period can be counted."
        ),
        "metrics": [
            _metric(
                "Active users per period",
                "Number of distinct users active in a %s." % unit_word,
                "COUNT(DISTINCT users with activity in the period)",
                "n/a",
                [ident, desc["column"]],
            ),
            _metric(
                "Stickiness (DAU / MAU)",
                "How frequently the average monthly user returns.",
                "average daily active users",
                "monthly active users",
                [ident, desc["column"]],
            ),
        ],
        "formulas": formulas,
        "power_query": [
            "A PivotTable with the timestamp grouped by %s and value = Distinct Count of users is usually enough." % unit_word.capitalize(),
        ],
        "pivot": {
            "rows": [desc["column"]],
            "columns": [],
            "values": ["Count of distinct " + ident],
            "filters": [],
            "aggregation": "Distinct Count (requires the Data Model); group dates by %s" % unit_word.capitalize(),
        },
        "notes": [],
    }


# ---------------------------------------------------------------------------
# Analysis registry: requirements + objective keywords
# ---------------------------------------------------------------------------

ANALYSIS_DEFS = {
    "funnel": {
        "requires": ["identifier", "timestamp", "event"],
        "build": build_funnel,
        "keywords": [
            "funnel", "drop-off", "drop off", "dropoff", "conversion", "signup",
            "sign-up", "onboarding", "checkout", "step", "stage", "journey",
            "path", "registration", "flow", "cart", "purchase",
        ],
    },
    "cohort": {
        "requires": ["identifier", "timestamp"],
        "build": build_cohort,
        "keywords": [
            "retention", "cohort", "retain", "churn", "repeat", "returning",
            "stickiness", "lifetime", "ltv", "recurring", "week", "month",
        ],
    },
    "active_users": {
        "requires": ["identifier", "timestamp"],
        "build": build_active_users,
        "keywords": [
            "engagement", "active", "dau", "mau", "wau", "usage", "daily active",
            "monthly active", "weekly active",
        ],
    },
    "conversion": {
        "requires": ["identifier", "event"],
        "build": build_conversion,
        "keywords": ["conversion rate", "convert", "conversion", "signup rate"],
    },
    "segmentation": {
        "requires": ["dimension"],
        "build": build_segmentation,
        "keywords": [
            "segment", "segmentation", "breakdown", "break down", "split",
            "by country", "by channel", "by device", "group by", "compare segments",
        ],
    },
    "timeseries": {
        "requires": ["timestamp"],
        "build": build_timeseries,
        "keywords": [
            "trend", "over time", "monthly", "daily", "weekly", "growth",
            "seasonality", "time series", "timeline", "by month", "by day",
        ],
    },
    "ab_test": {
        "requires": ["group", "measure"],
        "build": build_ab_test,
        "keywords": ["a/b", "ab test", "experiment", "variant", "control", "split test", "a-b"],
    },
    "revenue": {
        "requires": ["currency"],
        "build": build_revenue,
        "keywords": [
            "revenue", "sales", "roas", "arpu", "aov", "monetary", "spend",
            "gmv", "mrr", "arr", "income", "profit",
        ],
    },
    "topn": {
        "requires": ["dimension", "measure"],
        "build": build_topn,
        "keywords": [
            "top", "best", "worst", "ranking", "rank", "most", "highest",
            "lowest", "leaderboard",
        ],
    },
}


def _keyword_score(objective, keywords):
    text = (objective or "").lower()
    return sum(2 for kw in keywords if kw in text)


def recommend_analyses(schema, objective, granularity=""):
    """Return (chosen_keys, blocked) based on capabilities + objective.

    * chosen_keys: up to 4 fully-supported analyses, ranked by objective fit.
    * blocked: analyses the objective implies but the schema cannot support,
      with the missing inputs listed.
    """
    caps = analyze_capabilities(schema)
    supported = []
    blocked = []
    for key, definition in ANALYSIS_DEFS.items():
        missing = [req for req in definition["requires"] if not caps.get(req)]
        score = _keyword_score(objective, definition["keywords"])
        if not missing:
            supported.append((score, key))
        elif score > 0:
            blocked.append({
                "key": key,
                "title": _analysis_title(key),
                "missing": missing,
            })

    supported.sort(key=lambda pair: (-pair[0], pair[1]))

    positively_matched = [key for score, key in supported if score > 0]
    chosen = positively_matched[:4]
    if not chosen:
        # Objective matched nothing: fall back to the few best-supported
        # analyses rather than returning a generic list of everything.
        chosen = [key for _score, key in supported][:3]
    return chosen, blocked


def _analysis_title(key):
    builder = ANALYSIS_DEFS[key]["build"]
    try:
        sample = [
            {"name": "x_id", "data_type": "identifier", "role": "identifier"},
            {"name": "x_time", "data_type": "datetime", "role": "timestamp", "unit": "datetime"},
            {"name": "x_event", "data_type": "categorical", "role": "event"},
            {"name": "x_dim", "data_type": "categorical", "role": "dimension"},
            {"name": "x_measure", "data_type": "numeric", "role": "measure"},
            {"name": "x_group", "data_type": "text", "role": "group"},
            {"name": "x_money", "data_type": "currency", "role": "measure"},
        ]
        return builder(normalize_schema(sample), {})["title"]
    except Exception:
        return key.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Dashboard blueprint, instructions, checklist
# ---------------------------------------------------------------------------

_READINESS_ROLES = [
    ("identifier", "a stable user identifier"),
    ("timestamp", "timestamps"),
    ("event", "event / stage names"),
    ("dimension", "segments / dimensions"),
    ("measure", "numeric measures"),
    ("group", "A/B groups"),
    ("currency", "a currency field"),
]


def build_summary(schema, analyses, blocked):
    """A short, top-of-page summary: readiness, top analyses, missing inputs."""
    caps = analyze_capabilities(schema)
    present = [label for role, label in _READINESS_ROLES if caps.get(role)]
    missing_roles = [label for role, label in _READINESS_ROLES if not caps.get(role)]
    return {
        "readiness_present": present,
        "readiness_missing": missing_roles,
        "top": [{"title": a["title"], "why": a["why"]} for a in analyses[:3]],
        "missing": [{"title": b["title"], "missing": b["missing"]} for b in blocked],
    }


def build_dashboard_blueprint(schema, analyses, stages=None, granularity="monthly"):
    """Blank KPI + chart layout. Numbers are NEVER invented (placeholders only)."""
    keys = {a["key"] for a in analyses}
    ident = id_col(schema)
    event = event_col(schema)
    desc = timestamp_descriptor(schema)
    ts = desc["column"]
    dim = dim_col(schema)
    measure = measure_col(schema)
    currency = currency_col(schema)
    stages = stages or ["Stage 1", "Stage 2", "Stage 3"]
    unit_word, letter, _ = _granularity_terms(granularity)

    kpis = [{"label": "Total rows", "placeholder": "?", "definition": "COUNT of data rows"}]
    charts = []

    if "funnel" in keys:
        kpis.append({
            "label": "Ordered funnel conversion",
            "placeholder": "?",
            "definition": "Final ordered step users / first step users",
        })
        charts.append({
            "chart_type": "Funnel (schematic)",
            "title": "Funnel steps",
            "axis": "Stages: " + " -> ".join(stages) + " (Y = distinct users)",
            "note": "Schematic stage layout only - widths and values are NOT data.",
            "stages": stages,
        })
    if "cohort" in keys:
        charts.append({
            "chart_type": "Heatmap (blank)",
            "title": "Cohort retention matrix",
            "axis": "Rows = cohort %s, Columns = relative %s index (W0-W8)" % (unit_word, unit_word),
            "note": "Cells stay blank until real data is loaded; far-right cells are right-censored.",
            "granularity": granularity,
            "letter": letter,
        })
    if "active_users" in keys:
        kpis.append({
            "label": "Active users",
            "placeholder": "?",
            "definition": "COUNT(DISTINCT users) per %s" % unit_word,
        })
        charts.append({
            "chart_type": "Line (blank)",
            "title": "Active users over time",
            "axis": "X = period, Y = distinct active users",
            "note": "Needs chronological grouping.",
        })
    if "conversion" in keys:
        kpis.append({
            "label": "Conversion rate",
            "placeholder": "?",
            "definition": "Target-stage users / start-stage users",
        })
    if "segmentation" in keys:
        charts.append({
            "chart_type": "Bar (blank)",
            "title": "Segment comparison",
            "axis": "X = " + (dim or "segment") + ", Y = " + (measure or "count"),
            "note": "One bar per segment value.",
        })
    if "timeseries" in keys:
        charts.append({
            "chart_type": "Line (blank)",
            "title": "Volume trend",
            "axis": "X = " + (ts or "date") + " grouped by %s, Y = count" % unit_word,
            "note": "Group dates by %s." % unit_word,
        })
    if "revenue" in keys:
        kpis.append({
            "label": "Total revenue",
            "placeholder": "?",
            "definition": "SUM of " + (currency or "currency"),
        })
    if ident:
        kpis.append({
            "label": "Distinct users",
            "placeholder": "?",
            "definition": "COUNT(DISTINCT " + ident + ")",
        })

    return {"kpis": kpis, "charts": charts}


def build_instructions(schema, granularity="monthly"):
    desc = timestamp_descriptor(schema)
    unit_word, _letter, _ = _granularity_terms(granularity)
    steps = [
        "Import the file into Excel 365 (Data > Get Data > From Text/CSV or From Excel Workbook).",
        "Format the imported range as a Table named 'Data' (Ctrl+T, then rename on Table Design).",
    ]
    if desc["kind"] in ("unix_ms", "unix_s"):
        steps.append(
            "Convert the Unix timestamp to Date/Time using the conversion helper column before grouping (%s)."
            % desc["note"]
        )
    else:
        steps.append("Make sure the timestamp column is typed as Date/Time (not text).")
    steps += [
        "Add calculated helper columns where the blueprint provides a formula (see each analysis).",
        "For ordered funnels and cohorts, use the Power Query steps - a single formula is not reliable.",
        "Insert a PivotTable for each analysis using the Rows/Columns/Values/Filters given, grouped by %s." % unit_word,
        "For 'Distinct Count', add the table to the Data Model when creating the PivotTable.",
        "Build the dashboard by placing the KPI cells and charts on a new sheet; link them to the PivotTables.",
    ]
    return steps


def build_checklist(schema, analyses):
    checklist = [
        "Every required column in your declared schema actually exists in the file, spelled the same way.",
        "Row granularity is honoured: confirm what one row represents before trusting any count.",
        "The timestamp column is a real Date/Time type (convert Unix ms/seconds first) and uses one consistent timezone.",
        "The user identifier is stable over time (not re-created on each visit).",
    ]
    keys = {a["key"] for a in analyses}
    if "funnel" in keys:
        checklist.append(
            "Funnel steps are checked in chronological order per user (each stage AFTER the previous), not by raw event counts."
        )
    if "cohort" in keys:
        checklist += [
            "Cohort user-period rows are deduplicated (one row per user per week/month) before counting.",
            "The retention denominator is the cohort size (first-period users), not total users.",
            "Far-right cohort cells are left blank for incomplete (right-censored) observation periods.",
        ]
    if "revenue" in keys:
        checklist.append("Revenue figures come from a real currency column, not a proxy.")
    checklist.append("Cross-check one PivotTable figure by hand (filter to one user or period).")
    return checklist


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_blueprint(columns, objective, granularity, dataset_name="",
                       time_granularity="auto", funnel_stages=None):
    """Produce a complete rule-based blueprint dict."""
    schema = normalize_schema(columns)
    chosen, blocked = recommend_analyses(schema, objective, granularity)
    resolved_granularity = resolve_time_granularity(objective, time_granularity)
    stages = parse_stages(funnel_stages)
    ctx = {
        "objective": objective,
        "time_granularity": resolved_granularity,
        "funnel_stages": stages,
    }
    analyses = []
    for rank, key in enumerate(chosen):
        analysis = ANALYSIS_DEFS[key]["build"](schema, ctx)
        analysis["rank"] = rank + 1
        analysis["primary"] = rank < 3  # top 3 expanded; the rest collapsed
        analyses.append(analysis)

    desc = timestamp_descriptor(schema)
    return {
        "source": "rule-based",
        "dataset_name": (dataset_name or "").strip(),
        "objective": (objective or "").strip(),
        "granularity": (granularity or "").strip(),
        "time_granularity": resolved_granularity,
        "funnel_stages": stages,
        "schema": schema,
        "capabilities": analyze_capabilities(schema),
        "timestamp": desc,
        "analyses": analyses,
        "blocked": blocked,
        "summary": build_summary(schema, analyses, blocked),
        "dashboard": build_dashboard_blueprint(schema, analyses, stages, resolved_granularity),
        "instructions": build_instructions(schema, resolved_granularity),
        "checklist": build_checklist(schema, analyses),
        "table_note": TABLE_NOTE,
    }


def validate_inputs(objective, granularity, schema):
    """Return a list of human-readable validation errors (empty = valid)."""
    errors = []
    if not (objective or "").strip():
        errors.append("Business analysis objective is required.")
    if not (granularity or "").strip():
        errors.append("Row granularity (what one row represents) is required.")
    if not schema:
        errors.append("Add at least one column to the schema.")
    names = [c["name"] for c in schema]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        errors.append("Duplicate column names: " + ", ".join(sorted(duplicates)))
    return errors
