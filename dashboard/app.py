"""DataPilot dashboard.

Run from the project root:
    python -m streamlit run dashboard/app.py
"""
import html
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs"

# Palette: chart paper, flight-plan navy, taxi yellow, route blue, alert red.
INK, YELLOW, PAPER = "#14213D", "#F5B800", "#F4F6F9"
ROUTE, ALERT, MUTED, GRID = "#2F6DB5", "#C8433F", "#8A94A6", "#D9DFE8"
APP_NAME = "DataPilot"  # change the dashboard name here
APP_TAGLINE = "Spark data analysis"  # and the line under it
# Fonts: any Google Fonts name. Brand needs weight 700, heading 600+700, body 400+500+600.
BRAND_FONT = "Playfair Display"  # the "DataPilot" name in the sidebar
HEADING_FONT = "Bricolage Grotesque"  # page titles, card numbers
BODY_FONT = "Public Sans"  # everything else
FONT_PX = 15
LOWER_IS_BETTER = ("rmse", "mae", "mse", "time", "seconds", "error", "loss")

st.set_page_config(page_title=APP_NAME, page_icon="🛫", layout="wide")

def build_css(scale):
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family={BRAND_FONT.replace(" ", "+")}:wght@700&family={HEADING_FONT.replace(" ", "+")}:wght@600;700&family={BODY_FONT.replace(" ", "+")}:wght@400;500;600&display=swap');
html {{ font-size: {scale}%; }}
.stApp, .stApp p, .stApp li, .stApp label {{ font-family: '{BODY_FONT}', sans-serif; }}
[data-testid="stCaptionContainer"] {{ font-size: 1rem; }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stMetricValue"] {{ font-family: '{HEADING_FONT}', sans-serif; letter-spacing: -0.01em; }}
[data-testid="stSidebar"] {{ border-right: 6px solid {YELLOW}; }}
.hero h1 {{ font-size: 2.3rem; margin: 0; padding: 0; line-height: 1.05; }}
.hero p {{ margin: .25rem 0 1.5rem; color: #44506b; font-size: 1.2rem; }}
.strip {{ display: flex; margin: .25rem 0 .6rem; }}
.stop {{ flex: 1; position: relative; text-align: center; min-width: 0; padding: 0 .25rem; }}
.stop::before {{ content: ""; position: absolute; top: 14px; right: 50%; width: 100%; height: 6px;
  background-color: {GRID}; z-index: 0; }}
.stop:first-child::before {{ display: none; }}
.stop.done::before {{ background-color: {INK}; background-image: linear-gradient(90deg, {YELLOW} 50%, transparent 50%);
  background-size: 22px 2px; background-repeat: repeat-x; background-position: center; }}
.dot {{ position: relative; z-index: 1; width: 34px; height: 34px; margin: 0 auto .5rem; border-radius: 50%;
  display: flex; align-items: center; justify-content: center; font-weight: 700;
  background: {PAPER}; border: 3px dashed {MUTED}; color: {MUTED}; }}
.stop.done .dot {{ background: {YELLOW}; border: 3px solid {INK}; color: {INK}; }}
.stop .name {{ font-family: '{HEADING_FONT}', sans-serif; font-weight: 700; font-size: 1.2rem; }}
.stop .note {{ font-size: .95rem; color: #55617c; overflow-wrap: anywhere; }}
.stop:not(.done) .name {{ color: {MUTED}; }}
.card {{ background: #fff; border-left: 6px solid {INK}; border-radius: 4px; padding: .9rem 1.2rem; margin-bottom: 1rem; }}
.card.warn {{ border-left-color: {ALERT}; }}
.card.go {{ border-left-color: {YELLOW}; }}
.card h4 {{ margin: 0 0 .4rem; }}
.card ul {{ margin: 0; padding-left: 1.1rem; }}
.card li {{ margin-bottom: .4rem; line-height: 1.5; font-size: 1.05rem; }}
.verdict {{ background: {INK}; color: #fff; border-radius: 4px; padding: 1rem 1.3rem; margin: .5rem 0 1.2rem;
  font-size: 1.2rem; line-height: 1.5; }}
.verdict b {{ color: {YELLOW}; }}
.stApp {{ background: {PAPER}; }}
div[class*="st-key-card"], [data-testid="stVerticalBlockBorderWrapper"] {{ background: #fff; border: 1px solid {GRID}; border-radius: 8px; }}
.kpi {{ background: #fff; border: 1px solid {GRID}; border-top: 5px solid {INK}; border-radius: 8px; padding: 1rem 1.2rem; min-height: 8.5rem; margin-bottom: 1rem; }}
.kpi-label {{ font-size: .95rem; color: #55617c; }}
.kpi-value {{ font-family: '{HEADING_FONT}', sans-serif; font-size: 1.8rem; font-weight: 700; line-height: 1.15; overflow-wrap: anywhere; }}
.kpi-value.long {{ font-size: 1.25rem; line-height: 1.35; padding: .35rem 0; }}
.kpi-sub {{ font-size: .9rem; color: #55617c; margin-top: .15rem; }}
.panel {{ background: #fff; border: 1px solid {GRID}; border-radius: 8px; padding: 1rem 1rem .4rem; margin-bottom: 1rem; }}
.panel-title {{ font-family: '{HEADING_FONT}', sans-serif; font-weight: 700; font-size: 1.25rem; margin-bottom: .9rem; }}
.brand {{ display: flex; align-items: center; gap: .7rem; margin: .5rem 0 1rem; }}
.brand-mark {{ width: 2.6rem; height: 2.6rem; border-radius: 8px; background: {YELLOW}; color: {INK}; border: 3px solid {INK};
  display: flex; align-items: center; justify-content: center; font-family: '{BRAND_FONT}', sans-serif; font-weight: 700; font-size: 1.4rem; }}
.brand-name {{ font-family: '{BRAND_FONT}', sans-serif; font-weight: 700; font-size: 1.5rem; line-height: 1.1; }}
.brand-tag {{ font-size: .9rem; color: #55617c; }}
@media (max-width: 700px) {{ .hero h1 {{ font-size: 2.2rem; }} .stop .note {{ display: none; }} }}
</style>
"""


# ---------- helpers ----------
def esc(x) -> str:
    return html.escape(str(x))


def load_json(path: Path):
    """Return parsed JSON, or None if the file is missing or invalid."""
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def list_datasets():
    if not OUT.exists():
        return []
    return sorted(p.name for p in OUT.iterdir() if p.is_dir())


def missing(name: str, dataset: str):
    st.info(
        f"No {name} for **{dataset}** yet. Click **Run pipeline** in the sidebar, "
        f"or run `python -m agents.supervisor {dataset}`."
    )


def style(fig, h=None):
    """Apply the dashboard look to a Plotly figure."""
    fig.update_layout(
        font=dict(family=f"{BODY_FONT}, sans-serif", color=INK, size=FONT_PX),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, l=0, r=0, b=0),
        colorway=[ROUTE, YELLOW, INK, ALERT],
    )
    if h:
        fig.update_layout(height=h)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def render(node, key="x"):
    """Show any JSON value in a readable way."""
    if isinstance(node, dict):
        scalars = {k: v for k, v in node.items() if not isinstance(v, (dict, list))}
        nested = {k: v for k, v in node.items() if isinstance(v, (dict, list))}
        if scalars:
            st.dataframe(
                pd.DataFrame({"field": list(scalars), "value": [str(v) for v in scalars.values()]}),
                hide_index=True,
                use_container_width=True,
            )
        for k, v in nested.items():
            with st.expander(str(k)):
                render(v, f"{key}.{k}")
    elif isinstance(node, list):
        if node and all(isinstance(x, dict) for x in node):
            st.dataframe(pd.json_normalize(node), hide_index=True, use_container_width=True)
        else:
            for item in node:
                st.write(f"- {item}")
    else:
        st.write(node)


def note_card(title, items, kind=""):
    if not items:
        return
    li = "".join(f"<li>{esc(i)}</li>" for i in items)
    st.markdown(f'<div class="card {kind}"><h4>{esc(title)}</h4><ul>{li}</ul></div>', unsafe_allow_html=True)


# ---------- overview ----------
def title_bar(ds):
    eda = load_json(OUT / ds / "eda_results.json") or {}
    rows, target = eda.get("row_count"), eda.get("target")
    sub = f"Predicting {target} from {rows:,} rows." if rows and target else "Results from the DataPilot pipeline."
    st.markdown(f'<div class="hero"><h1>{esc(ds.capitalize())}</h1><p>{esc(sub)}</p></div>', unsafe_allow_html=True)


def road(ds):
    """Pipeline status as a road with five stops."""
    d = OUT / ds
    eda = load_json(d / "eda_results.json") or {}
    ml = load_json(d / "model_results.json") or {}
    ins = load_json(d / "insights.json") or {}
    stops = [
        ("Profile", (d / "profile.json").exists(), "Column types and counts"),
        ("Clean", (d / "cleaning_report.json").exists(), "Fixes and removals"),
        ("Explore", bool(eda), f"{eda['seconds']} s" if "seconds" in eda else "Charts and correlations"),
        ("Model", bool(ml), f"Best: {ml['best_model']}" if ml.get("best_model") else "Model comparison"),
        ("Insights", bool(ins), "Written by the LLM" if ins.get("mode") == "llm" else "Written from templates"),
    ]
    parts = []
    for name, done, note in stops:
        mark = "✓" if done else ""
        text = note if done else "Not run yet"
        parts.append(
            f'<div class="stop {"done" if done else ""}"><div class="dot">{mark}</div>'
            f'<div class="name">{name}</div><div class="note">{esc(text)}</div></div>'
        )
    st.markdown(
        f'<div class="panel"><div class="panel-title">Pipeline status</div>'
        f'<div class="strip">{"".join(parts)}</div></div>',
        unsafe_allow_html=True,
    )


PRIMARY_METRICS = ["r2", "f1", "f1_score", "auc", "roc_auc", "accuracy", "rmse"]


def primary_metric(metrics: dict):
    for k in PRIMARY_METRICS:
        if k in metrics:
            return k
    return next(iter(metrics), None)


def kpi(label, value, sub, accent):
    return (
        f'<div class="kpi" style="border-top-color:{accent}"><div class="kpi-label">{esc(label)}</div>'
        f'<div class="kpi-value{" long" if len(str(value)) > 14 else ""}">{esc(value)}</div><div class="kpi-sub">{esc(sub)}</div></div>'
    )


def page_overview(ds):
    d = OUT / ds
    eda = load_json(d / "eda_results.json") or {}
    ml = load_json(d / "model_results.json") or {}
    ins = load_json(d / "insights.json") or {}
    if not (eda or ml or ins):
        return missing("results", ds)

    models = ml.get("models") or []
    best = next((m for m in models if m.get("name") == ml.get("best_model")), models[0] if models else None)
    pm = primary_metric(best.get("metrics", {})) if best else None
    secs = sum(x for x in (eda.get("seconds"), ml.get("seconds")) if isinstance(x, (int, float)))

    tiles = [
        ("Rows analysed", f"{eda['row_count']:,}" if eda.get("row_count") else "n/a", f"Target: {eda.get('target', 'n/a')}", ROUTE),
        ("Best model", best["name"] if best else "n/a", str(ml.get("problem_type", "")).capitalize(), YELLOW),
        (pm.upper() if pm else "Score", f"{best['metrics'][pm]:.4g}" if best and pm else "n/a", "On held-out test rows", INK),
        ("Compute time", f"{secs:.0f} s" if secs else "n/a", "EDA plus model training", ALERT),
    ]
    for col, tile in zip(st.columns(4), tiles):
        col.markdown(kpi(*tile), unsafe_allow_html=True)

    road(ds)

    left, right = st.columns(2)
    with left:
        with st.container(border=True, key="card-models"):
            st.subheader("Model comparison")
            if models and pm:
                df = pd.DataFrame([{"model": m["name"], pm: m.get("metrics", {}).get(pm)} for m in models])
                df["status"] = df["model"].eq(ml.get("best_model")).map({True: "Best", False: "Other"})
                fig = px.bar(df, x="model", y=pm, color="status", text_auto=".4g",
                             color_discrete_map={"Best": YELLOW, "Other": INK})
                fig.update_traces(marker_line_color=INK, marker_line_width=2, textposition="outside")
                fig.update_layout(xaxis_title=None, showlegend=False)
                st.plotly_chart(style(fig, 320), use_container_width=True)
                lower = any(w in pm.lower() for w in LOWER_IS_BETTER)
                st.caption(f"{pm.upper()}: {'lower' if lower else 'higher'} is better.")
            else:
                st.info("No model results yet.")
    with right:
        with st.container(border=True, key="card-corr"):
            st.subheader("Strongest links with the target")
            cors = eda.get("top_correlations")
            if cors:
                df = pd.DataFrame(cors)
                df["Column type"] = df["leakage"].map({True: "Leaks the answer", False: "Safe to use"})
                fig = px.bar(df, x="correlation", y="col_a", orientation="h", color="Column type",
                             color_discrete_map={"Leaks the answer": ALERT, "Safe to use": ROUTE},
                             labels={"col_a": "", "correlation": "Correlation"})
                fig.update_layout(yaxis=dict(autorange="reversed"), legend_title_text="")
                st.plotly_chart(style(fig, 320), use_container_width=True)
                st.caption("Red columns leak the answer, so the model leaves them out.")
            else:
                st.info("No correlations yet.")

    left, right = st.columns(2)
    with left:
        with st.container(border=True, key="card-trend"):
            tr = (eda.get("chart_data") or {}).get("trend_by_hour")
            fi = ml.get("feature_importance")
            if tr:
                st.subheader(f"Average {tr.get('target', 'target')} by hour")
                df = pd.DataFrame({"hour": tr["hours"], "average": tr["avg_target"]})
                fig = px.area(df, x="hour", y="average")
                fig.update_traces(line_color=INK, fillcolor="rgba(245,184,0,0.35)")
                st.plotly_chart(style(fig, 320), use_container_width=True)
            elif fi:
                st.subheader("What the model relies on")
                fdf = pd.DataFrame(fi).sort_values("importance")
                fig = px.bar(fdf, x="importance", y="feature", orientation="h")
                fig.update_traces(marker_color=ROUTE)
                fig.update_layout(yaxis_title=None)
                st.plotly_chart(style(fig, 320), use_container_width=True)
            else:
                st.subheader("Trend")
                st.info("No trend or feature importance yet.")
    with right:
        with st.container(border=True, key="card-insights"):
            st.subheader("Key findings")
            findings = ins.get("findings") or []
            if findings:
                for f in findings[:4]:
                    st.markdown(f"- {f}")
                st.caption("See the Insights page for limits and next steps.")
            else:
                st.info(f"No insights yet. Run `python -m agents.insight_agent {ds}`.")


# ---------- pages ----------
def page_dataset(ds):
    st.header("Dataset info")
    profile = load_json(OUT / ds / "profile.json")
    if profile is None:
        return missing("profile", ds)
    scalars = {k: v for k, v in profile.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if scalars:
        cols = st.columns(min(len(scalars), 4))
        for col, (k, v) in zip(cols, list(scalars.items())[:4]):
            col.metric(k.replace("_", " ").capitalize(), f"{v:,}")
    cleaning = load_json(OUT / ds / "cleaning_report.json")
    tab1, tab2 = st.tabs(["Profile", "Cleaning report"])
    with tab1:
        render(profile, "profile")
    with tab2:
        if cleaning:
            render(cleaning, "clean")
        else:
            missing("cleaning report", ds)


def page_eda(ds):
    st.header("Exploratory analysis")
    eda = load_json(OUT / ds / "eda_results.json")
    if eda is None:
        return missing("EDA results", ds)

    chart = eda.get("chart_data", {})
    t_sum, t_cor, t_dist, t_trend, t_cat, t_out = st.tabs(
        ["Summary", "Correlations", "Distributions", "Daily rhythm", "Categories", "Outliers"]
    )

    with t_sum:
        stats = eda.get("summary_stats")
        if stats:
            st.dataframe(pd.DataFrame(stats).T, use_container_width=True)
        else:
            st.info("No summary statistics in this file.")

    with t_cor:
        cors = eda.get("top_correlations")
        if cors:
            df = pd.DataFrame(cors)
            df["Column type"] = df["leakage"].map({True: "Leaks the answer", False: "Safe to use"})
            fig = px.bar(
                df, x="correlation", y="col_a", orientation="h", color="Column type",
                color_discrete_map={"Leaks the answer": ALERT, "Safe to use": ROUTE},
                labels={"col_a": "", "correlation": f"Correlation with {eda.get('target', 'target')}"},
            )
            fig.update_layout(yaxis=dict(autorange="reversed"), legend_title_text="")
            st.plotly_chart(style(fig), use_container_width=True)
            st.caption(
                "Red columns are only known after the trip ends, or contain the answer, "
                "so the model must not use them."
            )
        else:
            st.info("No correlations in this file.")

    with t_dist:
        hists = chart.get("histograms", {})
        if hists:
            col = st.selectbox("Column", list(hists), key="hist_col")
            h = hists[col]
            fig = px.bar(x=h["bins"], y=h["counts"], labels={"x": col, "y": "Rows"})
            fig.update_traces(marker_color=ROUTE)
            fig.update_layout(bargap=0.05)
            st.plotly_chart(style(fig), use_container_width=True)
            if h.get("clipped_to"):
                st.caption(f"Values clipped to {h['clipped_to']} so extreme outliers do not stretch the chart.")
        sc = chart.get("scatter")
        if sc and sc.get("points"):
            st.subheader(f"{sc['y']} against {sc['x']}")
            df = pd.DataFrame(sc["points"], columns=[sc["x"], sc["y"]])
            fig = px.scatter(df, x=sc["x"], y=sc["y"], opacity=0.4)
            fig.update_traces(marker_color=ROUTE)
            st.plotly_chart(style(fig), use_container_width=True)
            st.caption(f"A sample of {len(df):,} rows.")

    with t_trend:
        tr = chart.get("trend_by_hour")
        if tr:
            target = tr.get("target", "target")
            df = pd.DataFrame({"hour": tr["hours"], "average": tr["avg_target"], "trips": tr["counts"]})
            df["angle"] = df["hour"] * 15
            df["label"] = df["hour"].map(lambda h: f"{h:02d}:00")
            st.subheader("A day on the clock")
            st.caption(f"Bar length is the number of trips. Colour is the average {target}: pale is low, dark is high.")
            fig = px.bar_polar(
                df, r="trips", theta="angle", color="average",
                color_continuous_scale=["#FCE8A6", YELLOW, INK],
                hover_data={"label": True, "angle": False},
                labels={"average": f"Average {target}", "trips": "Trips"},
            )
            fig.update_layout(
                height=520,
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    angularaxis=dict(
                        direction="clockwise", rotation=90, tickmode="array",
                        tickvals=[0, 90, 180, 270], ticktext=["00:00", "06:00", "12:00", "18:00"],
                        gridcolor=GRID,
                    ),
                    radialaxis=dict(gridcolor=GRID, showticklabels=False),
                ),
            )
            st.plotly_chart(style(fig), use_container_width=True)
            st.subheader(f"Average {target} by pickup hour")
            fig = px.line(df, x="hour", y="average", markers=True, labels={"average": "Average"})
            fig.update_traces(line_color=INK, marker_color=YELLOW)
            st.plotly_chart(style(fig), use_container_width=True)
        else:
            st.info("No hourly trend in this file.")

    with t_cat:
        cats = eda.get("top_categories", {})
        if cats:
            col = st.selectbox("Column", list(cats), key="cat_col")
            df = pd.DataFrame(cats[col])
            fig = px.bar(df, x="value", y="count", labels={"value": col, "count": "Rows"})
            fig.update_traces(marker_color=INK)
            fig.update_xaxes(type="category")
            st.plotly_chart(style(fig), use_container_width=True)
        else:
            st.info("No category counts in this file.")

    with t_out:
        outs = eda.get("outliers")
        if outs:
            df = pd.DataFrame([{"column": k, **v} for k, v in outs.items()])
            total = eda.get("row_count") or 1
            df["% of rows"] = (df["count"] / total * 100).round(1)
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.info("No outlier counts in this file.")


def page_models(ds):
    st.header("Model performance")
    res = load_json(OUT / ds / "model_results.json")
    if res is None:
        return missing("model results", ds)
    models = res.get("models") or []
    if not models:
        st.warning("model_results.json has no models. Showing the raw file.")
        return render(res, "ml")

    best = res.get("best_model")
    rows = [
        {"model": m["name"], **m.get("metrics", {}), "train_seconds": m.get("train_seconds")}
        for m in models
    ]
    df = pd.DataFrame(rows)

    # Plain-language verdict for regression.
    best_row = df[df["model"] == best]
    if res.get("problem_type") == "regression" and not best_row.empty and {"mae", "rmse", "r2"} <= set(df.columns):
        r = best_row.iloc[0]
        st.markdown(
            f'<div class="verdict"><b>{esc(best)}</b> predicts <b>{esc(res.get("target", "the target"))}</b> '
            f'and is off by <b>{r["mae"]:.2f}</b> on average (MAE). Big misses push the typical error to '
            f'<b>{r["rmse"]:.2f}</b> (RMSE). It explains <b>{r["r2"]:.0%}</b> of the variation (R²).</div>',
            unsafe_allow_html=True,
        )
    elif best:
        st.success(f"Best model: {best}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Problem", str(res.get("problem_type", "n/a")).capitalize())
    c2.metric("Training rows", f"{res.get('rows_train', 0):,}")
    c3.metric("Test rows", f"{res.get('rows_test', 0):,}")

    metrics = [c for c in df.columns if c != "model"]
    metric = st.selectbox("Compare models on", metrics)
    lower_better = any(w in metric.lower() for w in LOWER_IS_BETTER)
    st.caption("Lower is better." if lower_better else "Higher is better.")
    df["status"] = df["model"].eq(best).map({True: "Best", False: "Other"})
    fig = px.bar(
        df, x="model", y=metric, color="status", text_auto=".4g",
        color_discrete_map={"Best": YELLOW, "Other": INK},
    )
    fig.update_traces(marker_line_color=INK, marker_line_width=2, textposition="outside")
    fig.update_layout(xaxis_title=None, showlegend=False)
    st.plotly_chart(style(fig), use_container_width=True)
    st.dataframe(df.drop(columns="status"), hide_index=True, use_container_width=True)

    fi = res.get("feature_importance")
    if fi:
        st.subheader("What the model relies on")
        fdf = pd.DataFrame(fi).sort_values("importance")
        fig = px.bar(fdf, x="importance", y="feature", orientation="h")
        fig.update_traces(marker_color=ROUTE)
        fig.update_layout(yaxis_title=None)
        st.plotly_chart(style(fig), use_container_width=True)
        st.caption(
            "Names like RatecodeID_oh_1 are one-hot columns: one per category value "
            "(here, RatecodeID equal to 1)."
        )

    for w in res.get("warnings", []):
        st.warning(w)

    used, excluded = res.get("features_used"), res.get("excluded_columns")
    left, right = st.columns(2)
    with left:
        if used:
            st.subheader("Columns the model used")
            st.write("**Numeric:** " + ", ".join(used.get("numeric", [])))
            st.write("**Categorical:** " + ", ".join(used.get("categorical", [])))
    with right:
        if excluded:
            with st.expander(f"Columns left out ({len(excluded)})"):
                st.dataframe(
                    pd.DataFrame({"column": list(excluded), "reason": list(excluded.values())}),
                    hide_index=True, use_container_width=True,
                )


def page_insights(ds):
    st.header("Insights")
    ins = load_json(OUT / ds / "insights.json")
    if ins is None:
        return missing("insights", ds)
    mode, model = ins.get("mode", "unknown"), ins.get("model")
    st.caption(f"Written by: {'LLM (' + model + ')' if mode == 'llm' and model else mode}")
    if ins.get("fallback_reason"):
        st.warning(f"Template fallback used: {ins['fallback_reason']}")
    note_card("What we found", ins.get("findings"))
    left, right = st.columns(2)
    with left:
        note_card("Limits to keep in mind", ins.get("limitations"), "warn")
    with right:
        note_card("What to do next", ins.get("next_steps"), "go")
    if ins.get("facts"):
        with st.expander("Facts the insights are based on"):
            st.json(ins["facts"])


def page_benchmarks(_ds):
    st.header("Benchmarks")
    summary = OUT / "benchmark_summary.csv"
    results = OUT / "benchmark_results.csv"
    plot = OUT / "scalability.png"
    if not (summary.exists() or results.exists() or plot.exists()):
        st.info("No benchmark files found in outputs/.")
        return
    if summary.exists():
        st.subheader("Summary")
        st.dataframe(pd.read_csv(summary), hide_index=True, use_container_width=True)
    if plot.exists():
        st.subheader("Scalability")
        st.image(str(plot))
    if results.exists():
        with st.expander("All benchmark runs"):
            st.dataframe(pd.read_csv(results), hide_index=True, use_container_width=True)


PAGES = {
    "Overview": page_overview,
    "Dataset info": page_dataset,
    "EDA": page_eda,
    "Model performance": page_models,
    "Insights": page_insights,
    "Benchmarks": page_benchmarks,
}

# ---------- sidebar ----------
st.sidebar.markdown(
    f'<div class="brand"><div class="brand-mark">{esc(APP_NAME[:1].upper())}</div>'
    f'<div><div class="brand-name">{esc(APP_NAME)}</div>'
    f'<div class="brand-tag">{esc(APP_TAGLINE)}</div></div></div>',
    unsafe_allow_html=True,
)
datasets = list_datasets()
if not datasets:
    st.error("No outputs/ folders found. Run the pipeline first.")
    st.stop()

dataset = st.sidebar.selectbox("Dataset", datasets)
size = st.sidebar.select_slider("Text size", options=["Small", "Medium", "Large"], value="Medium")
SCALE = {"Small": 100, "Medium": 112, "Large": 125}[size]
FONT_PX = round(14 * SCALE / 100)
st.markdown(build_css(SCALE), unsafe_allow_html=True)

st.sidebar.divider()
if st.sidebar.button("Run pipeline", help=f"Runs all stages for {dataset}. This can take several minutes."):
    try:
        from agents.supervisor import run_pipeline

        with st.spinner(f"Running pipeline for {dataset}..."):
            run_pipeline(dataset)
        st.sidebar.success("Pipeline finished.")
    except Exception as exc:  # show the reason instead of a stack trace
        st.sidebar.error(f"Pipeline failed: {exc}")

run_info = load_json(OUT / dataset / "pipeline_run.json")
if run_info:
    with st.sidebar.expander("Last pipeline run"):
        st.json(run_info)

# ---------- navigation and main ----------
ICONS = {
    "Overview": ":material/dashboard:",
    "Dataset info": ":material/database:",
    "EDA": ":material/bar_chart:",
    "Model performance": ":material/model_training:",
    "Insights": ":material/lightbulb:",
    "Benchmarks": ":material/speed:",
}


def _wrap(fn):
    def run():
        fn(dataset)

    return run


if hasattr(st, "navigation"):
    nav_pages = [
        st.Page(
            _wrap(fn), title=name, icon=ICONS[name],
            url_path=name.lower().replace(" ", "-"), default=(i == 0),
        )
        for i, (name, fn) in enumerate(PAGES.items())
    ]
    try:
        nav = st.navigation(nav_pages, position="top")  # horizontal menu bar
    except TypeError:  # older Streamlit: menu goes in the sidebar instead
        nav = st.navigation(nav_pages)
    title_bar(dataset)
    nav.run()
else:  # very old Streamlit: simple radio menu
    page = st.sidebar.radio("Page", list(PAGES))
    title_bar(dataset)
    PAGES[page](dataset)