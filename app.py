"""
Streamlit — Per-Company Prediction Timelines (from Supabase)

Reads ONLY the `predictions` table from Supabase and, for a single selected
company, shows the last N weeks (default 2) going forward from the earliest
of that window:

    • Signal score per week            (total_points)
    • Sentiment per week               (sentiment_score)
    • Correct predictions per week     (prediction_result)
    • Predicted vs Actual close per week (predicted_close + actual_close)
    • Whether weekly avg sentiment relates to actual close (per-week + correlation)

Everything downstream operates on the dataframe already pulled into Streamlit —
no further Supabase calls per chart.

Env vars required:  SUPABASE_URL,  SUPABASE_KEY
Run:  streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from supabase import create_client, Client


# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Prediction Timelines", layout="wide")

DEFAULT_WEEKS = 52
FETCH_CAP = 5000  # bound the pull; we only ever chart a couple weeks per company


# ─────────────────────────────────────────────────────────────────
# Supabase
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def get_client() -> Client:
    url = (st.secrets.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL", "")).strip()
    key = (st.secrets.get("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY", "")).strip()
    if not url or not key:
        return None
    return create_client(url, key)


@st.cache_data(ttl=300)
def load_predictions() -> pd.DataFrame:
    """Pull the predictions table once; cached for 5 min so switching
    companies/weeks in the UI doesn't re-hit Supabase every interaction."""
    client = get_client()
    if client is None:
        return pd.DataFrame()

    res = (
        client.table("predictions")
        .select("*")
        .order("target_date", desc=True)
        .limit(FETCH_CAP)
        .execute()
    )
    df = pd.DataFrame(res.data or [])
    if df.empty:
        return df

    # Types
    df["target_date"] = pd.to_datetime(df["target_date"], errors="coerce")
    df["prediction_date"] = pd.to_datetime(df.get("prediction_date"), errors="coerce")
    for col in ["predicted_close", "actual_close", "sentiment_score",
                "total_points", "prediction_error_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["target_date"])

    # Recover the 0–3 Signal Score for older rows where total_points was
    # never saved (NULL) but the text `signal` label was. Same mapping the
    # source Gradio app uses. New rows keep their stored total_points;
    # only NULLs fall back to the label.
    signal_points = {
        "Strong Buy": 3,
        "Buy / Hold": 2,
        "Hold / Weak": 1,
        "Avoid / Sell": 0,
    }
    if "total_points" in df.columns:
        filled = df["total_points"].copy()
    else:
        filled = pd.Series(np.nan, index=df.index)
    if "signal" in df.columns:
        from_label = df["signal"].astype(str).str.strip().map(signal_points)
        filled = filled.where(filled.notna(), from_label)
    df["signal_points_filled"] = pd.to_numeric(filled, errors="coerce")

    # One genuine row per (ticker, target_date): the app was run more than
    # once a week historically, so a ticker can have >1 row for one Friday.
    # Keep the most recently MADE prediction for that week.
    df = (df.sort_values(["ticker", "target_date", "prediction_date"],
                         ascending=[True, True, False])
            .drop_duplicates(subset=["ticker", "target_date"], keep="first"))
    return df


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def company_label(df: pd.DataFrame, ticker: str) -> str:
    name = None
    if "company_name" in df.columns:
        vals = df.loc[df["ticker"] == ticker, "company_name"].dropna()
        if len(vals):
            name = vals.iloc[0]
    return f"{ticker} — {name}" if name else ticker


def last_n_weeks(df_company: pd.DataFrame, n: int) -> pd.DataFrame:
    """The n most recent target_date weeks for this company, ascending in time
    (so the timeline reads left→right, oldest→newest = 'going forward')."""
    weeks = sorted(df_company["target_date"].dropna().unique())[-n:]
    out = df_company[df_company["target_date"].isin(weeks)].copy()
    return out.sort_values("target_date")


def week_label(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────────
# UI — controls
# ─────────────────────────────────────────────────────────────────
st.title("📈 Per-Company Prediction Timelines")

st.info(
    "ℹ️ This is a personal project dashboard. The predictions are **run "
    "manually** in a separate app — the "
    "[Yahoo-Finance AI Ecosystem Walk-Forward ML Engine]"
    "(https://huggingface.co/spaces/antfr99/Yahoo-Finance-AI-Ecosystem-Walk-Forward-ML-Engine) "
    "on Hugging Face Spaces — and each run stores its results in Supabase. "
    "This dashboard only **reads** that stored data; it does not run any "
    "predictions itself."
)

client = get_client()
if client is None:
    st.error(
        "Supabase credentials not found. Set the `SUPABASE_URL` and "
        "`SUPABASE_KEY` environment variables and reload."
    )
    st.stop()

df_all = load_predictions()
if df_all.empty:
    st.warning("No rows returned from the `predictions` table.")
    st.stop()

tickers = sorted(df_all["ticker"].dropna().unique())

# ── Ecosystem → Sector → Company cascade ────────────────────────────────
# ecosystem/sector are stored on every prediction row, so we filter the
# ticker list down the same way the source Gradio app groups them.
has_eco = "ecosystem" in df_all.columns and df_all["ecosystem"].notna().any()
has_sec = "sector" in df_all.columns and df_all["sector"].notna().any()

f1, f2 = st.columns(2)
scope = df_all.copy()

with f1:
    if has_eco:
        eco_opts = ["All ecosystems"] + sorted(
            df_all["ecosystem"].dropna().unique().tolist()
        )
        ecosystem = st.selectbox("Ecosystem", options=eco_opts, key="eco")
        if ecosystem != "All ecosystems":
            scope = scope[scope["ecosystem"] == ecosystem]
    else:
        ecosystem = "All ecosystems"

with f2:
    if has_sec:
        sec_opts = ["All sectors"] + sorted(
            scope["sector"].dropna().unique().tolist()
        )
        sector = st.selectbox("Sector", options=sec_opts, key="sec")
        if sector != "All sectors":
            scope = scope[scope["sector"] == sector]
    else:
        sector = "All sectors"

# Company list respects the ecosystem/sector filters above.
scoped_tickers = sorted(scope["ticker"].dropna().unique().tolist())
if not scoped_tickers:
    st.info("No companies stored for this ecosystem/sector selection yet.")
    st.stop()

c1, c2, c3 = st.columns([3, 1, 1])
with c1:
    ticker = st.selectbox(
        "Company",
        options=scoped_tickers,
        format_func=lambda t: company_label(df_all, t),
    )
with c2:
    # Clamp any persisted widget state into range BEFORE rendering — on
    # Streamlit Cloud a session value saved under an old max_value can
    # otherwise exceed the current bounds and raise StreamlitValueAboveMaxError.
    WEEKS_MIN, WEEKS_MAX = 1, 52
    if "n_weeks" in st.session_state:
        st.session_state["n_weeks"] = int(
            min(max(st.session_state["n_weeks"], WEEKS_MIN), WEEKS_MAX)
        )
    n_weeks = st.number_input(
        "Weeks", min_value=WEEKS_MIN, max_value=WEEKS_MAX,
        value=DEFAULT_WEEKS, step=1, key="n_weeks",
    )
with c3:
    if st.button("🔄 Refresh"):
        load_predictions.clear()
        st.rerun()

df_company = df_all[df_all["ticker"] == ticker].copy()
df_win = last_n_weeks(df_company, int(n_weeks))

if df_win.empty:
    st.info("No predictions stored for this company yet.")
    st.stop()

df_win["Week"] = week_label(df_win["target_date"])

_scope_bits = []
if has_eco and ecosystem != "All ecosystems":
    _scope_bits.append(ecosystem)
if has_sec and sector != "All sectors":
    _scope_bits.append(sector)
_scope_txt = f" ({' · '.join(_scope_bits)})" if _scope_bits else ""

st.caption(
    f"Showing the last {len(df_win['Week'].unique())} week(s) for "
    f"**{company_label(df_all, ticker)}**{_scope_txt} — target dates "
    f"{df_win['Week'].iloc[0]} → {df_win['Week'].iloc[-1]}."
)

X_WEEK = alt.X("Week:O", title="Target Friday", sort=list(df_win["Week"]))


# ─────────────────────────────────────────────────────────────────
# 1) Signal score per week
# ─────────────────────────────────────────────────────────────────
st.subheader("1 · Signal Score per Week")
if "signal_points_filled" in df_win.columns and df_win["signal_points_filled"].notna().any():
    plot_df = df_win.copy()
    # Flag which points came from the stored number vs. recovered from the
    # text signal label (older rows where total_points was never saved).
    stored = plot_df["total_points"] if "total_points" in plot_df.columns else pd.Series(np.nan, index=plot_df.index)
    plot_df["Source"] = np.where(stored.notna(), "stored", "from signal label")

    base = alt.Chart(plot_df).encode(x=X_WEEK)
    line = base.mark_line(color="#4c78a8").encode(
        y=alt.Y("signal_points_filled:Q", title="Signal Score (0–3)",
                scale=alt.Scale(domain=[0, 3])),
    )
    pts = base.mark_point(size=70, filled=True).encode(
        y=alt.Y("signal_points_filled:Q"),
        color=alt.Color("Source:N",
                        scale=alt.Scale(domain=["stored", "from signal label"],
                                        range=["#4c78a8", "#f2a900"]),
                        title="Score source"),
        tooltip=["Week",
                 alt.Tooltip("signal_points_filled:Q", title="Signal Score"),
                 alt.Tooltip("signal:N", title="Signal") if "signal" in plot_df.columns else "Week",
                 "Source"],
    )
    st.altair_chart(line + pts, use_container_width=True)
    if (plot_df["Source"] == "from signal label").any():
        st.caption("Orange points were recovered from the text Signal label "
                   "(Strong Buy=3, Buy/Hold=2, Hold/Weak=1, Avoid/Sell=0) "
                   "because those older rows never stored a numeric score.")
else:
    st.info("No signal score data for this window.")


# ─────────────────────────────────────────────────────────────────
# 2) Sentiment per week
# ─────────────────────────────────────────────────────────────────
st.subheader("2 · Sentiment per Week")
if "sentiment_score" in df_win.columns and df_win["sentiment_score"].notna().any():
    base = alt.Chart(df_win).encode(x=X_WEEK)
    line = base.mark_line(point=True, color="#54a24b").encode(
        y=alt.Y("sentiment_score:Q", title="Avg Sentiment (−1 … +1)",
                scale=alt.Scale(domain=[-1, 1])),
        tooltip=["Week", "sentiment_score"],
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        color="grey", strokeDash=[4, 4]).encode(y="y:Q")
    st.altair_chart(zero + line, use_container_width=True)
else:
    st.info("No `sentiment_score` data for this window.")


# ─────────────────────────────────────────────────────────────────
# 3) Correct predictions per week
# ─────────────────────────────────────────────────────────────────
st.subheader("3 · Correct Predictions per Week")
if "prediction_result" in df_win.columns and df_win["prediction_result"].notna().any():
    tmp = df_win.copy()
    tmp["Outcome"] = tmp["prediction_result"].map({
        "Predicted Correctly": "Correct",
        "Predicted Incorrectly": "Incorrect",
    }).fillna("Not evaluated yet")
    color = alt.Color(
        "Outcome:N",
        scale=alt.Scale(
            domain=["Correct", "Incorrect", "Not evaluated yet"],
            range=["#2e8b57", "#b22222", "#bbbbbb"],
        ),
        title="Outcome",
    )
    bars = alt.Chart(tmp).mark_bar(size=40).encode(
        x=X_WEEK,
        y=alt.Y("prediction_error_pct:Q", title="Actual Error %"),
        color=color,
        tooltip=["Week", "Outcome", "prediction_error_pct", "grade"]
        if "grade" in tmp.columns else ["Week", "Outcome", "prediction_error_pct"],
    )
    st.altair_chart(bars, use_container_width=True)
    st.caption("Bar height = real error %; colour = whether it landed inside "
               "that week's volatility-adjusted threshold. Grey = the target "
               "Friday hasn't passed / been evaluated yet.")
else:
    st.info("No evaluated results for this window yet.")


# ─────────────────────────────────────────────────────────────────
# 4) Predicted vs Actual close per week (together)
# ─────────────────────────────────────────────────────────────────
st.subheader("4 · Predicted vs Actual Close per Week")
if {"predicted_close", "actual_close"}.issubset(df_win.columns):
    long = df_win.melt(
        id_vars=["Week"],
        value_vars=["predicted_close", "actual_close"],
        var_name="Series", value_name="Close",
    ).dropna(subset=["Close"])
    long["Series"] = long["Series"].map({
        "predicted_close": "Predicted",
        "actual_close": "Actual",
    })
    if not long.empty:
        chart = alt.Chart(long).mark_line(point=True).encode(
            x=X_WEEK,
            y=alt.Y("Close:Q", title="Close Price", scale=alt.Scale(zero=False)),
            color=alt.Color("Series:N",
                            scale=alt.Scale(domain=["Predicted", "Actual"],
                                            range=["#4c78a8", "#e45756"]),
                            title=""),
            tooltip=["Week", "Series", "Close"],
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption("Actual is blank for weeks not yet evaluated.")
    else:
        st.info("No close prices for this window.")
else:
    st.info("`predicted_close` / `actual_close` columns not present.")


# ─────────────────────────────────────────────────────────────────
# 5) Does weekly sentiment relate to actual close?
# ─────────────────────────────────────────────────────────────────
st.subheader("5 · Sentiment vs Actual Close — Relationship")
if {"sentiment_score", "actual_close"}.issubset(df_win.columns):
    rel = df_win.dropna(subset=["sentiment_score", "actual_close"]).copy()
    if len(rel) >= 2:
        # Dual-axis per-week overlay: sentiment (bars) vs actual close (line)
        base = alt.Chart(rel).encode(x=X_WEEK)
        sent_bars = base.mark_bar(opacity=0.45, color="#54a24b").encode(
            y=alt.Y("sentiment_score:Q", title="Avg Sentiment",
                    scale=alt.Scale(domain=[-1, 1])),
            tooltip=["Week", "sentiment_score"],
        )
        close_line = base.mark_line(point=True, color="#e45756").encode(
            y=alt.Y("actual_close:Q", title="Actual Close",
                    scale=alt.Scale(zero=False)),
            tooltip=["Week", "actual_close"],
        )
        st.altair_chart(
            alt.layer(sent_bars, close_line).resolve_scale(y="independent"),
            use_container_width=True,
        )

        # Simple quantified read: correlation of sentiment vs the week-over-week
        # change in actual close (sentiment is a directional signal, so pairing
        # it with the *move* is more meaningful than with the price level).
        rel = rel.sort_values("target_date")
        rel["close_change"] = rel["actual_close"].diff()
        pair = rel.dropna(subset=["close_change"])

        st.subheader("6 · Corr: sentiment ↔ weekly price move")
        colA, colB = st.columns(2)
        with colA:
            if len(pair) >= 2 and pair["sentiment_score"].nunique() > 1:
                corr_move = pair["sentiment_score"].corr(pair["close_change"])
                st.metric("Sentiment ↔ weekly price move",
                          f"{corr_move:+.2f}" if pd.notna(corr_move) else "n/a")
            else:
                st.metric("Sentiment ↔ weekly price move", "n/a")
        with colB:
            if rel["sentiment_score"].nunique() > 1 and rel["actual_close"].nunique() > 1:
                corr_lvl = rel["sentiment_score"].corr(rel["actual_close"])
                st.metric("Sentiment ↔ close level",
                          f"{corr_lvl:+.2f}" if pd.notna(corr_lvl) else "n/a")
            else:
                st.metric("Sentiment ↔ close level", "n/a")

        st.caption(
            "+1 = move together, −1 = move opposite, ~0 = no linear link. "
            "With only a couple of weeks this is directional, not statistical "
            "proof — extend the week count for a firmer read."
        )
    else:
        st.info("Need at least two evaluated weeks with both sentiment and "
                "actual close to assess a relationship.")
else:
    st.info("`sentiment_score` / `actual_close` columns not present.")


# ─────────────────────────────────────────────────────────────────
# Underlying rows
# ─────────────────────────────────────────────────────────────────
with st.expander("Show underlying rows for this window"):
    show_cols = [c for c in [
        "Week", "ticker", "company_name", "prediction_date", "target_date",
        "close_at_prediction", "predicted_close", "actual_close",
        "sentiment_score", "total_points", "signal",
        "prediction_error_pct", "prediction_result", "grade",
    ] if c in df_win.columns]
    st.dataframe(df_win[show_cols].reset_index(drop=True),
                 use_container_width=True)
