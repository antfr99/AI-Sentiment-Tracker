"""
Streamlit — Prediction Timelines (from Supabase)

Reads ONLY the `predictions` table from Supabase and presents it three ways,
one per tab:

    • Company   — a single ticker, week by week (the original view)
    • Ecosystem — every company in an ecosystem rolled up per week
    • Sector    — every company in a sector rolled up per week

Each tab shows, for the last N weeks (default 52) ascending in time:

    1 Signal score per week            (total_points, 0–3)
    2 Sentiment per week               (sentiment_score, −1…+1)
    3 Correct predictions per week     (prediction_result)
    4 Predicted vs Actual per week     (close for a company, % move for a group)
    5 Sentiment vs price overlay
    6 Correlation + a short plain-English explanation of what it means

Roll-up note: averaging raw close prices across different companies is
meaningless (a $900 stock would drown a $12 one), so the group tabs compare
the *percentage move* from close_at_prediction instead of the price level.

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
WEEKS_MIN, WEEKS_MAX = 1, 52
FETCH_CAP = 5000  # bound the pull

SIGNAL_POINTS = {
    "Strong Buy": 3,
    "Buy / Hold": 2,
    "Hold / Weak": 1,
    "Avoid / Sell": 0,
}

EVALUATED_LABELS = ["Predicted Correctly", "Predicted Incorrectly"]


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
    companies/groups/weeks in the UI doesn't re-hit Supabase every click."""
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
    for col in ["predicted_close", "actual_close", "close_at_prediction",
                "sentiment_score", "total_points", "prediction_error_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["target_date"])

    # Recover the 0–3 Signal Score for older rows where total_points was
    # never saved (NULL) but the text `signal` label was. Same mapping the
    # source Gradio app uses. New rows keep their stored total_points;
    # only NULLs fall back to the label.
    if "total_points" in df.columns:
        filled = df["total_points"].copy()
    else:
        filled = pd.Series(np.nan, index=df.index)
    if "signal" in df.columns:
        from_label = df["signal"].astype(str).str.strip().map(SIGNAL_POINTS)
        filled = filled.where(filled.notna(), from_label)
    df["signal_points_filled"] = pd.to_numeric(filled, errors="coerce")

    # One genuine row per (ticker, target_date): the app was run more than
    # once a week historically, so a ticker can have >1 row for one Friday.
    # Keep the most recently MADE prediction for that week.
    df = (df.sort_values(["ticker", "target_date", "prediction_date"],
                         ascending=[True, True, False])
            .drop_duplicates(subset=["ticker", "target_date"], keep="first"))

    # Percentage moves — the only fair way to average across companies.
    if {"close_at_prediction", "predicted_close"}.issubset(df.columns):
        base = df["close_at_prediction"].replace(0, np.nan)
        df["predicted_move_pct"] = (df["predicted_close"] - base) / base * 100
        if "actual_close" in df.columns:
            df["actual_move_pct"] = (df["actual_close"] - base) / base * 100

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


def last_n_weeks(df_scope: pd.DataFrame, n: int) -> pd.DataFrame:
    """The n most recent target_date weeks in this slice, ascending in time
    (so the timeline reads left→right, oldest→newest = 'going forward')."""
    weeks = sorted(df_scope["target_date"].dropna().unique())[-n:]
    out = df_scope[df_scope["target_date"].isin(weeks)].copy()
    return out.sort_values("target_date")


def week_label(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.strftime("%Y-%m-%d")


def weeks_input(key: str, label: str = "Weeks") -> int:
    """Clamp any persisted widget state into range BEFORE rendering — on
    Streamlit Cloud a session value saved under an old max_value can
    otherwise exceed the current bounds and raise StreamlitValueAboveMaxError."""
    if key in st.session_state:
        st.session_state[key] = int(
            min(max(st.session_state[key], WEEKS_MIN), WEEKS_MAX)
        )
    return st.number_input(
        label, min_value=WEEKS_MIN, max_value=WEEKS_MAX,
        value=DEFAULT_WEEKS, step=1, key=key,
    )


def safe_corr(df: pd.DataFrame, a: str, b: str) -> float:
    """Correlation that returns NaN instead of blowing up on flat/short data."""
    if a not in df.columns or b not in df.columns:
        return np.nan
    d = df.dropna(subset=[a, b])
    if len(d) < 2 or d[a].nunique() < 2 or d[b].nunique() < 2:
        return np.nan
    return d[a].corr(d[b])


def corr_strength(c: float) -> str:
    if pd.isna(c):
        return "not enough variation in this window to measure"
    a = abs(c)
    direction = "same direction" if c > 0 else "opposite directions"
    if a < 0.20:
        return "essentially no linear link"
    if a < 0.40:
        return f"a weak tendency to move in the {direction}"
    if a < 0.60:
        return f"a moderate tendency to move in the {direction}"
    if a < 0.80:
        return f"a strong tendency to move in the {direction}"
    return f"a very strong tendency to move in the {direction}"


def render_corr_note(scope_txt: str, n_pairs: int, items: list) -> None:
    """Short plain-English explanation of point 6.

    `items` is a list of (metric name, what it pairs, value) tuples."""
    st.markdown(
        f"**What point 6 is showing.** Each week in this window is reduced to "
        f"a single pair of numbers for {scope_txt}, and the correlation "
        f"measures how consistently those two numbers rose and fell together "
        f"across the **{n_pairs} week(s)** compared. The scale runs from "
        f"**+1** (they always moved together), through **0** (no linear link), "
        f"to **−1** (when one went up the other went down)."
    )
    for name, pairs_txt, val in items:
        shown = "n/a" if pd.isna(val) else f"{val:+.2f}"
        st.markdown(f"- **{name}** ({shown}) — {pairs_txt} {corr_strength(val)}.")
    st.markdown(
        "It describes what happened in this window; it does not prove one "
        "caused the other. A handful of weeks can throw up a large number by "
        "chance, so raise the week count before reading much into it — "
        "roughly 12+ weeks before a number is worth acting on."
    )


def aggregate_weeks(df_win: pd.DataFrame, group_col: str | None) -> pd.DataFrame:
    """Collapse rows to one record per (group, week) — or per week if
    group_col is None. Prices are averaged as % moves, not levels."""
    keys = ["target_date"] if group_col is None else [group_col, "target_date"]
    rows = []
    for keyvals, g in df_win.groupby(keys):
        if not isinstance(keyvals, tuple):
            keyvals = (keyvals,)
        rec = dict(zip(keys, keyvals))
        rec["companies"] = int(g["ticker"].nunique())

        for col in ["signal_points_filled", "sentiment_score",
                    "predicted_move_pct", "actual_move_pct",
                    "predicted_close", "actual_close"]:
            if col in g.columns:
                rec[col] = g[col].mean()

        if "prediction_error_pct" in g.columns:
            rec["prediction_error_pct"] = g["prediction_error_pct"].abs().mean()

        if "prediction_result" in g.columns:
            evaluated = int(g["prediction_result"].isin(EVALUATED_LABELS).sum())
            correct = int((g["prediction_result"] == "Predicted Correctly").sum())
            rec["evaluated"] = evaluated
            rec["correct"] = correct
            rec["accuracy_pct"] = (correct / evaluated * 100) if evaluated else np.nan

        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("target_date")
    out["Week"] = week_label(out["target_date"])
    return out


# ─────────────────────────────────────────────────────────────────
# Tab 1 — Company
# ─────────────────────────────────────────────────────────────────
def render_company_tab(df_all: pd.DataFrame) -> None:
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

    scoped_tickers = sorted(scope["ticker"].dropna().unique().tolist())
    if not scoped_tickers:
        st.info("No companies stored for this ecosystem/sector selection yet.")
        return

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        ticker = st.selectbox(
            "Company",
            options=scoped_tickers,
            format_func=lambda t: company_label(df_all, t),
        )
    with c2:
        n_weeks = weeks_input("n_weeks")
    with c3:
        if st.button("🔄 Refresh", key="refresh_company"):
            load_predictions.clear()
            st.rerun()

    df_company = df_all[df_all["ticker"] == ticker].copy()
    df_win = last_n_weeks(df_company, int(n_weeks))

    if df_win.empty:
        st.info("No predictions stored for this company yet.")
        return

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

    # 1) Signal score per week
    st.subheader("1 · Signal Score per Week")
    if df_win["signal_points_filled"].notna().any():
        plot_df = df_win.copy()
        stored = (plot_df["total_points"] if "total_points" in plot_df.columns
                  else pd.Series(np.nan, index=plot_df.index))
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
                     alt.Tooltip("signal:N", title="Signal")
                     if "signal" in plot_df.columns else "Week",
                     "Source"],
        )
        st.altair_chart(line + pts, use_container_width=True)
        if (plot_df["Source"] == "from signal label").any():
            st.caption("Orange points were recovered from the text Signal label "
                       "(Strong Buy=3, Buy/Hold=2, Hold/Weak=1, Avoid/Sell=0) "
                       "because those older rows never stored a numeric score.")
    else:
        st.info("No signal score data for this window.")

    # 2) Sentiment per week
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

    # 3) Correct predictions per week
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

    # 4) Predicted vs Actual close per week
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

    # 5 & 6) Sentiment vs actual close
    st.subheader("5 · Sentiment vs Actual Close — Relationship")
    if {"sentiment_score", "actual_close"}.issubset(df_win.columns):
        rel = df_win.dropna(subset=["sentiment_score", "actual_close"]).copy()
        if len(rel) >= 2:
            base = alt.Chart(rel).encode(x=X_WEEK)
            sent_bars = base.mark_bar(opacity=0.55).encode(
                y=alt.Y("sentiment_score:Q", title="Avg Sentiment",
                        scale=alt.Scale(domain=[-1, 1])),
                color=alt.condition(
                    alt.datum.sentiment_score >= 0,
                    alt.value("#54a24b"),   # green — non-negative sentiment
                    alt.value("#f2b077"),   # light orange — negative sentiment
                ),
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

            rel = rel.sort_values("target_date")
            rel["close_change"] = rel["actual_close"].diff()
            pair = rel.dropna(subset=["close_change"])

            st.subheader("6 · Corr: sentiment ↔ weekly price move")
            corr_move = safe_corr(pair, "sentiment_score", "close_change")
            corr_lvl = safe_corr(rel, "sentiment_score", "actual_close")

            colA, colB = st.columns(2)
            with colA:
                st.metric("Sentiment ↔ weekly price move",
                          f"{corr_move:+.2f}" if pd.notna(corr_move) else "n/a")
            with colB:
                st.metric("Sentiment ↔ close level",
                          f"{corr_lvl:+.2f}" if pd.notna(corr_lvl) else "n/a")

            render_corr_note(
                scope_txt=f"**{company_label(df_all, ticker)}**",
                n_pairs=len(pair) if len(pair) else len(rel),
                items=[
                    ("Sentiment ↔ weekly price move",
                     "pairs that week's average news sentiment with how much the "
                     "actual close moved versus the week before. This is the more "
                     "meaningful of the two, because sentiment is a directional "
                     "signal and a price *move* is directional too. Right now it shows",
                     corr_move),
                    ("Sentiment ↔ close level",
                     "pairs sentiment with the price itself rather than the move. "
                     "A trending stock can make this look impressive even when "
                     "sentiment adds nothing, so treat it as context only. "
                     "Right now it shows",
                     corr_lvl),
                ],
            )
        else:
            st.info("Need at least two evaluated weeks with both sentiment and "
                    "actual close to assess a relationship.")
    else:
        st.info("`sentiment_score` / `actual_close` columns not present.")

    # Underlying rows
    with st.expander("Show underlying rows for this window"):
        show_cols = [c for c in [
            "Week", "ticker", "company_name", "prediction_date", "target_date",
            "close_at_prediction", "predicted_close", "actual_close",
            "sentiment_score", "total_points", "signal",
            "prediction_error_pct", "prediction_result", "grade",
        ] if c in df_win.columns]
        st.dataframe(df_win[show_cols].reset_index(drop=True),
                     use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# Tabs 2 & 3 — Ecosystem / Sector roll-ups
# ─────────────────────────────────────────────────────────────────
def render_group_tab(df_all: pd.DataFrame, group_col: str, label: str,
                     key_prefix: str) -> None:
    if group_col not in df_all.columns or df_all[group_col].dropna().empty:
        st.info(f"No `{group_col}` values stored on the predictions table yet.")
        return

    groups = sorted(df_all[group_col].dropna().unique().tolist())
    compare_opt = f"All {label.lower()}s (compare)"
    opts = [compare_opt] + groups

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        choice = st.selectbox(label, options=opts, key=f"{key_prefix}_sel")
    with c2:
        n_weeks = weeks_input(f"{key_prefix}_weeks")
    with c3:
        if st.button("🔄 Refresh", key=f"{key_prefix}_refresh"):
            load_predictions.clear()
            st.rerun()

    scope = df_all.dropna(subset=[group_col]).copy()
    multi = choice == compare_opt
    if not multi:
        scope = scope[scope[group_col] == choice]

    df_win = last_n_weeks(scope, int(n_weeks))
    if df_win.empty:
        st.info(f"No predictions stored for this {label.lower()} yet.")
        return

    agg = aggregate_weeks(df_win, group_col if multi else None)
    if agg.empty:
        st.info("Nothing to roll up for this window.")
        return

    week_order = list(dict.fromkeys(agg["Week"]))
    X_WEEK = alt.X("Week:O", title="Target Friday", sort=week_order)
    color_enc = (alt.Color(f"{group_col}:N", title=label) if multi
                 else alt.value("#4c78a8"))

    scope_name = f"all {label.lower()}s" if multi else f"**{choice}**"
    st.caption(
        f"Rolling up **{df_win['ticker'].nunique()} companies** across "
        f"{scope_name} over the last {agg['Week'].nunique()} week(s) — "
        f"{week_order[0]} → {week_order[-1]}. Every point is an average of the "
        f"companies that had a prediction stored for that week."
    )

    # 1) Average signal score
    st.subheader("1 · Average Signal Score per Week")
    if agg.get("signal_points_filled") is not None and agg["signal_points_filled"].notna().any():
        chart = alt.Chart(agg).mark_line(point=True).encode(
            x=X_WEEK,
            y=alt.Y("signal_points_filled:Q", title="Avg Signal Score (0–3)",
                    scale=alt.Scale(domain=[0, 3])),
            color=color_enc,
            tooltip=["Week", alt.Tooltip("signal_points_filled:Q",
                                         title="Avg score", format=".2f"),
                     "companies"] + ([group_col] if multi else []),
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption("An average near 3 means the engine was broadly bullish on "
                   "this basket that week; near 0 means broadly negative.")
    else:
        st.info("No signal score data for this window.")

    # 2) Average sentiment
    st.subheader("2 · Average Sentiment per Week")
    if "sentiment_score" in agg.columns and agg["sentiment_score"].notna().any():
        line = alt.Chart(agg).mark_line(point=True).encode(
            x=X_WEEK,
            y=alt.Y("sentiment_score:Q", title="Avg Sentiment (−1 … +1)",
                    scale=alt.Scale(domain=[-1, 1])),
            color=color_enc,
            tooltip=["Week", alt.Tooltip("sentiment_score:Q", title="Avg sentiment",
                                         format=".3f"), "companies"]
                    + ([group_col] if multi else []),
        )
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
            color="grey", strokeDash=[4, 4]).encode(y="y:Q")
        st.altair_chart(zero + line, use_container_width=True)
    else:
        st.info("No `sentiment_score` data for this window.")

    # 3) Hit rate
    st.subheader("3 · Hit Rate per Week")
    if "accuracy_pct" in agg.columns and agg["accuracy_pct"].notna().any():
        tooltip = ["Week", alt.Tooltip("accuracy_pct:Q", title="Correct %",
                                       format=".0f"),
                   alt.Tooltip("correct:Q", title="Correct"),
                   alt.Tooltip("evaluated:Q", title="Evaluated")]
        if multi:
            tooltip.append(group_col)
            chart = alt.Chart(agg.dropna(subset=["accuracy_pct"])).mark_line(
                point=True).encode(
                x=X_WEEK,
                y=alt.Y("accuracy_pct:Q", title="Correct predictions (%)",
                        scale=alt.Scale(domain=[0, 100])),
                color=color_enc,
                tooltip=tooltip,
            )
        else:
            chart = alt.Chart(agg.dropna(subset=["accuracy_pct"])).mark_bar(
                size=30, color="#2e8b57").encode(
                x=X_WEEK,
                y=alt.Y("accuracy_pct:Q", title="Correct predictions (%)",
                        scale=alt.Scale(domain=[0, 100])),
                tooltip=tooltip,
            )
        fifty = alt.Chart(pd.DataFrame({"y": [50]})).mark_rule(
            color="grey", strokeDash=[4, 4]).encode(y="y:Q")
        st.altair_chart(chart + fifty, use_container_width=True)

        ev = int(agg["evaluated"].sum()) if "evaluated" in agg.columns else 0
        cor = int(agg["correct"].sum()) if "correct" in agg.columns else 0
        if ev:
            st.caption(f"Window total: {cor} of {ev} evaluated predictions "
                       f"correct ({cor / ev * 100:.0f}%). Weeks whose target "
                       f"Friday hasn't been evaluated yet are omitted. Dashed "
                       f"line = 50%.")
    else:
        st.info("No evaluated results for this window yet.")

    # 4) Predicted vs actual move %
    st.subheader("4 · Average Predicted vs Actual Move per Week")
    if {"predicted_move_pct", "actual_move_pct"}.issubset(agg.columns):
        id_vars = ["Week"] + ([group_col] if multi else [])
        long = agg.melt(
            id_vars=id_vars,
            value_vars=["predicted_move_pct", "actual_move_pct"],
            var_name="Series", value_name="Move",
        ).dropna(subset=["Move"])
        long["Series"] = long["Series"].map({
            "predicted_move_pct": "Predicted",
            "actual_move_pct": "Actual",
        })
        if not long.empty:
            enc = dict(
                x=X_WEEK,
                y=alt.Y("Move:Q", title="Avg move vs price at prediction (%)"),
                tooltip=id_vars + ["Series",
                                   alt.Tooltip("Move:Q", title="Move %",
                                               format="+.2f")],
            )
            if multi:
                chart = alt.Chart(long).mark_line(point=True).encode(
                    color=alt.Color(f"{group_col}:N", title=label),
                    strokeDash=alt.StrokeDash("Series:N", title=""),
                    **enc,
                )
            else:
                chart = alt.Chart(long).mark_line(point=True).encode(
                    color=alt.Color("Series:N",
                                    scale=alt.Scale(domain=["Predicted", "Actual"],
                                                    range=["#4c78a8", "#e45756"]),
                                    title=""),
                    **enc,
                )
            zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
                color="grey", strokeDash=[4, 4]).encode(y="y:Q")
            st.altair_chart(chart + zero, use_container_width=True)
            st.caption("Prices are compared as **% move from the close at "
                       "prediction time**, not as raw prices — averaging a $900 "
                       "stock with a $12 one would otherwise be meaningless. "
                       "Actual is blank for weeks not yet evaluated.")
        else:
            st.info("No move data for this window.")
    else:
        st.info("`close_at_prediction` is needed to compare moves across "
                "companies and isn't present on these rows.")

    # 5) Sentiment vs actual move overlay
    st.subheader("5 · Sentiment vs Actual Move — Relationship")
    rel_cols = {"sentiment_score", "actual_move_pct"}
    rel = (agg.dropna(subset=list(rel_cols)).copy()
           if rel_cols.issubset(agg.columns) else pd.DataFrame())

    if len(rel) >= 2:
        if multi:
            chart = alt.Chart(rel).mark_circle(size=90, opacity=0.75).encode(
                x=alt.X("sentiment_score:Q", title="Avg sentiment that week"),
                y=alt.Y("actual_move_pct:Q", title="Avg actual move (%)"),
                color=alt.Color(f"{group_col}:N", title=label),
                tooltip=["Week", group_col,
                         alt.Tooltip("sentiment_score:Q", format=".3f"),
                         alt.Tooltip("actual_move_pct:Q", format="+.2f")],
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption("Each dot is one week for one " + label.lower() +
                       ". Dots sloping up-right = more positive sentiment went "
                       "with stronger actual moves.")
        else:
            base = alt.Chart(rel).encode(x=X_WEEK)
            sent_bars = base.mark_bar(opacity=0.55).encode(
                y=alt.Y("sentiment_score:Q", title="Avg Sentiment",
                        scale=alt.Scale(domain=[-1, 1])),
                color=alt.condition(
                    alt.datum.sentiment_score >= 0,
                    alt.value("#54a24b"),
                    alt.value("#f2b077"),
                ),
                tooltip=["Week", "sentiment_score", "companies"],
            )
            move_line = base.mark_line(point=True, color="#e45756").encode(
                y=alt.Y("actual_move_pct:Q", title="Avg Actual Move (%)",
                        scale=alt.Scale(zero=False)),
                tooltip=["Week", alt.Tooltip("actual_move_pct:Q", format="+.2f")],
            )
            st.altair_chart(
                alt.layer(sent_bars, move_line).resolve_scale(y="independent"),
                use_container_width=True,
            )
    else:
        st.info("Need at least two evaluated weeks with both sentiment and an "
                "actual move to assess a relationship.")

    # 6) Correlations
    st.subheader("6 · Corr: sentiment ↔ outcome")
    if multi:
        rows = []
        for gname, g in agg.groupby(group_col):
            g = g.sort_values("target_date")
            rows.append({
                label: gname,
                "Weeks": int(g["Week"].nunique()),
                "Sentiment ↔ actual move": safe_corr(g, "sentiment_score",
                                                     "actual_move_pct"),
                "Sentiment ↔ hit rate": safe_corr(g, "sentiment_score",
                                                  "accuracy_pct"),
            })
        corr_tbl = pd.DataFrame(rows).sort_values(
            "Sentiment ↔ actual move", ascending=False, na_position="last")
        st.dataframe(
            corr_tbl.style.format({"Sentiment ↔ actual move": "{:+.2f}",
                                   "Sentiment ↔ hit rate": "{:+.2f}"},
                                  na_rep="n/a"),
            use_container_width=True, hide_index=True,
        )
        best = corr_tbl.dropna(subset=["Sentiment ↔ actual move"])
        render_corr_note(
            scope_txt=f"each {label.lower()} in turn (its weekly averages "
                      f"across its member companies)",
            n_pairs=int(agg["Week"].nunique()),
            items=[
                ("Sentiment ↔ actual move",
                 f"pairs a {label.lower()}'s average weekly sentiment with the "
                 f"average actual move of its companies that week. Across the "
                 f"table it ranges from "
                 f"{best['Sentiment ↔ actual move'].min():+.2f} to "
                 f"{best['Sentiment ↔ actual move'].max():+.2f}, which is"
                 if len(best) else
                 f"pairs a {label.lower()}'s average weekly sentiment with the "
                 f"average actual move of its companies that week. Currently",
                 best["Sentiment ↔ actual move"].max() if len(best) else np.nan),
                ("Sentiment ↔ hit rate",
                 "pairs weekly sentiment with the share of that week's "
                 "predictions that came out correct — it asks whether the model "
                 "does better in optimistic weeks than pessimistic ones. "
                 "The strongest value in the table shows",
                 corr_tbl["Sentiment ↔ hit rate"].max()
                 if corr_tbl["Sentiment ↔ hit rate"].notna().any() else np.nan),
            ],
        )
        st.caption(f"Sorted by the sentiment↔move column. A {label.lower()} with "
                   f"only one or two evaluated weeks will show n/a.")
    else:
        srt = agg.sort_values("target_date")
        corr_move = safe_corr(srt, "sentiment_score", "actual_move_pct")
        corr_hit = safe_corr(srt, "sentiment_score", "accuracy_pct")
        corr_sig = safe_corr(srt, "signal_points_filled", "actual_move_pct")

        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Sentiment ↔ actual move",
                      f"{corr_move:+.2f}" if pd.notna(corr_move) else "n/a")
        with colB:
            st.metric("Sentiment ↔ hit rate",
                      f"{corr_hit:+.2f}" if pd.notna(corr_hit) else "n/a")
        with colC:
            st.metric("Signal score ↔ actual move",
                      f"{corr_sig:+.2f}" if pd.notna(corr_sig) else "n/a")

        render_corr_note(
            scope_txt=f"**{choice}** as a whole (every member company averaged "
                      f"into one number per week)",
            n_pairs=int(agg["Week"].nunique()),
            items=[
                ("Sentiment ↔ actual move",
                 f"pairs the {label.lower()}'s average weekly news sentiment "
                 f"with the average actual % move of its companies that week. "
                 f"It asks whether good news weeks were also up weeks for the "
                 f"basket. Right now it shows", corr_move),
                ("Sentiment ↔ hit rate",
                 "pairs weekly sentiment with the share of that week's "
                 "predictions that landed correctly — i.e. whether the model is "
                 "sharper in upbeat weeks than gloomy ones. Right now it shows",
                 corr_hit),
                ("Signal score ↔ actual move",
                 f"pairs the average 0–3 signal score with what the "
                 f"{label.lower()} actually did. This is the closest thing here "
                 f"to a scorecard for the signal itself. Right now it shows",
                 corr_sig),
            ],
        )

    # Underlying rows
    with st.expander("Show weekly roll-up rows"):
        cols = [c for c in [
            "Week", group_col, "companies", "signal_points_filled",
            "sentiment_score", "predicted_move_pct", "actual_move_pct",
            "prediction_error_pct", "correct", "evaluated", "accuracy_pct",
        ] if c in agg.columns]
        st.dataframe(agg[cols].reset_index(drop=True), use_container_width=True)

    with st.expander("Show the individual company rows behind these averages"):
        show_cols = [c for c in [
            "target_date", "ticker", "company_name", "ecosystem", "sector",
            "close_at_prediction", "predicted_close", "actual_close",
            "predicted_move_pct", "actual_move_pct", "sentiment_score",
            "total_points", "signal", "prediction_error_pct",
            "prediction_result", "grade",
        ] if c in df_win.columns]
        st.dataframe(df_win[show_cols].reset_index(drop=True),
                     use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────
st.title("📈 Prediction Timelines")

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

tab_company, tab_eco, tab_sector = st.tabs(
    ["🏢 Company", "🌐 AI Ecosystem", "🏭 Sector"]
)

with tab_company:
    render_company_tab(df_all)

with tab_eco:
    render_group_tab(df_all, "ecosystem", "Ecosystem", "eco_tab")

with tab_sector:
    render_group_tab(df_all, "sector", "Sector", "sec_tab")
