"""
app.py — Nigeria Health AI Data Refinery Dashboard
===================================================
Production-ready Streamlit dashboard — 6 pages:
  1. Overview       — database health, key metrics
  2. Data Explorer  — browse, filter, download all tables
  3. Disease Map    — incidence by zone/state, seasonal, climate
  4. Maternal Health — maternal & child indicators
  5. Data Quality   — completeness, bias flags, reporting weights
  6. Insights       — top risk LGAs, poverty correlation, model readiness

Run:
    streamlit run app.py
"""

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────
# ── Cloud database bootstrap ──────────────────────────────────────────────────
import subprocess, sys
_DB = Path(os.getenv("DB_PATH", "./nigeria.db"))

if not _DB.exists() or _DB.stat().st_size < 1_000_000:
    _placeholder = st.empty()
    _placeholder.info("⏳ Building database on first launch — please wait 3–4 minutes...")
    try:
        import startup
        startup.run_setup()
    except Exception as _e:
        st.error(f"Startup error: {_e}")
    _placeholder.empty()
    st.rerun()
DB_PATH = os.getenv("DB_PATH", "./nigeria.db")

st.set_page_config(
    page_title="Nigeria Health AI Refinery",
    page_icon="🇳🇬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour maps ───────────────────────────────────────────────────────────────
ZONE_COLOURS = {
    "NC": "#4CAF50", "NE": "#F44336", "NW": "#FF9800",
    "SE": "#2196F3", "SS": "#9C27B0", "SW": "#00BCD4",
}
DISEASE_COLOURS = {
    "malaria":      "#E53935", "cholera":      "#1E88E5",
    "typhoid":      "#FDD835", "tuberculosis": "#8E24AA",
    "meningitis":   "#FB8C00", "lassa_fever":  "#D81B60",
    "diarrhoeal":   "#43A047", "yellow_fever": "#F4511E",
}

# ── Database helpers ──────────────────────────────────────────────────────────
@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

@st.cache_data(ttl=300)
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)

@st.cache_data(ttl=60)
def table_counts() -> dict:
    tables = [
        "lga", "disease_record", "surveillance_alert",
        "climate_health", "socioeconomic", "maternal_health",
        "data_quality_log", "feature_store",
    ]
    counts = {}
    with connect() as conn:
        for t in tables:
            try:
                counts[t] = conn.execute(
                    f"SELECT COUNT(*) FROM {t}"
                ).fetchone()[0]
            except Exception:
                counts[t] = 0
    return counts

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/7/79/Flag_of_Nigeria.svg",
        width=80,
    )
    st.title("Nigeria Health AI")
    st.caption("Data Refinery Dashboard")
    st.divider()

    page = st.radio(
        "Navigate",
        [
        "🏠 Overview",
        "🔍 Data Explorer",
        "🗺️ Disease Map",
        "👶 Maternal Health",
        "📊 Data Quality",
        "💡 Insights",
        "🔮 Predictions",        # ← add this line
        ],
        label_visibility="collapsed",
    )

    st.divider()
    counts = table_counts()
    total  = sum(counts.values())
    st.caption("**Database status**")
    st.metric("Total rows", f"{total:,}")
    if Path(DB_PATH).exists():
        size_mb = Path(DB_PATH).stat().st_size / 1_048_576
        st.metric("DB size", f"{size_mb:.1f} MB")
    st.caption(f"📍 `{DB_PATH}`")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🇳🇬 Nigeria Health AI — Data Refinery")
    st.caption(
        "Disease pattern detection trained on Nigerian-native data across "
        "770 LGAs, 8 diseases, and 6 geopolitical zones."
    )
    st.divider()

    # Row 1 — key counts
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📍 LGAs",             f"{counts.get('lga', 0):,}")
    c2.metric("🦠 Disease Records",  f"{counts.get('disease_record', 0):,}")
    c3.metric("🌧️ Climate Records",  f"{counts.get('climate_health', 0):,}")
    c4.metric("🧠 Feature Rows",     f"{counts.get('feature_store', 0):,}")

    # Row 2
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("💰 Socioeconomic",    f"{counts.get('socioeconomic', 0):,}")
    c6.metric("👶 Maternal Health",  f"{counts.get('maternal_health', 0):,}")
    c7.metric("🚨 Alerts",           f"{counts.get('surveillance_alert', 0):,}")
    c8.metric("✅ Quality Log",       f"{counts.get('data_quality_log', 0):,}")

    st.divider()

    col1, col2 = st.columns(2)

    # Zone distribution
    with col1:
        st.subheader("Feature rows by zone")
        zone_df = query("""
            SELECT zone,
                   COUNT(*) AS rows,
                   ROUND(AVG(incidence_rate), 3) AS avg_incidence
            FROM feature_store
            GROUP BY zone ORDER BY zone
        """)
        if not zone_df.empty:
            fig = px.bar(
                zone_df, x="zone", y="rows",
                color="zone", color_discrete_map=ZONE_COLOURS,
                text="rows",
                labels={"zone": "Zone", "rows": "Feature Rows"},
            )
            fig.update_traces(texttemplate="%{text:,}", textposition="outside")
            fig.update_layout(
                showlegend=False, height=320,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Disease coverage
    with col2:
        st.subheader("Disease coverage")
        dis_df = query("""
            SELECT disease_category, COUNT(*) AS rows
            FROM feature_store
            GROUP BY disease_category ORDER BY rows DESC
        """)
        if not dis_df.empty:
            fig2 = px.pie(
                dis_df, names="disease_category", values="rows",
                color="disease_category",
                color_discrete_map=DISEASE_COLOURS,
                hole=0.45,
            )
            fig2.update_layout(
                height=320,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="v", x=1.0),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Weekly trend
    st.subheader("National weekly incidence trend")
    trend_df = query("""
        SELECT epi_year, epi_week,
               ROUND(AVG(incidence_rate), 4) AS mean_incidence,
               disease_category
        FROM feature_store
        WHERE incidence_rate IS NOT NULL
        GROUP BY epi_year, epi_week, disease_category
        ORDER BY epi_year, epi_week
    """)
    if not trend_df.empty:
        trend_df["period"] = (
            trend_df["epi_year"].astype(str) + "-W"
            + trend_df["epi_week"].astype(str).str.zfill(2)
        )
        fig3 = px.line(
            trend_df, x="period", y="mean_incidence",
            color="disease_category",
            color_discrete_map=DISEASE_COLOURS,
            labels={
                "period": "Epidemiological Week",
                "mean_incidence": "Mean Incidence / 10k population",
            },
        )
        fig3.update_layout(
            height=360,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickangle=45, nticks=20),
        )
        st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA EXPLORER
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Data Explorer":
    st.title("🔍 Data Explorer")
    st.caption("Browse, filter, and export any table from the Nigeria health database.")

    TABLE_OPTIONS = {
        "feature_store":      "Feature Store (AI training data)",
        "disease_record":     "Disease Records",
        "maternal_health":    "Maternal Health",
        "socioeconomic":      "Socioeconomic Indicators",
        "climate_health":     "Climate & Environment",
        "surveillance_alert": "Surveillance Alerts",
        "lga":                "LGA Reference",
        "data_quality_log":   "Data Quality Log",
    }

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        selected_table = st.selectbox(
            "Table",
            options=list(TABLE_OPTIONS.keys()),
            format_func=lambda x: TABLE_OPTIONS[x],
        )
    with col2:
        row_limit = st.select_slider(
            "Rows to display",
            options=[100, 500, 1000, 5000, 10000],
            value=500,
        )

    # Filters
    with st.expander("⚙️ Filters", expanded=True):
        fcol1, fcol2, fcol3 = st.columns(3)
        zone_filter    = ""
        disease_filter = ""
        year_filter    = ""

        # Zone
        zones = query("SELECT DISTINCT zone FROM lga ORDER BY zone")["zone"].tolist()
        sel_zones = fcol1.multiselect("Zone", zones)
        if sel_zones:
            z_list = ",".join(f"'{z}'" for z in sel_zones)
            zone_filter = (
                f"AND zone IN ({z_list})"
                if selected_table == "lga"
                else f"AND l.zone IN ({z_list})"
            )

        # Disease
        if selected_table in ["feature_store", "disease_record", "surveillance_alert"]:
            diseases = query(
                "SELECT DISTINCT disease_category FROM feature_store ORDER BY 1"
            )["disease_category"].tolist()
            sel_dis = fcol2.multiselect("Disease", diseases)
            if sel_dis:
                d_list = ",".join(f"'{d}'" for d in sel_dis)
                disease_filter = f"AND disease_category IN ({d_list})"

        # Year
        if selected_table in ["feature_store", "socioeconomic",
                               "maternal_health", "climate_health"]:
            yr_col = "epi_year" if selected_table == "feature_store" else "year"
            yr_tbl = "feature_store" if selected_table == "feature_store" else selected_table
            years  = query(
                f"SELECT DISTINCT {yr_col} AS yr FROM {yr_tbl} ORDER BY 1"
            )["yr"].dropna().astype(int).tolist()
            if len(years) > 1:
                yr_range = fcol3.select_slider(
                    "Year range", options=years,
                    value=(min(years), max(years)),
                )
                year_filter = (
                    f"AND epi_year BETWEEN {yr_range[0]} AND {yr_range[1]}"
                    if selected_table == "feature_store"
                    else f"AND year BETWEEN {yr_range[0]} AND {yr_range[1]}"
                )
            elif years:
                year_filter = (
                    f"AND epi_year = {years[0]}"
                    if selected_table == "feature_store"
                    else f"AND year = {years[0]}"
                )

    QUERIES = {
        "feature_store": f"""
            SELECT fs.*, l.lga_name, l.state
            FROM feature_store fs
            JOIN lga l ON l.lga_id = fs.lga_id
            WHERE 1=1 {zone_filter} {disease_filter} {year_filter}
            ORDER BY fs.epi_year DESC, fs.epi_week DESC
            LIMIT {row_limit}
        """,
        "disease_record": f"""
            SELECT dr.*, l.lga_name, l.state, l.zone
            FROM disease_record dr
            JOIN lga l ON l.lga_id = dr.lga_id
            WHERE 1=1 {zone_filter} {disease_filter}
            ORDER BY dr.report_date DESC
            LIMIT {row_limit}
        """,
        "maternal_health": f"""
            SELECT mh.*, l.lga_name, l.state, l.zone
            FROM maternal_health mh
            JOIN lga l ON l.lga_id = mh.lga_id
            WHERE 1=1 {year_filter}
            ORDER BY mh.year DESC, mh.quarter DESC
            LIMIT {row_limit}
        """,
        "socioeconomic": f"""
            SELECT se.*, l.lga_name, l.state, l.zone
            FROM socioeconomic se
            JOIN lga l ON l.lga_id = se.lga_id
            WHERE 1=1 {year_filter}
            ORDER BY se.year DESC
            LIMIT {row_limit}
        """,
        "climate_health": f"""
            SELECT ch.*, l.lga_name, l.state, l.zone
            FROM climate_health ch
            JOIN lga l ON l.lga_id = ch.lga_id
            WHERE 1=1 {year_filter}
            ORDER BY ch.year DESC, ch.month DESC
            LIMIT {row_limit}
        """,
        "surveillance_alert": f"""
            SELECT sa.*, l.lga_name, l.state, l.zone
            FROM surveillance_alert sa
            JOIN lga l ON l.lga_id = sa.lga_id
            WHERE 1=1 {disease_filter}
            ORDER BY sa.alert_date DESC
            LIMIT {row_limit}
        """,
        "lga": f"""
            SELECT * FROM lga WHERE 1=1 {zone_filter}
            ORDER BY state, lga_name
            LIMIT {row_limit}
        """,
        "data_quality_log": f"""
            SELECT dql.*, l.lga_name, l.state, l.zone
            FROM data_quality_log dql
            LEFT JOIN lga l ON l.lga_id = dql.lga_id
            ORDER BY dql.check_date DESC
            LIMIT {row_limit}
        """,
    }

    df = query(QUERIES.get(
        selected_table, f"SELECT * FROM {selected_table} LIMIT {row_limit}"
    ))

    st.write(f"**{len(df):,} rows** · {TABLE_OPTIONS.get(selected_table, selected_table)}")
    st.dataframe(df, use_container_width=True, height=480)

    dc1, dc2 = st.columns([1, 4])
    with dc1:
        st.download_button(
            "⬇️ Download CSV",
            df.to_csv(index=False),
            file_name=f"{selected_table}_export.csv",
            mime="text/csv",
        )
    with dc2:
        st.download_button(
            "⬇️ Download JSON",
            df.to_json(orient="records", indent=2),
            file_name=f"{selected_table}_export.json",
            mime="application/json",
        )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DISEASE MAP
# ════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Disease Map":
    st.title("🗺️ Disease Incidence Map")
    st.caption("Spatial disease patterns across Nigerian zones and states.")

    diseases = query(
        "SELECT DISTINCT disease_category FROM feature_store ORDER BY 1"
    )["disease_category"].tolist()
    years = query(
        "SELECT DISTINCT epi_year FROM feature_store WHERE epi_year IS NOT NULL ORDER BY 1"
    )["epi_year"].dropna().astype(int).tolist()

    col1, col2, col3 = st.columns(3)
    sel_disease = col1.selectbox("Disease", diseases)
    sel_year    = col2.selectbox("Year", years, index=len(years) - 1 if years else 0)
    agg_by      = col3.radio("Aggregate by", ["Zone", "State"], horizontal=True)

    if agg_by == "Zone":
        map_df = query("""
            SELECT fs.zone,
                   ROUND(AVG(fs.incidence_rate), 4)   AS mean_incidence,
                   ROUND(MAX(fs.incidence_rate), 4)   AS max_incidence,
                   SUM(fs.active_alert_flag)           AS alerts,
                   ROUND(AVG(fs.reporting_weight), 2) AS data_quality,
                   COUNT(*)                            AS feature_rows
            FROM feature_store fs
            WHERE fs.disease_category = ?
              AND fs.epi_year = ?
            GROUP BY fs.zone
            ORDER BY mean_incidence DESC
        """, (sel_disease, sel_year))

        if map_df.empty:
            st.warning("No data for this selection.")
        else:
            mc1, mc2 = st.columns([2, 1])
            with mc1:
                fig = px.bar(
                    map_df, x="zone", y="mean_incidence",
                    color="zone", color_discrete_map=ZONE_COLOURS,
                    text="mean_incidence",
                    title=f"{sel_disease.replace('_',' ').title()} — Mean Incidence by Zone ({sel_year})",
                    labels={"zone": "Zone", "mean_incidence": "Mean Incidence / 10k"},
                )
                fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                fig.update_layout(
                    showlegend=False, height=380,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)
            with mc2:
                st.subheader("Zone summary")
                st.dataframe(
                    map_df[["zone", "mean_incidence", "alerts", "data_quality"]],
                    use_container_width=True, hide_index=True,
                )
    else:
        map_df = query("""
            SELECT l.state,
                   ROUND(AVG(fs.incidence_rate), 4) AS mean_incidence,
                   SUM(fs.active_alert_flag)         AS alerts,
                   COUNT(DISTINCT fs.lga_id)         AS lgas_covered
            FROM feature_store fs
            JOIN lga l ON l.lga_id = fs.lga_id
            WHERE fs.disease_category = ?
              AND fs.epi_year = ?
            GROUP BY l.state
            ORDER BY mean_incidence DESC
        """, (sel_disease, sel_year))

        if not map_df.empty:
            fig = px.bar(
                map_df.head(20), x="state", y="mean_incidence",
                color="mean_incidence",
                color_continuous_scale="Reds",
                title=f"{sel_disease.replace('_',' ').title()} — Top 20 States ({sel_year})",
                labels={"state": "State", "mean_incidence": "Mean Incidence / 10k"},
            )
            fig.update_layout(
                height=420, xaxis_tickangle=45,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Seasonal pattern
    st.subheader(f"Seasonal pattern — {sel_disease.replace('_',' ').title()}")
    seasonal_df = query("""
        SELECT season,
               ROUND(AVG(incidence_rate), 4) AS mean_incidence,
               COUNT(*) AS n
        FROM feature_store
        WHERE disease_category = ?
          AND season IS NOT NULL
        GROUP BY season
        ORDER BY mean_incidence DESC
    """, (sel_disease,))

    if not seasonal_df.empty:
        sc1, sc2 = st.columns(2)
        with sc1:
            fig_s = px.bar(
                seasonal_df, x="season", y="mean_incidence",
                color="season",
                color_discrete_sequence=["#FF9800", "#4CAF50", "#2196F3"],
                text="mean_incidence",
                labels={"season": "Season", "mean_incidence": "Mean Incidence / 10k"},
            )
            fig_s.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig_s.update_layout(
                showlegend=False, height=300,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_s, use_container_width=True)
        with sc2:
            st.caption("**Seasonal breakdown**")
            st.dataframe(seasonal_df, use_container_width=True, hide_index=True)

    # Climate correlation — removed OLS trendline (requires statsmodels)
    st.subheader("Climate correlation")
    clim_df = query("""
        SELECT ROUND(rainfall_mm, 0) AS rainfall_mm,
               ROUND(incidence_rate, 4) AS incidence_rate,
               zone, season
        FROM feature_store
        WHERE disease_category = ?
          AND rainfall_mm IS NOT NULL
          AND incidence_rate IS NOT NULL
        LIMIT 3000
    """, (sel_disease,))
    if not clim_df.empty:
        fig_c = px.scatter(
            clim_df, x="rainfall_mm", y="incidence_rate",
            color="zone", color_discrete_map=ZONE_COLOURS,
            symbol="season", opacity=0.55,
            labels={
                "rainfall_mm": "Monthly Rainfall (mm)",
                "incidence_rate": "Incidence / 10k",
            },
            title=f"Rainfall vs {sel_disease.replace('_',' ').title()} incidence",
        )
        fig_c.update_layout(
            height=380,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_c, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MATERNAL HEALTH
# ════════════════════════════════════════════════════════════════════════════
elif page == "👶 Maternal Health":
    st.title("👶 Maternal & Child Health")
    st.caption("LGA-level maternal and child health indicators.")

    years = query(
        "SELECT DISTINCT year FROM maternal_health ORDER BY year"
    )["year"].tolist()
    zones = query("SELECT DISTINCT zone FROM lga ORDER BY zone")["zone"].tolist()

    col1, col2 = st.columns(2)
    sel_year = col1.selectbox("Year", years, index=len(years) - 1 if years else 0)
    sel_zone = col2.multiselect("Zone", zones, default=zones)

    zone_sql = ""
    if sel_zone:
        z_list   = ",".join(f"'{z}'" for z in sel_zone)
        zone_sql = f"AND l.zone IN ({z_list})"

    mh_df = query(f"""
        SELECT mh.*, l.lga_name, l.state, l.zone
        FROM maternal_health mh
        JOIN lga l ON l.lga_id = mh.lga_id
        WHERE mh.year = ? {zone_sql}
        ORDER BY mh.maternal_mortality_ratio DESC
    """, (sel_year,))

    if mh_df.empty:
        st.warning("No maternal health data for this selection.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Avg ANC Coverage",
                  f"{mh_df['anc_coverage_pct'].mean():.1f}%")
        k2.metric("Avg Skilled Birth Attendance",
                  f"{mh_df['skilled_birth_pct'].mean():.1f}%")
        k3.metric("Avg MMR (per 100k live births)",
                  f"{mh_df['maternal_mortality_ratio'].mean():.0f}")
        k4.metric("Avg Stunting Rate",
                  f"{mh_df['stunting_rate_pct'].mean():.1f}%")

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Maternal mortality ratio by zone")
            zone_mmr = (
                mh_df.groupby("zone")["maternal_mortality_ratio"]
                .mean().reset_index()
            )
            fig = px.bar(
                zone_mmr, x="zone", y="maternal_mortality_ratio",
                color="zone", color_discrete_map=ZONE_COLOURS,
                text="maternal_mortality_ratio",
                labels={"zone": "Zone",
                        "maternal_mortality_ratio": "MMR per 100k"},
            )
            fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
            fig.update_layout(
                showlegend=False, height=340,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("ANC coverage vs skilled birth attendance")
            fig2 = px.scatter(
                mh_df, x="anc_coverage_pct", y="skilled_birth_pct",
                color="zone", color_discrete_map=ZONE_COLOURS,
                size="maternal_mortality_ratio",
                hover_data=["lga_name", "state"],
                labels={
                    "anc_coverage_pct":  "ANC Coverage (%)",
                    "skilled_birth_pct": "Skilled Birth Attendance (%)",
                },
            )
            fig2.update_layout(
                height=340,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Nutrition indicators by zone")
        nutri_df = (
            mh_df.groupby("zone")[
                ["stunting_rate_pct", "wasting_rate_pct", "exclusive_bf_pct"]
            ].mean().reset_index()
        )
        nutri_melt = nutri_df.melt(
            id_vars="zone",
            value_vars=["stunting_rate_pct", "wasting_rate_pct", "exclusive_bf_pct"],
            var_name="indicator", value_name="value",
        )
        nutri_melt["indicator"] = nutri_melt["indicator"].map({
            "stunting_rate_pct":  "Stunting Rate (%)",
            "wasting_rate_pct":   "Wasting Rate (%)",
            "exclusive_bf_pct":   "Exclusive Breastfeeding (%)",
        })
        fig3 = px.bar(
            nutri_melt, x="zone", y="value", color="indicator",
            barmode="group",
            labels={"zone": "Zone", "value": "%", "indicator": "Indicator"},
            color_discrete_sequence=["#E53935", "#FB8C00", "#43A047"],
        )
        fig3.update_layout(
            height=340,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("🚨 Highest maternal mortality LGAs")
        top_risk = mh_df.nlargest(10, "maternal_mortality_ratio")[[
            "lga_name", "state", "zone",
            "maternal_mortality_ratio", "anc_coverage_pct",
            "skilled_birth_pct", "stunting_rate_pct", "data_quality_score",
        ]]
        st.dataframe(top_risk, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — DATA QUALITY
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Quality":
    st.title("📊 Data Quality Audit")
    st.caption(
        "Transparency layer documenting completeness, reporting weights, "
        "and bias flags across all data sources."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Reporting weight by zone")
        rw_df = query("""
            SELECT zone,
                   ROUND(AVG(reporting_weight), 3) AS mean_weight,
                   ROUND(MIN(reporting_weight), 3) AS min_weight,
                   ROUND(MAX(reporting_weight), 3) AS max_weight
            FROM feature_store
            WHERE reporting_weight IS NOT NULL
            GROUP BY zone ORDER BY mean_weight
        """)
        if not rw_df.empty:
            fig = px.bar(
                rw_df, x="zone", y="mean_weight",
                color="zone", color_discrete_map=ZONE_COLOURS,
                text="mean_weight",
                labels={"zone": "Zone", "mean_weight": "Mean Reporting Weight (0–1)"},
            )
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_layout(
                showlegend=False, height=340,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(range=[0, 1.1]),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Data quality by LGA type")
        lga_type_df = query("""
            SELECT lga_type,
                   ROUND(AVG(reporting_weight), 3)  AS mean_weight,
                   ROUND(AVG(completeness_pct), 1)  AS mean_completeness,
                   COUNT(*) AS rows
            FROM feature_store
            WHERE lga_type IS NOT NULL
            GROUP BY lga_type
        """)
        if not lga_type_df.empty:
            lga_melt = lga_type_df.melt(
            id_vars="lga_type",
            value_vars=["mean_weight","mean_completeness"],
            var_name="Metric", value_name="Score"
            )
            lga_melt["Score"] = pd.to_numeric(lga_melt["Score"], errors="coerce")
            fig2 = px.bar(
             lga_melt, x="lga_type", y="Score", color="Metric",
             barmode="group",
             labels={"lga_type":"LGA Type"},
             color_discrete_sequence=["#2196F3","#4CAF50"],
            )
            fig2.update_layout(
                height=340,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    st.subheader("Records by data source")
    src_df = query("""
        SELECT source,
               COUNT(*) AS records,
               ROUND(AVG(data_quality_score), 2) AS avg_quality
        FROM disease_record
        GROUP BY source ORDER BY records DESC
    """)
    sc1, sc2 = st.columns(2)
    with sc1:
        if not src_df.empty:
            fig3 = px.pie(
                src_df, names="source", values="records",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig3.update_layout(
                height=300,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig3, use_container_width=True)
    with sc2:
        st.dataframe(src_df, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Lag feature coverage by disease")
    lag_df = query("""
        SELECT disease_category,
               COUNT(*) AS total,
               SUM(CASE WHEN incidence_lag_1w IS NOT NULL THEN 1 ELSE 0 END) AS lag_1w,
               SUM(CASE WHEN incidence_lag_4w IS NOT NULL THEN 1 ELSE 0 END) AS lag_4w,
               SUM(CASE WHEN incidence_lag_8w IS NOT NULL THEN 1 ELSE 0 END) AS lag_8w
        FROM feature_store
        GROUP BY disease_category
    """)
    if not lag_df.empty:
        lag_df["lag_1w_pct"] = (lag_df["lag_1w"] / lag_df["total"] * 100).round(1)
        lag_df["lag_4w_pct"] = (lag_df["lag_4w"] / lag_df["total"] * 100).round(1)
        lag_df["lag_8w_pct"] = (lag_df["lag_8w"] / lag_df["total"] * 100).round(1)

        fig4 = px.bar(
            lag_df.melt(
                id_vars="disease_category",
                value_vars=["lag_1w_pct", "lag_4w_pct", "lag_8w_pct"],
                var_name="lag", value_name="coverage_pct",
            ),
            x="disease_category", y="coverage_pct", color="lag",
            barmode="group",
            labels={
                "disease_category": "Disease",
                "coverage_pct": "Coverage (%)",
                "lag": "Lag Window",
            },
            color_discrete_sequence=["#4CAF50", "#FF9800", "#F44336"],
        )
        fig4.update_layout(
            height=360, xaxis_tickangle=30,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(range=[0, 105]),
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    st.subheader("Validity threat assessment")
    threats = [
        ("Reporting bias",
         "Urban LGAs over-represented in disease records",
         "reporting_weight < 0.6 in rural zones",
         "⚠️ Mitigated via sample weighting"),
        ("Selection bias",
         "Hospital records skew toward higher-SES patients",
         "nhia_coverage_pct < 5% in NE/NW zones",
         "⚠️ NHIA coverage included as explicit model feature"),
        ("Label noise",
         "Inconsistent ICD-10 coding across facilities",
         "icd10_code NULL in raw disease records",
         "✅ ICD-10 harmoniser applied in refinery"),
        ("Temporal confounding",
         "COVID-19 disrupted health-seeking behaviour (2020–2022)",
         "Post-2020 data requires careful interpretation",
         "⚠️ TimeSeriesSplit CV prevents data leakage"),
        ("Ecological fallacy",
         "LGA-level patterns do not imply individual-level risk",
         "All predictions are population-level only",
         "✅ Documented in all model outputs"),
        ("External validity",
         "Zone-specific models may not generalise across zones",
         "Zone stratification applied in feature store",
         "✅ All 6 zones represented in training data"),
        ("Rural data gap",
         "Rural LGAs have systematically lower reporting completeness",
         "lga_type=rural completeness < 60%",
         "✅ Rural gap imputer applied in refinery"),
        ("Synthetic maternal data",
         "Maternal health baseline generated from NDHS zone distributions",
         "data_quality_score = 0.45 for synthetic records",
         "⚠️ Replace with real DHS microdata when available"),
    ]
    threat_df = pd.DataFrame(
        threats,
        columns=["Threat", "Description", "Signal", "Mitigation"],
    )
    st.dataframe(threat_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 6 — INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
elif page == "💡 Insights":
    st.title("💡 Insights")
    st.caption(
        "Top risk LGAs, disease-poverty relationships, "
        "and model readiness across all disease categories."
    )

    diseases = query(
        "SELECT DISTINCT disease_category FROM feature_store ORDER BY 1"
    )["disease_category"].tolist()
    sel_disease = st.selectbox("Select disease", diseases)

    # Top risk LGAs
    st.subheader("🚨 Highest incidence LGAs")
    top_lga_df = query("""
        SELECT l.lga_name, l.state, l.zone, l.lga_type,
               ROUND(AVG(fs.incidence_rate), 4)        AS mean_incidence,
               ROUND(MAX(fs.incidence_rate), 4)        AS peak_incidence,
               ROUND(AVG(fs.reporting_weight), 2)      AS data_quality,
               SUM(fs.active_alert_flag)               AS total_alerts,
               ROUND(AVG(fs.poverty_headcount_pct), 1) AS poverty_pct,
               ROUND(AVG(fs.rainfall_mm), 1)           AS avg_rainfall_mm
        FROM feature_store fs
        JOIN lga l ON l.lga_id = fs.lga_id
        WHERE fs.disease_category = ?
          AND fs.incidence_rate IS NOT NULL
        GROUP BY fs.lga_id
        ORDER BY mean_incidence DESC
        LIMIT 20
    """, (sel_disease,))

    if not top_lga_df.empty:
        col1, col2 = st.columns([3, 2])
        with col1:
            fig = px.bar(
                top_lga_df.head(15), x="lga_name", y="mean_incidence",
                color="zone", color_discrete_map=ZONE_COLOURS,
                text="mean_incidence",
                title=f"Top 15 LGAs — {sel_disease.replace('_',' ').title()}",
                labels={"lga_name": "LGA",
                        "mean_incidence": "Mean Incidence / 10k"},
            )
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig.update_layout(
                height=400, xaxis_tickangle=45,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.caption("**Top 20 highest-incidence LGAs**")
            st.dataframe(
                top_lga_df[[
                    "lga_name", "state", "zone",
                    "mean_incidence", "data_quality", "total_alerts",
                ]],
                use_container_width=True, hide_index=True, height=400,
            )

    st.divider()

    # Poverty × disease
    st.subheader("Poverty rate vs disease incidence")
    pov_df = query("""
        SELECT ROUND(se.poverty_headcount_pct, 0) AS poverty_pct,
               ROUND(AVG(fs.incidence_rate), 4)   AS mean_incidence,
               fs.zone
        FROM feature_store fs
        JOIN socioeconomic se
            ON se.lga_id = fs.lga_id
            AND se.year  = fs.epi_year
        WHERE fs.disease_category = ?
          AND se.poverty_headcount_pct IS NOT NULL
          AND fs.incidence_rate IS NOT NULL
        GROUP BY ROUND(se.poverty_headcount_pct, 0), fs.zone
        LIMIT 2000
    """, (sel_disease,))
    if not pov_df.empty:
        fig_p = px.scatter(
            pov_df, x="poverty_pct", y="mean_incidence",
            color="zone", color_discrete_map=ZONE_COLOURS,
            opacity=0.65,
            labels={
                "poverty_pct":     "Poverty Headcount (%)",
                "mean_incidence":  "Mean Incidence / 10k",
            },
            title=f"Poverty vs {sel_disease.replace('_',' ').title()} incidence by zone",
        )
        fig_p.update_layout(
            height=380,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_p, use_container_width=True)
    else:
        st.info("Socioeconomic data not yet joined for this disease. "
                "Run fix_socioeconomic.py to populate the socioeconomic table.")

    st.divider()

    # Model readiness — fixed query joining socioeconomic directly
    st.subheader("Model readiness by disease")
    readiness_df = query("""
        SELECT fs.disease_category,
               COUNT(DISTINCT fs.lga_id)                                          AS lgas_covered,
               COUNT(*)                                                            AS total_rows,
               ROUND(AVG(fs.reporting_weight), 2)                                 AS avg_quality,
               ROUND(100.0 * SUM(CASE WHEN fs.incidence_lag_4w IS NOT NULL
                                      THEN 1 ELSE 0 END) / COUNT(*), 1)           AS lag4w_pct,
               ROUND(100.0 * SUM(CASE WHEN fs.rainfall_mm IS NOT NULL
                                      THEN 1 ELSE 0 END) / COUNT(*), 1)           AS climate_pct,
               ROUND(100.0 * SUM(CASE WHEN se.poverty_headcount_pct IS NOT NULL
                                      THEN 1 ELSE 0 END) / COUNT(*), 1)           AS ses_pct
        FROM feature_store fs
        LEFT JOIN socioeconomic se
            ON se.lga_id = fs.lga_id
            AND se.year  = fs.epi_year
        GROUP BY fs.disease_category
        ORDER BY avg_quality DESC
    """)

    if not readiness_df.empty:
        readiness_df["overall_score"] = (
            readiness_df["avg_quality"] * 40
            + readiness_df["lag4w_pct"] * 0.30
            + readiness_df["climate_pct"] * 0.15
            + readiness_df["ses_pct"] * 0.15
        ).round(1)

        def badge(score):
            if score >= 70:   return "🟢 Ready"
            if score >= 50:   return "🟡 Partial"
            return "🔴 Needs data"

        readiness_df["status"] = readiness_df["overall_score"].apply(badge)

        fig_r = px.bar(
            readiness_df, x="disease_category", y="overall_score",
            color="status",
            color_discrete_map={
                "🟢 Ready":      "#4CAF50",
                "🟡 Partial":    "#FF9800",
                "🔴 Needs data": "#F44336",
            },
            text="overall_score",
            labels={"disease_category": "Disease",
                    "overall_score": "Readiness Score"},
            title="AI Model Readiness Score by Disease",
        )
        fig_r.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_r.update_layout(
            height=380, xaxis_tickangle=30,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(range=[0, 105]),
        )
        st.plotly_chart(fig_r, use_container_width=True)

        st.caption(
            "**Readiness score** combines data quality weight (40%), "
            "lag feature coverage (30%), climate coverage (15%), "
            "and socioeconomic coverage (15%)."
        )
        st.dataframe(readiness_df, use_container_width=True, hide_index=True)
        # ════════════════════════════════════════════════════════════════════════════
# PAGE 7 — PREDICTIONS
# ════════════════════════════════════════════════════════════════════════════
       # ════════════════════════════════════════════════════════════════════════════
# PREDICTIONS PAGE — paste this at the bottom of app.py
# Replaces the existing elif page == "🔮 Predictions": block entirely
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predictions":
    import json
    from pathlib import Path

    st.title("🔮 RLRF Outbreak Predictions")
    st.caption(
        "4-week ahead outbreak forecasting using N-Step SARSA + Random Forest. "
        "Trained exclusively on Nigerian epidemiological data."
    )
    st.divider()

    # ── Load model registry ───────────────────────────────────────────────────
    REGISTRY_PATH = Path("models/model_registry.json")
    registry      = {}
    diseases_meta = []

    if REGISTRY_PATH.exists():
        registry      = json.loads(REGISTRY_PATH.read_text())
        diseases_meta = registry.get("diseases", [])
    else:
        st.warning(
            "Model registry not found. Ensure `models/model_registry.json` "
            "is present in the repository."
        )

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Model Performance Summary (your core novelty)
    # ════════════════════════════════════════════════════════════════════════
    if diseases_meta:
        st.subheader("📈 Model Performance Summary")
        st.caption(
            "R² scores from 5-fold TimeSeriesSplit cross-validation. "
            "All models trained exclusively on Nigerian data."
        )

        # Aggregate metrics row
        mean_r2   = sum(d.get("r2", 0) for d in diseases_meta) / len(diseases_meta)
        mean_gain = sum(d.get("sarsa_improvement_r2", 0) for d in diseases_meta) / len(diseases_meta)
        mean_acc  = sum(d.get("outbreak_accuracy", 0) for d in diseases_meta) / len(diseases_meta)
        agg       = registry.get("aggregate_performance", {})

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Mean R² (RLRF)",       f"{mean_r2:.4f}")
        k2.metric("Mean SARSA Gain",      f"+{mean_gain:.4f}",
                  delta=f"+{mean_gain:.4f} vs RF-only")
        k3.metric("Mean Outbreak Accuracy", f"{mean_acc:.1%}")
        k4.metric("Diseases Modelled",    str(len(diseases_meta)))

        st.divider()

        # Per-disease R² cards
        cols = st.columns(4)
        for i, d in enumerate(diseases_meta):
            col  = cols[i % 4]
            r2   = d.get("r2", 0)
            gain = d.get("sarsa_improvement_r2", 0)
            name = d["disease"].replace("_", " ").title()
            col.metric(
                label=name,
                value=f"R² {r2:.4f}",
                delta=f"+{gain:.4f} vs RF-only",
            )

        st.divider()

        # ── Ablation Study bar chart ──────────────────────────────────────────
        st.subheader("🔬 Ablation Study — RLRF vs RF-Only (Table 3)")
        st.caption(
            "Demonstrates the contribution of N-Step SARSA to predictive performance. "
            "This is the core novelty of the RLRF architecture."
        )

        ablation_rows = []
        for d in diseases_meta:
            ablation_rows.append({
                "Disease":     d["disease"].replace("_", " ").title(),
                "RLRF R²":    round(d.get("r2", 0), 4),
                "RF-only R²": round(d.get("r2_no_sarsa", 0), 4),
                "SARSA Gain": round(d.get("sarsa_improvement_r2", 0), 4),
                "MAE":        round(d.get("mae", 0), 4),
                "RMSE":       round(d.get("rmse", 0), 4),
                "Accuracy":   f"{d.get('outbreak_accuracy', 0):.1%}",
            })
        ablation_df = pd.DataFrame(ablation_rows)

        # Grouped bar — RLRF vs RF-only
        chart_df = ablation_df.melt(
            id_vars="Disease",
            value_vars=["RLRF R²", "RF-only R²"],
            var_name="Model", value_name="R²",
        )
        fig_abl = px.bar(
            chart_df, x="Disease", y="R²", color="Model",
            barmode="group",
            color_discrete_map={
                "RLRF R²":    "#2196F3",
                "RF-only R²": "#B0BEC5",
            },
            text="R²",
            labels={"Disease": "Disease", "R²": "R² Score"},
            title="RLRF vs RF-Only: R² Comparison Across All Diseases",
        )
        fig_abl.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_abl.update_layout(
            height=420,
            yaxis=dict(range=[0, 0.80]),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig_abl, use_container_width=True)

        # SARSA gain bar
        fig_gain = px.bar(
            ablation_df, x="Disease", y="SARSA Gain",
            color="SARSA Gain",
            color_continuous_scale=["#90CAF9", "#1565C0"],
            text="SARSA Gain",
            title="N-Step SARSA Contribution (ΔR²) per Disease",
            labels={"SARSA Gain": "ΔR² (SARSA improvement)"},
        )
        fig_gain.update_traces(texttemplate="+%{text:.4f}", textposition="outside")
        fig_gain.update_layout(
            height=360,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_gain, use_container_width=True)

        # Full metrics table
        st.caption("**Full metrics table (suitable for thesis Table 3)**")
        st.dataframe(ablation_df, use_container_width=True, hide_index=True)

        # Download as CSV for thesis
        st.download_button(
            "⬇️ Download Table 3 (CSV)",
            ablation_df.to_csv(index=False),
            "ablation_study_table3.csv",
            "text/csv",
        )

        st.divider()

        # ── Zone-level performance ────────────────────────────────────────────
        st.subheader("🗺️ Zone-Stratified Model Performance")
        st.caption(
            "R² scores by geopolitical zone reveal how prediction accuracy "
            "varies with data quality and disease dynamics across Nigeria."
        )

        sel_disease_perf = st.selectbox(
            "Select disease for zone breakdown",
            [d["disease"].replace("_", " ").title() for d in diseases_meta],
            key="zone_perf_select",
        )
        sel_key = sel_disease_perf.replace(" ", "_").lower()
        disease_entry = next(
            (d for d in diseases_meta if d["disease"] == sel_key), None
        )

        if disease_entry and "zone_metrics" in disease_entry:
            zone_rows = []
            for zone, metrics in disease_entry["zone_metrics"].items():
                zone_rows.append({
                    "Zone": zone,
                    "R²":   round(metrics.get("r2", 0), 4),
                    "MAE":  round(metrics.get("mae", 0), 4),
                    "N":    metrics.get("n", 0),
                })
            zone_df = pd.DataFrame(zone_rows).sort_values("R²", ascending=False)

            zc1, zc2 = st.columns([2, 1])
            with zc1:
                fig_zone = px.bar(
                    zone_df, x="Zone", y="R²",
                    color="Zone", color_discrete_map=ZONE_COLOURS,
                    text="R²",
                    title=f"{sel_disease_perf} — R² by Geopolitical Zone",
                    labels={"Zone": "Zone", "R²": "R² Score"},
                )
                fig_zone.update_traces(
                    texttemplate="%{text:.4f}", textposition="outside"
                )
                fig_zone.update_layout(
                    showlegend=False, height=340,
                    yaxis=dict(range=[0, 0.85]),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_zone, use_container_width=True)
            with zc2:
                st.caption("**Zone metrics**")
                st.dataframe(zone_df, use_container_width=True, hide_index=True)

        st.divider()

        # ── Ethical notes ─────────────────────────────────────────────────────
        ethical = registry.get("ethical_notes", [])
        if ethical:
            st.subheader("📋 Ethical Framework")
            for note in ethical:
                st.markdown(f"✅ {note}")
            st.divider()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Live 4-week Forecasts
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("📡 Live 4-Week Outbreak Forecasts")
    st.caption(
        "Predicted incidence rates per LGA for the next 4 epidemiological weeks."
    )

    try:
        pred_count = query(
            "SELECT COUNT(*) AS n FROM model_predictions"
        )["n"].iloc[0]
    except Exception:
        pred_count = 0

    if pred_count == 0:
        st.info(
            "Predictions not yet generated. "
            "Run `python export_predictions.py` locally, "
            "then push the updated `nigeria.db` to GitHub."
        )
    else:
        st.success(f"✅ {pred_count:,} predictions available across all diseases.")

        diseases_pred = query(
            "SELECT DISTINCT disease_category FROM model_predictions ORDER BY 1"
        )["disease_category"].tolist()
        zones_all = query(
            "SELECT DISTINCT zone FROM lga ORDER BY zone"
        )["zone"].tolist()

        pc1, pc2 = st.columns(2)
        sel_dis_pred  = pc1.selectbox("Disease", diseases_pred, key="pred_disease")
        sel_zone_pred = pc2.multiselect("Zone", zones_all,
                                        default=zones_all, key="pred_zone")

        zone_sql = ""
        if sel_zone_pred:
            z_list   = ",".join(f"'{z}'" for z in sel_zone_pred)
            zone_sql = f"AND l.zone IN ({z_list})"

        pred_df = query(f"""
            SELECT mp.lga_id, l.lga_name, l.state, l.zone,
                   mp.predicted_year, mp.predicted_week,
                   ROUND(mp.predicted_incidence, 4) AS predicted_incidence,
                   ROUND(mp.lower_bound, 4)         AS lower_bound,
                   ROUND(mp.upper_bound, 4)         AS upper_bound,
                   mp.model_type
            FROM model_predictions mp
            JOIN lga l ON l.lga_id = mp.lga_id
            WHERE mp.disease_category = ? {zone_sql}
            ORDER BY mp.predicted_incidence DESC
            LIMIT 2000
        """, (sel_dis_pred,))

        if not pred_df.empty:
            pk1, pk2, pk3 = st.columns(3)
            pk1.metric("LGAs with forecasts",
                       f"{pred_df['lga_id'].nunique():,}")
            pk2.metric("Mean predicted incidence",
                       f"{pred_df['predicted_incidence'].mean():.4f}")
            pk3.metric("Max predicted incidence",
                       f"{pred_df['predicted_incidence'].max():.4f}")

            # Zone forecast bar
            zone_pred = (
                pred_df.groupby("zone")["predicted_incidence"]
                .mean().reset_index()
                .rename(columns={"predicted_incidence": "mean_predicted"})
                .sort_values("mean_predicted", ascending=False)
            )
            fig_pred = px.bar(
                zone_pred, x="zone", y="mean_predicted",
                color="zone", color_discrete_map=ZONE_COLOURS,
                text="mean_predicted",
                title=f"4-Week Ahead Forecast — {sel_dis_pred.replace('_',' ').title()} by Zone",
                labels={"zone": "Zone", "mean_predicted": "Predicted Incidence / 10k"},
            )
            fig_pred.update_traces(
                texttemplate="%{text:.4f}", textposition="outside"
            )
            fig_pred.update_layout(
                showlegend=False, height=360,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            # Top risk LGAs
            st.subheader("🚨 Highest forecast LGAs")
            top_pred = (
                pred_df.groupby(["lga_name", "state", "zone"])
                ["predicted_incidence"].mean().reset_index()
                .nlargest(15, "predicted_incidence")
                .round(4)
            )
            fc1, fc2 = st.columns([2, 1])
            with fc1:
                fig_top = px.bar(
                    top_pred, x="lga_name", y="predicted_incidence",
                    color="zone", color_discrete_map=ZONE_COLOURS,
                    text="predicted_incidence",
                    labels={"lga_name": "LGA",
                            "predicted_incidence": "Predicted Incidence / 10k"},
                )
                fig_top.update_traces(
                    texttemplate="%{text:.4f}", textposition="outside"
                )
                fig_top.update_layout(
                    height=380, xaxis_tickangle=45,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_top, use_container_width=True)
            with fc2:
                st.dataframe(top_pred, use_container_width=True,
                             hide_index=True, height=380)

            # Full table with confidence intervals
            st.subheader("Full forecast table with 90% prediction intervals")
            st.dataframe(
                pred_df[[
                    "lga_name", "state", "zone",
                    "predicted_year", "predicted_week",
                    "predicted_incidence", "lower_bound", "upper_bound",
                ]],
                use_container_width=True, hide_index=True, height=400,
            )
            st.download_button(
                "⬇️ Download forecasts (CSV)",
                pred_df.to_csv(index=False),
                f"forecasts_{sel_dis_pred}.csv",
                "text/csv",
            )
        else:
            st.info("No predictions available for this selection.")