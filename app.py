"""
app.py — Nigeria Health AI Data Refinery Dashboard
===================================================
Single-file Streamlit dashboard with 4 pages:
  1. 🏠 Overview       — database health, key metrics
  2. 🔍 Data Explorer  — browse, filter, download all tables
  3. 🗺️  Disease Map    — incidence choropleth by zone/LGA
  4. 👶 Maternal Health — maternal & child indicators
  5. 📊 Data Quality   — completeness, bias flags, reporting lag
  6. 💡 Insights       — AI-ready summaries, top risk LGAs

Run:
    streamlit run app.py

Requirements:
    pip install streamlit plotly pandas
"""

import sqlite3
import os
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Config ──────────────────────────────────────────────────────────────────
DB_PATH = os.getenv("DB_PATH", "./nigeria.db")

# ── Auto-download database from Google Drive if not present ─────────────────
def _download_db():
    """Download nigeria.db from Google Drive on first run (Streamlit Cloud)."""
    db = Path(DB_PATH)
    if db.exists() and db.stat().st_size > 10_000_000:
        return  # Already downloaded (>10MB means real DB)
    
    GDRIVE_FILE_ID = "1b1UJ058XGy80W-iH0tCFmSFxX6LPexLb"
    url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}&confirm=t"
    
    try:
        import requests
        st.info("⏳ First run: downloading database (~160MB). This takes ~2 minutes...")
        progress = st.progress(0, text="Connecting to database server...")
        
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 160_000_000))
            downloaded = 0
            with open(DB_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        pct = min(downloaded / total, 1.0)
                        progress.progress(pct, text=f"Downloading... {downloaded/1e6:.0f}MB / {total/1e6:.0f}MB")
        
        progress.progress(1.0, text="✅ Database ready!")
        st.success("Database downloaded successfully! Refreshing...")
        st.rerun()
    except Exception as e:
        st.warning(f"⚠️ Could not download database: {e}. Running in demo mode with limited data.")

_download_db()

st.set_page_config(
    page_title="Nigeria Health AI Refinery",
    page_icon="🇳🇬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colours ──────────────────────────────────────────────────────────────────
ZONE_COLOURS = {
    "NC": "#4CAF50", "NE": "#F44336", "NW": "#FF9800",
    "SE": "#2196F3", "SS": "#9C27B0", "SW": "#00BCD4",
}
DISEASE_COLOURS = {
    "malaria": "#E53935", "cholera": "#1E88E5",
    "typhoid": "#FDD835", "tuberculosis": "#8E24AA",
    "meningitis": "#FB8C00", "lassa_fever": "#D81B60",
    "diarrhoeal": "#43A047", "yellow_fever": "#F4511E",
}

# ── DB helpers ────────────────────────────────────────────────────────────────
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
                counts[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            except Exception:
                counts[t] = 0
    return counts

# ── Sidebar navigation ────────────────────────────────────────────────────────
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
            "🤖 Predictions",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    counts = table_counts()
    st.caption("**Database status**")
    total = sum(counts.values())
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
        "AI-native disease pattern detection trained exclusively on Nigerian data. "
        "No Western benchmarks. No imported assumptions."
    )
    st.divider()

    # Table count cards
    cols = st.columns(4)
    cards = [
        ("LGAs", "lga", "📍"),
        ("Disease Records", "disease_record", "🦠"),
        ("Climate Records", "climate_health", "🌧️"),
        ("Feature Rows", "feature_store", "🧠"),
    ]
    for col, (label, table, icon) in zip(cols, cards):
        col.metric(f"{icon} {label}", f"{counts.get(table, 0):,}")

    cols2 = st.columns(4)
    cards2 = [
        ("Socioeconomic", "socioeconomic", "💰"),
        ("Maternal Health", "maternal_health", "👶"),
        ("Surveillance Alerts", "surveillance_alert", "🚨"),
        ("Quality Log", "data_quality_log", "✅"),
    ]
    for col, (label, table, icon) in zip(cols2, cards2):
        col.metric(f"{icon} {label}", f"{counts.get(table, 0):,}")

    st.divider()

    # Zone distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Feature rows by zone")
        zone_df = query("""
            SELECT zone, COUNT(*) as rows,
                   ROUND(AVG(incidence_rate),2) as avg_incidence
            FROM feature_store GROUP BY zone ORDER BY zone
        """)
        if not zone_df.empty:
            fig = px.bar(
                zone_df, x="zone", y="rows",
                color="zone",
                color_discrete_map=ZONE_COLOURS,
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

    with col2:
        st.subheader("Disease coverage")
        dis_df = query("""
            SELECT disease_category, COUNT(*) as rows
            FROM feature_store GROUP BY disease_category ORDER BY rows DESC
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

    # Incidence trend
    st.subheader("National weekly incidence trend (all diseases)")
    trend_df = query("""
        SELECT epi_year, epi_week,
               ROUND(AVG(incidence_rate), 3) as mean_incidence,
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
            labels={"period": "Epi Week", "mean_incidence": "Mean Incidence / 10k"},
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
    st.caption("Browse, filter, and download any table from nigeria.db")

    TABLE_OPTIONS = {
        "feature_store":      "Feature Store (AI training data)",
        "disease_record":     "Disease Records",
        "maternal_health":    "Maternal Health",
        "socioeconomic":      "Socioeconomic",
        "climate_health":     "Climate & Environment",
        "surveillance_alert": "Surveillance Alerts",
        "lga":                "LGA Reference",
        "data_quality_log":   "Data Quality Log",
    }

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        selected_table = st.selectbox(
            "Table", options=list(TABLE_OPTIONS.keys()),
            format_func=lambda x: TABLE_OPTIONS[x],
        )
    with col2:
        row_limit = st.select_slider(
            "Rows to display", options=[100, 500, 1000, 5000, 10000], value=500
        )
    with col3:
        st.write("")
        st.write("")
        refresh = st.button("🔄 Refresh")

    # Dynamic filters
    with st.expander("⚙️ Filters", expanded=True):
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)

        zone_filter = ""
        disease_filter = ""
        year_filter = ""
        state_filter = ""

        # Zone filter (tables with zone column)
        zone_tables = ["feature_store", "disease_record", "lga"]
        if selected_table in zone_tables or selected_table == "lga":
            zones = query("SELECT DISTINCT zone FROM lga ORDER BY zone")["zone"].tolist()
            selected_zones = fcol1.multiselect("Zone", zones, default=[])
            if selected_zones:
                z_list = ",".join(f"'{z}'" for z in selected_zones)
                if selected_table == "lga":
                    zone_filter = f"AND zone IN ({z_list})"
                else:
                    zone_filter = f"AND l.zone IN ({z_list})"

        # Disease filter
        if selected_table in ["feature_store", "disease_record", "surveillance_alert"]:
            diseases = query(
                "SELECT DISTINCT disease_category FROM feature_store ORDER BY 1"
            )["disease_category"].tolist()
            sel_diseases = fcol2.multiselect("Disease", diseases, default=[])
            if sel_diseases:
                d_list = ",".join(f"'{d}'" for d in sel_diseases)
                disease_filter = f"AND disease_category IN ({d_list})"

        # Year filter
        year_tables = ["feature_store", "socioeconomic", "maternal_health", "climate_health"]
        if selected_table in year_tables:
            years = query(f"SELECT DISTINCT epi_year as yr FROM feature_store ORDER BY 1"
                          if selected_table == "feature_store"
                          else f"SELECT DISTINCT year as yr FROM {selected_table} ORDER BY 1"
                         )["yr"].dropna().astype(int).tolist()
            if years and len(years) > 1:
                yr_range = fcol3.select_slider(
                    "Year range",
                    options=years,
                    value=(min(years), max(years)),
                )
                if selected_table == "feature_store":
                    year_filter = f"AND epi_year BETWEEN {yr_range[0]} AND {yr_range[1]}"
                else:
                    year_filter = f"AND year BETWEEN {yr_range[0]} AND {yr_range[1]}"
            elif years:
                fcol3.info(f"Year: {years[0]}")
                if selected_table == "feature_store":
                    year_filter = f"AND epi_year = {years[0]}"
                else:
                    year_filter = f"AND year = {years[0]}"

    # Build query per table
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

    df = query(QUERIES.get(selected_table, f"SELECT * FROM {selected_table} LIMIT {row_limit}"))

    st.write(f"**{len(df):,} rows** · {selected_table}")
    st.dataframe(df, use_container_width=True, height=480)

    # Download
    dcol1, dcol2 = st.columns([1, 4])
    with dcol1:
        csv = df.to_csv(index=False)
        st.download_button(
            "⬇️ Download CSV", csv,
            file_name=f"{selected_table}_export.csv",
            mime="text/csv",
        )
    with dcol2:
        json_str = df.to_json(orient="records", indent=2)
        st.download_button(
            "⬇️ Download JSON", json_str,
            file_name=f"{selected_table}_export.json",
            mime="application/json",
        )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DISEASE MAP
# ════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Disease Map":
    st.title("🗺️ Disease Incidence Map")
    st.caption("Spatial disease patterns across Nigerian zones and LGAs")

    col1, col2, col3 = st.columns(3)
    diseases = query(
        "SELECT DISTINCT disease_category FROM feature_store ORDER BY 1"
    )["disease_category"].tolist()
    years = query(
        "SELECT DISTINCT epi_year FROM feature_store ORDER BY 1"
    )["epi_year"].dropna().astype(int).tolist()

    sel_disease = col1.selectbox("Disease", diseases)
    sel_year    = col2.selectbox("Year", years, index=len(years)-1)
    agg_by      = col3.radio("Aggregate by", ["Zone", "State"], horizontal=True)

    # Zone-level incidence
    if agg_by == "Zone":
        map_df = query("""
            SELECT fs.zone,
                   ROUND(AVG(fs.incidence_rate), 3)      AS mean_incidence,
                   ROUND(MAX(fs.incidence_rate), 3)      AS max_incidence,
                   SUM(fs.active_alert_flag)              AS alerts,
                   ROUND(AVG(fs.reporting_weight), 2)    AS data_quality,
                   COUNT(*)                               AS feature_rows
            FROM feature_store fs
            WHERE fs.disease_category = ?
              AND fs.epi_year = ?
            GROUP BY fs.zone
            ORDER BY mean_incidence DESC
        """, (sel_disease, sel_year))

        if map_df.empty:
            st.warning("No data for this combination. Try a different disease or year.")
        else:
            mcol1, mcol2 = st.columns([2, 1])
            with mcol1:
                fig = px.bar(
                    map_df, x="zone", y="mean_incidence",
                    color="zone",
                    color_discrete_map=ZONE_COLOURS,
                    error_y=None,
                    text="mean_incidence",
                    title=f"{sel_disease.replace('_',' ').title()} — Mean Incidence by Zone ({sel_year})",
                    labels={"zone": "Zone", "mean_incidence": "Mean Incidence / 10k pop"},
                )
                fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                fig.update_layout(
                    showlegend=False, height=380,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

            with mcol2:
                st.subheader("Zone summary")
                st.dataframe(
                    map_df[["zone","mean_incidence","alerts","data_quality"]],
                    use_container_width=True, hide_index=True,
                )

    else:  # State level
        map_df = query("""
            SELECT l.state,
                   ROUND(AVG(fs.incidence_rate), 3)   AS mean_incidence,
                   ROUND(MAX(fs.incidence_rate), 3)   AS max_incidence,
                   SUM(fs.active_alert_flag)           AS alerts,
                   COUNT(DISTINCT fs.lga_id)           AS lgas_covered
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
                title=f"{sel_disease.replace('_',' ').title()} — Top 20 States by Incidence ({sel_year})",
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
               ROUND(AVG(incidence_rate), 3) AS mean_incidence,
               COUNT(*) AS n
        FROM feature_store
        WHERE disease_category = ?
          AND season IS NOT NULL
        GROUP BY season
        ORDER BY mean_incidence DESC
    """, (sel_disease,))

    if not seasonal_df.empty:
        scol1, scol2 = st.columns(2)
        with scol1:
            fig_s = px.bar(
                seasonal_df, x="season", y="mean_incidence",
                color="season",
                color_discrete_sequence=["#FF9800","#4CAF50","#2196F3"],
                text="mean_incidence",
                labels={"season": "Season", "mean_incidence": "Mean Incidence / 10k"},
            )
            fig_s.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig_s.update_layout(
                showlegend=False, height=300,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_s, use_container_width=True)
        with scol2:
            st.caption("**Seasonal breakdown**")
            st.dataframe(seasonal_df, use_container_width=True, hide_index=True)

    # Climate correlation
    st.subheader("Climate correlation")
    clim_df = query("""
        SELECT ROUND(rainfall_mm, 0) as rainfall_mm,
               ROUND(incidence_rate, 3) as incidence_rate,
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
            symbol="season", opacity=0.6,
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
    st.caption("LGA-level maternal and child indicators from NDHS/DHIS2")

    years = query("SELECT DISTINCT year FROM maternal_health ORDER BY year"
                  )["year"].tolist()
    zones = query("SELECT DISTINCT zone FROM lga ORDER BY zone")["zone"].tolist()

    col1, col2 = st.columns(2)
    sel_year = col1.selectbox("Year", years, index=len(years)-1 if years else 0)
    sel_zone = col2.multiselect("Zone", zones, default=zones)

    zone_sql = ""
    if sel_zone:
        z_list = ",".join(f"'{z}'" for z in sel_zone)
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
        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Avg ANC Coverage",
                  f"{mh_df['anc_coverage_pct'].mean():.1f}%")
        k2.metric("Avg Skilled Birth",
                  f"{mh_df['skilled_birth_pct'].mean():.1f}%")
        k3.metric("Avg MMR (per 100k)",
                  f"{mh_df['maternal_mortality_ratio'].mean():.0f}")
        k4.metric("Avg Stunting Rate",
                  f"{mh_df['stunting_rate_pct'].mean():.1f}%")

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Maternal mortality by zone")
            zone_mmr = mh_df.groupby("zone")["maternal_mortality_ratio"].mean().reset_index()
            fig = px.bar(
                zone_mmr, x="zone", y="maternal_mortality_ratio",
                color="zone", color_discrete_map=ZONE_COLOURS,
                text="maternal_mortality_ratio",
                labels={"zone": "Zone", "maternal_mortality_ratio": "MMR per 100k"},
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
                    "anc_coverage_pct": "ANC Coverage (%)",
                    "skilled_birth_pct": "Skilled Birth (%)",
                },
                )
            fig2.update_layout(
                height=340,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Nutrition indicators by zone")
        nutri_df = mh_df.groupby("zone")[
            ["stunting_rate_pct", "wasting_rate_pct", "exclusive_bf_pct"]
        ].mean().reset_index()
        nutri_melt = nutri_df.melt(
            id_vars="zone",
            value_vars=["stunting_rate_pct", "wasting_rate_pct", "exclusive_bf_pct"],
            var_name="indicator", value_name="value",
        )
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

        # Top 10 highest risk LGAs
        st.subheader("🚨 Highest maternal mortality LGAs")
        top_risk = mh_df.nlargest(10, "maternal_mortality_ratio")[
            ["lga_name", "state", "zone",
             "maternal_mortality_ratio", "anc_coverage_pct",
             "skilled_birth_pct", "stunting_rate_pct", "data_quality_score"]
        ]
        st.dataframe(top_risk, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — DATA QUALITY
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Quality":
    st.title("📊 Data Quality Audit")
    st.caption(
        "Transparency layer — completeness, bias flags, reporting weights. "
        "Essential for study validity."
    )

    # Reporting weight distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Reporting weight distribution")
        rw_df = query("""
            SELECT zone,
                   ROUND(AVG(reporting_weight), 3) AS mean_weight,
                   ROUND(MIN(reporting_weight), 3) AS min_weight,
                   ROUND(MAX(reporting_weight), 3) AS max_weight,
                   COUNT(*) AS rows
            FROM feature_store
            WHERE reporting_weight IS NOT NULL
            GROUP BY zone ORDER BY mean_weight
        """)
        fig = px.bar(
            rw_df, x="zone", y="mean_weight",
            color="zone", color_discrete_map=ZONE_COLOURS,
            error_y=rw_df["max_weight"] - rw_df["mean_weight"],
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
                   ROUND(AVG(reporting_weight), 3) AS mean_weight,
                   ROUND(AVG(completeness_pct), 1) AS mean_completeness,
                   COUNT(*) AS rows
            FROM feature_store
            WHERE lga_type IS NOT NULL
            GROUP BY lga_type
        """)
        lga_type_melt = lga_type_df.melt(
            id_vars="lga_type",
            value_vars=["mean_weight", "mean_completeness"],
            var_name="Metric", value_name="Score"
        )
        lga_type_melt["Score"] = lga_type_melt["Score"].astype(float)
        fig2 = px.bar(
            lga_type_melt, x="lga_type", y="Score", color="Metric",
            barmode="group",
            labels={"lga_type": "LGA Type"},
            color_discrete_sequence=["#2196F3", "#4CAF50"],
        )
        fig2.update_layout(
            height=340,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # Source breakdown
    st.subheader("Records by data source")
    src_df = query("""
        SELECT source, COUNT(*) as records,
               ROUND(AVG(data_quality_score), 2) as avg_quality
        FROM disease_record
        GROUP BY source ORDER BY records DESC
    """)
    scol1, scol2 = st.columns(2)
    with scol1:
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
    with scol2:
        st.dataframe(src_df, use_container_width=True, hide_index=True)

    st.divider()

    # Lag coverage
    st.subheader("Lag feature coverage (AI model readiness)")
    lag_df = query("""
        SELECT disease_category,
               COUNT(*) AS total,
               SUM(CASE WHEN incidence_lag_1w IS NOT NULL THEN 1 ELSE 0 END) AS lag_1w,
               SUM(CASE WHEN incidence_lag_4w IS NOT NULL THEN 1 ELSE 0 END) AS lag_4w,
               SUM(CASE WHEN incidence_lag_8w IS NOT NULL THEN 1 ELSE 0 END) AS lag_8w
        FROM feature_store
        GROUP BY disease_category
    """)
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

    # Validity threat flags
    st.subheader("🚩 Validity threat checklist")
    threats = [
        ("Reporting bias",        "Urban LGAs over-represented in disease_record",
         "reporting_weight < 0.6 in rural zones",     "⚠️ Mitigated by reporting_weight"),
        ("Selection bias",        "EHR patients skew higher SES",
         "nhia_coverage_pct < 5% in NE/NW",           "⚠️ Flagged in feature_store"),
        ("Label noise",           "Inconsistent ICD-10 coding across facilities",
         "icd10_code NULL in raw disease_record",      "✅ ICD-10 harmoniser applied"),
        ("Temporal confounding",  "Post-COVID health-seeking behaviour shift",
         "Data from 2020 onward needs flag",           "⚠️ Flag year >= 2020 in models"),
        ("Ecological fallacy",    "LGA-level patterns ≠ individual-level",
         "All predictions are population-level",       "✅ Documented in schema"),
        ("External validity",     "SW model may not generalise to NE",
         "Zone stratification in feature_store",       "✅ All 6 zones present"),
        ("Rural data gap",        "Rural LGAs have lower completeness",
         "lga_type=rural completeness < 60%",          "✅ Rural gap imputer applied"),
        ("Synthetic data",        "maternal_health rows are synthetic baseline",
         "data_quality_score = 0.45 for synthetic",   "⚠️ Replace with real DHS data"),
    ]
    threat_df = pd.DataFrame(
        threats,
        columns=["Threat", "Description", "Signal", "Status"],
    )
    st.dataframe(threat_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 6 — INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
elif page == "💡 Insights":
    st.title("💡 AI Insights")
    st.caption("Top risk LGAs, outbreak signals, and model-ready summaries")

    # Top risk LGAs by disease
    st.subheader("🚨 Highest incidence LGAs by disease")
    diseases = query(
        "SELECT DISTINCT disease_category FROM feature_store ORDER BY 1"
    )["disease_category"].tolist()
    sel_disease = st.selectbox("Select disease", diseases)

    top_lga_df = query("""
        SELECT l.lga_name, l.state, l.zone, l.lga_type,
               ROUND(AVG(fs.incidence_rate), 3)       AS mean_incidence,
               ROUND(MAX(fs.incidence_rate), 3)       AS peak_incidence,
               ROUND(AVG(fs.reporting_weight), 2)     AS data_quality,
               SUM(fs.active_alert_flag)              AS total_alerts,
               ROUND(AVG(fs.poverty_headcount_pct),1) AS poverty_pct,
               ROUND(AVG(fs.rainfall_mm), 1)          AS avg_rainfall_mm
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
                title=f"Top 15 LGAs — {sel_disease.replace('_',' ').title()} incidence",
                labels={"lga_name": "LGA", "mean_incidence": "Mean Incidence / 10k"},
            )
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_layout(
                height=400, xaxis_tickangle=45,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.caption("**Top 20 risk LGAs**")
            st.dataframe(
                top_lga_df[["lga_name","state","zone","mean_incidence","data_quality","total_alerts"]],
                use_container_width=True, hide_index=True, height=400,
            )

    st.divider()

    # Poverty-disease correlation
    st.subheader("💰 Poverty × Disease incidence")
    pov_df = query("""
        SELECT ROUND(poverty_headcount_pct, 0) AS poverty_pct,
               ROUND(AVG(incidence_rate), 3)   AS mean_incidence,
               zone, disease_category
        FROM feature_store
        WHERE poverty_headcount_pct IS NOT NULL
          AND incidence_rate IS NOT NULL
          AND disease_category = ?
        GROUP BY ROUND(poverty_headcount_pct, 0), zone
        LIMIT 2000
    """, (sel_disease,))
    if not pov_df.empty:
        fig_p = px.scatter(
            pov_df, x="poverty_pct", y="mean_incidence",
            color="zone", color_discrete_map=ZONE_COLOURS,
            trendline="ols", opacity=0.7,
            labels={
                "poverty_pct": "Poverty Headcount (%)",
                "mean_incidence": "Mean Incidence / 10k",
            },
        )
        fig_p.update_layout(
            height=360,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_p, use_container_width=True)

    st.divider()

    # Model readiness summary
    st.subheader("🧠 Model readiness by disease")
    readiness_df = query("""
        SELECT disease_category,
               COUNT(DISTINCT lga_id)                                         AS lgas_covered,
               COUNT(*)                                                        AS total_rows,
               ROUND(AVG(reporting_weight), 2)                                AS avg_quality,
               ROUND(100.0 * SUM(CASE WHEN incidence_lag_4w IS NOT NULL
                                      THEN 1 ELSE 0 END) / COUNT(*), 1)       AS lag4w_pct,
               ROUND(100.0 * SUM(CASE WHEN rainfall_mm IS NOT NULL
                                      THEN 1 ELSE 0 END) / COUNT(*), 1)       AS climate_pct,
               ROUND(100.0 * SUM(CASE WHEN poverty_headcount_pct IS NOT NULL
                                      THEN 1 ELSE 0 END) / COUNT(*), 1)       AS ses_pct
        FROM feature_store
        GROUP BY disease_category
        ORDER BY avg_quality DESC
    """)
    if not readiness_df.empty:
        readiness_df["overall_score"] = (
            readiness_df["avg_quality"] * 40
            + readiness_df["lag4w_pct"] * 0.3
            + readiness_df["climate_pct"] * 0.15
            + readiness_df["ses_pct"] * 0.15
        ).round(1)

        def score_badge(score):
            if score >= 70:   return "🟢 Ready"
            elif score >= 50: return "🟡 Partial"
            else:             return "🔴 Needs data"

        readiness_df["status"] = readiness_df["overall_score"].apply(score_badge)

        fig_r = px.bar(
            readiness_df, x="disease_category", y="overall_score",
            color="status",
            color_discrete_map={
                "🟢 Ready": "#4CAF50",
                "🟡 Partial": "#FF9800",
                "🔴 Needs data": "#F44336",
            },
            text="overall_score",
            labels={"disease_category": "Disease", "overall_score": "Readiness Score"},
        )
        fig_r.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_r.update_layout(
            height=360, xaxis_tickangle=30,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(range=[0, 105]),
        )
        st.plotly_chart(fig_r, use_container_width=True)
        st.dataframe(readiness_df, use_container_width=True, hide_index=True)

    st.divider()
    st.info(
        "**Next step — Phase 3:** Train your AI models directly on the `feature_store` table. "
        "Use `reporting_weight` as sample weights in scikit-learn or PyTorch. "
        "Evaluate only on African benchmarks (APHRC). Never use WHO global baselines.",
        icon="🧠",
    )

# ════════════════════════════════════════════════════════════════════════════
# PAGE 7 — PREDICTIONS (RLRF Model)
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Predictions":
    st.title("🤖 RLRF Outbreak Predictions")
    st.caption(
        "4-week ahead outbreak forecasting using N-Step SARSA + Random Forest. "
        "Select a disease and LGA to generate a prediction."
    )
    st.divider()

    # ── Model status ─────────────────────────────────────────────────────
    from pathlib import Path as _Path
    import json as _json

    models_dir = _Path("models")
    trained = sorted([f.stem.replace("_rlrf","") for f in models_dir.glob("*_rlrf.pkl")]) if models_dir.exists() else []
    registry_path = models_dir / "model_registry.json"

    if not trained:
        st.warning(
            "No trained models found. Run `python train_model.py` first.",
            icon="⚠️"
        )
        st.code("python train_model.py", language="bash")
        st.stop()

    # ── Model performance cards ───────────────────────────────────────────
    st.subheader("📊 Model Performance Summary")

    # Load registry; patch in any missing diseases so all 8 show up
    registry = {}
    if registry_path.exists():
        with open(registry_path) as f:
            registry = _json.load(f)

    registered_diseases = {d["disease"] for d in registry.get("diseases", []) if "disease" in d}
    patched = False
    for dis in trained:
        if dis not in registered_diseases:
            registry.setdefault("diseases", []).append({
                "disease": dis,
                "r2": 0.0,
                "r2_no_sarsa": 0.0,
                "sarsa_improvement_r2": 0.0,
            })
            patched = True
    if patched:
        st.info(
            "ℹ️ Some models are trained but missing from model_registry.json. "
            "Showing all diseases. Re-run `python train_model.py` to populate full metrics.",
            icon="ℹ️",
        )

    perf_data = [
        d for d in registry.get("diseases", [])
        if "r2" in d
    ]
    if perf_data:
        cols = st.columns(min(len(perf_data), 4))
        for i, m in enumerate(perf_data[:4]):
            with cols[i % 4]:
                delta = f"+{m.get('sarsa_improvement_r2',0):.4f} vs RF-only"
                st.metric(
                    label=m["disease"].replace("_"," ").title(),
                    value=f"R² {m.get('r2',0):.4f}",
                    delta=delta,
                )
        if len(perf_data) > 4:
            cols2 = st.columns(min(len(perf_data)-4, 4))
            for i, m in enumerate(perf_data[4:]):
                with cols2[i % 4]:
                    delta = f"+{m.get('sarsa_improvement_r2',0):.4f} vs RF-only"
                    st.metric(
                        label=m["disease"].replace("_"," ").title(),
                        value=f"R² {m.get('r2',0):.4f}",
                        delta=delta,
                    )

        # Ablation chart — RLRF vs RF
        import plotly.graph_objects as _go
        st.subheader("RLRF vs RF-only (Ablation Study — your Table 3)")
        ab_df = pd.DataFrame([
            {
                "Disease": d["disease"].replace("_"," ").title(),
                "RLRF (R²)": d.get("r2", 0),
                "RF-only (R²)": d.get("r2_no_sarsa", 0),
            }
            for d in perf_data
        ])
        fig_ab = _go.Figure()
        fig_ab.add_bar(
            name="RLRF (N-Step SARSA + RF)",
            x=ab_df["Disease"],
            y=ab_df["RLRF (R²)"],
            marker_color="#1E88E5",
            text=ab_df["RLRF (R²)"].round(4),
            textposition="outside",
        )
        fig_ab.add_bar(
            name="RF-only (no SARSA)",
            x=ab_df["Disease"],
            y=ab_df["RF-only (R²)"],
            marker_color="#E53935",
            text=ab_df["RF-only (R²)"].round(4),
            textposition="outside",
        )
        fig_ab.update_layout(
            barmode="group",
            height=380,
            yaxis=dict(title="R² Score", range=[0, 0.80]),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_ab, use_container_width=True)

    st.divider()

    # ── Live prediction ───────────────────────────────────────────────────
    st.subheader("🔮 Generate 4-Week Outbreak Forecast")

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        selected_disease = st.selectbox(
            "Select disease",
            options=trained,
            format_func=lambda x: x.replace("_"," ").title()
        )
    with col_sel2:
        lgas_df = query("SELECT lga_id, lga_name, state, zone, lga_type FROM lga ORDER BY state, lga_name")
        lga_options = {
            f"{r.lga_name} — {r.state} ({r.zone})": r.lga_id
            for r in lgas_df.itertuples()
        }
        selected_lga_label = st.selectbox("Select LGA", options=list(lga_options.keys()))
        selected_lga_id = lga_options[selected_lga_label]

    # Load latest feature_store row for this LGA+disease
    fs_row = query("""
        SELECT fs.*, l.pop_2022_est, l.area_km2
        FROM feature_store fs
        JOIN lga l ON l.lga_id = fs.lga_id
        WHERE fs.lga_id = ?
          AND fs.disease_category = ?
        ORDER BY fs.epi_year DESC, fs.epi_week DESC
        LIMIT 1
    """, (selected_lga_id, selected_disease))

    if st.button("🚀 Generate Prediction", type="primary"):
        if fs_row.empty:
            st.error("No feature data found for this LGA + disease combination.")
        else:
            try:
                import sys as _sys
                import importlib as _il
                _sys.path.insert(0, ".")
                tm = _il.import_module("train_model")
                _il.reload(tm)

                row_dict = fs_row.iloc[0].to_dict()
                result = tm.predict_outbreak(selected_disease, row_dict)

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success("Prediction generated successfully!")
                    st.divider()

                    # Main result cards
                    c1, c2, c3 = st.columns(3)
                    inc = result["predicted_incidence_per_10k"]
                    prob = result["outbreak_probability"]
                    unc = result["prediction_uncertainty"]

                    c1.metric(
                        "Predicted Incidence",
                        f"{inc:.2f} / 10,000",
                        help="Cases per 10,000 population in 4 weeks"
                    )
                    c2.metric(
                        "Outbreak Probability",
                        f"{prob:.1%}",
                        delta="High risk" if prob > 0.6 else "Moderate" if prob > 0.35 else "Low risk",
                        delta_color="inverse",
                        help="Probability that incidence exceeds 75th percentile"
                    )
                    c3.metric(
                        "Prediction Uncertainty",
                        f"± {unc:.3f}",
                        help="Standard deviation across 300 RF trees"
                    )

                    # Risk level banner
                    if prob > 0.70:
                        st.error(f"🚨 HIGH OUTBREAK RISK — Immediate surveillance response recommended")
                    elif prob > 0.45:
                        st.warning(f"⚠️ ELEVATED RISK — Enhanced monitoring advised")
                    else:
                        st.success(f"✅ LOW RISK — Routine surveillance adequate")

                    # Current state context
                    st.divider()
                    st.subheader("📋 Current State Context")
                    ctx_cols = st.columns(4)
                    ctx_cols[0].metric("Current Incidence", f"{row_dict.get('incidence_rate', 0):.2f}")
                    ctx_cols[1].metric("Lag 1w", f"{row_dict.get('incidence_lag_1w', 0):.2f}")
                    ctx_cols[2].metric("Lag 4w", f"{row_dict.get('incidence_lag_4w', 0):.2f}")
                    ctx_cols[3].metric("Rainfall", f"{row_dict.get('rainfall_mm', 0):.1f} mm")

                    # Historical trend for this LGA
                    hist_df = query("""
                        SELECT epi_week, incidence_rate
                        FROM feature_store
                        WHERE lga_id = ? AND disease_category = ?
                        ORDER BY epi_year, epi_week
                    """, (selected_lga_id, selected_disease))

                    if not hist_df.empty:
                        import plotly.graph_objects as _go2
                        fig_hist = _go2.Figure()
                        fig_hist.add_scatter(
                            x=hist_df["epi_week"],
                            y=hist_df["incidence_rate"],
                            mode="lines",
                            name="Historical incidence",
                            line=dict(color="#1E88E5", width=2),
                        )
                        # Add prediction point
                        last_week = hist_df["epi_week"].max()
                        pred_week = min(last_week + 4, 52)
                        fig_hist.add_scatter(
                            x=[pred_week],
                            y=[inc],
                            mode="markers",
                            name="4-week forecast",
                            marker=dict(
                                color="#E53935", size=14,
                                symbol="star",
                            ),
                        )
                        # Uncertainty band
                        fig_hist.add_scatter(
                            x=[pred_week, pred_week],
                            y=[max(0, inc-unc), inc+unc],
                            mode="lines",
                            name="Uncertainty range",
                            line=dict(color="#E53935", dash="dash", width=1),
                        )
                        fig_hist.update_layout(
                            title=f"{selected_disease.replace('_',' ').title()} — "
                                  f"{selected_lga_label.split('—')[0].strip()} weekly incidence",
                            xaxis_title="Epi Week",
                            yaxis_title="Incidence / 10,000",
                            height=350,
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            legend=dict(orientation="h", y=1.1),
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                    st.caption(
                        f"Model: RLRF (N-Step SARSA + Random Forest + GB) | "
                        f"Forecast horizon: 4 weeks | "
                        f"Training: epi-weeks 1-36 | "
                        f"Model: {result.get('model_type','RLRF')}"
                    )

            except Exception as e:
                st.error(f"Prediction error: {e}")
                import traceback
                st.code(traceback.format_exc())

    st.divider()

    # ── Feature importance chart ──────────────────────────────────────────
    st.subheader("🔑 Feature Importance (SARSA features highlighted)")
    fi_path = _Path("results") / f"{selected_disease}_feature_importance.csv"
    if fi_path.exists():
        fi_df = pd.read_csv(fi_path).head(20)
        sarsa_features = [
            "v_state","advantage","td_error_1w","td_error_4w",
            "td_error_8w","v_x_climate","v_x_deprivation",
            "td_ratio","q_no_alert","q_rumour","q_suspected",
            "q_confirmed","policy_score","advantage_x_zone"
        ]
        fi_df["is_sarsa"] = fi_df["feature"].isin(sarsa_features)
        fi_df["colour"] = fi_df["is_sarsa"].map({True: "#E53935", False: "#1E88E5"})
        fi_df["label"] = fi_df["feature"] + fi_df["is_sarsa"].map(
            {True: " ← SARSA", False: ""}
        )

        import plotly.express as _px
        fig_fi = _px.bar(
            fi_df,
            x="importance", y="label",
            orientation="h",
            color="is_sarsa",
            color_discrete_map={True: "#E53935", False: "#1E88E5"},
            labels={"importance": "Feature Importance", "label": "Feature"},
            title=f"Top 20 features — {selected_disease.replace('_',' ').title()}",
        )
        fig_fi.update_layout(
            height=520,
            showlegend=True,
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                title="Feature type",
                orientation="h", y=1.05,
            ),
        )
        fig_fi.update_traces(
            selector=dict(marker_color="#E53935"),
            name="SARSA feature",
        )
        fig_fi.update_traces(
            selector=dict(marker_color="#1E88E5"),
            name="Standard feature",
        )
        st.plotly_chart(fig_fi, use_container_width=True)
        st.caption("🔴 Red = SARSA-derived features | 🔵 Blue = standard features")
    else:
        st.info(f"Run `python train_model.py --disease {selected_disease}` to generate feature importance data.")

