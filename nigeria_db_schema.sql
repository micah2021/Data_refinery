-- nigeria.db — full schema for Nigerian health AI data refinery
-- Generated for disease pattern detection study
-- All tables join on lga_id as the geographic spine

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ============================================================
-- 1. GEOGRAPHY — spine of the entire database
-- ============================================================
CREATE TABLE IF NOT EXISTS lga (
    lga_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    lga_code        TEXT    UNIQUE NOT NULL,       -- official NPC LGA code
    lga_name        TEXT    NOT NULL,
    state           TEXT    NOT NULL,
    zone            TEXT    NOT NULL               -- NW / NE / NC / SW / SE / SS
                    CHECK (zone IN ('NW','NE','NC','SW','SE','SS')),
    lat             REAL,
    lng             REAL,
    lga_type        TEXT    CHECK (lga_type IN ('urban','semi-urban','rural')),
    pop_2022_est    INTEGER,                       -- NPC projection
    pop_density     REAL,                          -- persons per km²
    area_km2        REAL,
    created_at      TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_lga_state ON lga(state);
CREATE INDEX IF NOT EXISTS idx_lga_zone  ON lga(zone);


-- ============================================================
-- 2. FACILITY — health infrastructure
-- ============================================================
CREATE TABLE IF NOT EXISTS facility (
    facility_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    lga_id          INTEGER NOT NULL REFERENCES lga(lga_id),
    facility_code   TEXT    UNIQUE,                -- DHIS2 org unit UID
    facility_name   TEXT    NOT NULL,
    facility_type   TEXT    NOT NULL
                    CHECK (facility_type IN (
                        'federal_teaching_hospital',
                        'state_specialist_hospital',
                        'general_hospital',
                        'primary_health_centre',
                        'private_clinic',
                        'private_hospital',
                        'maternity_home'
                    )),
    ownership       TEXT    CHECK (ownership IN ('federal','state','lga','private','ngo','faith')),
    beds            INTEGER,
    has_laboratory  INTEGER DEFAULT 0 CHECK (has_laboratory IN (0,1)),
    has_maternity   INTEGER DEFAULT 0 CHECK (has_maternity IN (0,1)),
    lat             REAL,
    lng             REAL,
    dhis2_active    INTEGER DEFAULT 1 CHECK (dhis2_active IN (0,1)),
    created_at      TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_facility_lga  ON facility(lga_id);
CREATE INDEX IF NOT EXISTS idx_facility_type ON facility(facility_type);


-- ============================================================
-- 3. DISEASE_RECORD — primary morbidity/mortality table
--    Sources: DHIS2, NHMIS, NCDC IDSR, facility EHR, NHIA claims
-- ============================================================
CREATE TABLE IF NOT EXISTS disease_record (
    record_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    lga_id              INTEGER NOT NULL REFERENCES lga(lga_id),
    facility_id         INTEGER          REFERENCES facility(facility_id),
    report_date         TEXT    NOT NULL,          -- ISO 8601: YYYY-MM-DD
    epi_week            INTEGER,                   -- ISO week number
    epi_year            INTEGER,
    icd10_code          TEXT,                      -- harmonised by refinery
    disease_name        TEXT    NOT NULL,
    disease_category    TEXT    NOT NULL
                        CHECK (disease_category IN (
                            'malaria',
                            'cholera',
                            'typhoid',
                            'tuberculosis',
                            'hiv_aids',
                            'meningitis',
                            'yellow_fever',
                            'lassa_fever',
                            'hypertension',
                            'diabetes',
                            'malnutrition',
                            'diarrhoeal',
                            'respiratory',
                            'other_infectious',
                            'other_ncd'
                        )),
    case_count          INTEGER NOT NULL DEFAULT 0,
    death_count         INTEGER NOT NULL DEFAULT 0,
    age_group           TEXT    CHECK (age_group IN (
                            'under_1','1_4','5_9','10_14',
                            '15_24','25_34','35_44','45_54','55_64','65_plus','unknown'
                        )),
    sex                 TEXT    CHECK (sex IN ('male','female','unknown')),
    is_confirmed        INTEGER DEFAULT 0 CHECK (is_confirmed IN (0,1)),
    data_quality_score  REAL    CHECK (data_quality_score BETWEEN 0 AND 1),
    source              TEXT    NOT NULL
                        CHECK (source IN ('dhis2','nhmis','ncdc_idsr','ehr','nhia_claims','manual')),
    raw_record_ref      TEXT,                      -- original source ID for audit
    ingested_at         TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_dr_lga      ON disease_record(lga_id);
CREATE INDEX IF NOT EXISTS idx_dr_date     ON disease_record(report_date);
CREATE INDEX IF NOT EXISTS idx_dr_category ON disease_record(disease_category);
CREATE INDEX IF NOT EXISTS idx_dr_epiweek  ON disease_record(epi_year, epi_week);
CREATE INDEX IF NOT EXISTS idx_dr_facility ON disease_record(facility_id);


-- ============================================================
-- 4. MATERNAL_HEALTH — NDHS + DHIS2 maternal & child indicators
--    Stored at LGA-quarter level (aggregated survey data)
-- ============================================================
CREATE TABLE IF NOT EXISTS maternal_health (
    record_id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    lga_id                      INTEGER NOT NULL REFERENCES lga(lga_id),
    year                        INTEGER NOT NULL,
    quarter                     INTEGER NOT NULL CHECK (quarter IN (1,2,3,4)),
    anc_coverage_pct            REAL,              -- % women with 4+ ANC visits
    skilled_birth_pct           REAL,              -- % deliveries by skilled attendant
    institutional_delivery_pct  REAL,
    maternal_mortality_ratio    REAL,              -- per 100,000 live births
    neonatal_mortality_rate     REAL,              -- per 1,000 live births
    under5_mortality_rate       REAL,
    stunting_rate_pct           REAL,              -- children under 5
    wasting_rate_pct            REAL,
    exclusive_bf_pct            REAL,              -- exclusive breastfeeding 0–6 months
    vitamin_a_coverage_pct      REAL,
    data_quality_score          REAL    CHECK (data_quality_score BETWEEN 0 AND 1),
    source                      TEXT    CHECK (source IN ('ndhs','dhis2','mics','manual')),
    ingested_at                 TEXT    DEFAULT (datetime('now')),
    UNIQUE (lga_id, year, quarter)
);

CREATE INDEX IF NOT EXISTS idx_mh_lga  ON maternal_health(lga_id);
CREATE INDEX IF NOT EXISTS idx_mh_year ON maternal_health(year);


-- ============================================================
-- 5. SURVEILLANCE_ALERT — NCDC IDSR outbreak alerts
--    Early-warning signal; includes suspected before confirmed
-- ============================================================
CREATE TABLE IF NOT EXISTS surveillance_alert (
    alert_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    lga_id              INTEGER NOT NULL REFERENCES lga(lga_id),
    alert_date          TEXT    NOT NULL,
    epi_week            INTEGER,
    epi_year            INTEGER,
    disease             TEXT    NOT NULL,
    alert_level         TEXT    NOT NULL
                        CHECK (alert_level IN ('rumour','suspected','confirmed','outbreak_declared')),
    suspected_cases     INTEGER DEFAULT 0,
    confirmed_cases     INTEGER DEFAULT 0,
    deaths              INTEGER DEFAULT 0,
    attack_rate         REAL,                      -- per 10,000 population
    response_activated  INTEGER DEFAULT 0 CHECK (response_activated IN (0,1)),
    ncdc_ref            TEXT,
    notes               TEXT,
    ingested_at         TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_sa_lga     ON surveillance_alert(lga_id);
CREATE INDEX IF NOT EXISTS idx_sa_date    ON surveillance_alert(alert_date);
CREATE INDEX IF NOT EXISTS idx_sa_disease ON surveillance_alert(disease);
CREATE INDEX IF NOT EXISTS idx_sa_level   ON surveillance_alert(alert_level);


-- ============================================================
-- 6. CLIMATE_HEALTH — NiMET + FAO climate data per LGA per month
--    Key predictor for malaria, cholera, meningitis seasonality
-- ============================================================
CREATE TABLE IF NOT EXISTS climate_health (
    record_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    lga_id          INTEGER NOT NULL REFERENCES lga(lga_id),
    year            INTEGER NOT NULL,
    month           INTEGER NOT NULL CHECK (month BETWEEN 1 AND 12),
    season          TEXT    CHECK (season IN ('dry','wet','harmattan')),
    rainfall_mm     REAL,
    temp_max_c      REAL,
    temp_min_c      REAL,
    humidity_pct    REAL,
    ndvi            REAL,                          -- vegetation index (malaria vector proxy)
    flood_risk_flag INTEGER DEFAULT 0 CHECK (flood_risk_flag IN (0,1)),
    drought_flag    INTEGER DEFAULT 0 CHECK (drought_flag IN (0,1)),
    source          TEXT    CHECK (source IN ('nimet','fao','era5','manual')),
    ingested_at     TEXT    DEFAULT (datetime('now')),
    UNIQUE (lga_id, year, month)
);

CREATE INDEX IF NOT EXISTS idx_ch_lga   ON climate_health(lga_id);
CREATE INDEX IF NOT EXISTS idx_ch_year  ON climate_health(year, month);
CREATE INDEX IF NOT EXISTS idx_ch_season ON climate_health(season);


-- ============================================================
-- 7. SOCIOECONOMIC — World Bank, UN, FAO indicators per LGA/year
--    Controls for SES confounders in AI models
-- ============================================================
CREATE TABLE IF NOT EXISTS socioeconomic (
    record_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    lga_id                  INTEGER NOT NULL REFERENCES lga(lga_id),
    year                    INTEGER NOT NULL,
    poverty_headcount_pct   REAL,                  -- % below national poverty line
    food_insecurity_pct     REAL,                  -- FAO FIES metric
    gdp_per_capita_usd      REAL,                  -- World Bank (state proxy)
    gini_coefficient        REAL,
    nhia_coverage_pct       REAL,                  -- formal health insurance coverage
    literacy_rate_pct       REAL,
    piped_water_pct         REAL,                  -- access to clean water
    sanitation_pct          REAL,                  -- improved sanitation access
    electricity_access_pct  REAL,
    oil_spill_incidents     INTEGER,               -- NNPC/NOSDRA data
    cpi_score               REAL,                  -- Transparency Int. (state level)
    source                  TEXT,
    ingested_at             TEXT    DEFAULT (datetime('now')),
    UNIQUE (lga_id, year)
);

CREATE INDEX IF NOT EXISTS idx_se_lga  ON socioeconomic(lga_id);
CREATE INDEX IF NOT EXISTS idx_se_year ON socioeconomic(year);


-- ============================================================
-- 8. DATA_QUALITY_LOG — refinery audit trail
--    Records completeness & bias checks; underpins study validity
-- ============================================================
CREATE TABLE IF NOT EXISTS data_quality_log (
    log_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name          TEXT    NOT NULL,
    lga_id              INTEGER          REFERENCES lga(lga_id),
    check_date          TEXT    NOT NULL DEFAULT (date('now')),
    records_checked     INTEGER,
    completeness_pct    REAL    CHECK (completeness_pct BETWEEN 0 AND 100),
    consistency_score   REAL    CHECK (consistency_score BETWEEN 0 AND 1),
    urban_rural_gap     REAL,                      -- completeness gap between urban/rural
    reporting_lag_days  INTEGER,                   -- median days from event to report
    bias_flags          TEXT,                      -- JSON array of flag strings
    action_taken        TEXT,                      -- imputed / dropped / flagged
    notes               TEXT,
    created_at          TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_dql_table ON data_quality_log(table_name);
CREATE INDEX IF NOT EXISTS idx_dql_lga   ON data_quality_log(lga_id);
CREATE INDEX IF NOT EXISTS idx_dql_date  ON data_quality_log(check_date);


-- ============================================================
-- 9. FEATURE_STORE — pre-computed, model-ready training table
--    AI models read ONLY from here; never from raw tables
-- ============================================================
CREATE TABLE IF NOT EXISTS feature_store (
    feature_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    lga_id                  INTEGER NOT NULL REFERENCES lga(lga_id),
    epi_year                INTEGER NOT NULL,
    epi_week                INTEGER NOT NULL,
    disease_category        TEXT    NOT NULL,

    -- target variable
    incidence_rate          REAL,                  -- cases per 10,000 population

    -- lag features (autoregressive signal)
    incidence_lag_1w        REAL,
    incidence_lag_2w        REAL,
    incidence_lag_4w        REAL,
    incidence_lag_8w        REAL,

    -- climate features
    rainfall_mm             REAL,
    temp_max_c              REAL,
    humidity_pct            REAL,
    ndvi                    REAL,
    season                  TEXT,
    flood_risk_flag         INTEGER,

    -- socioeconomic features
    poverty_headcount_pct   REAL,
    food_insecurity_pct     REAL,
    nhia_coverage_pct       REAL,
    piped_water_pct         REAL,
    sanitation_pct          REAL,

    -- geography features
    zone                    TEXT,
    lga_type                TEXT,
    pop_density             REAL,

    -- quality / weighting
    reporting_weight        REAL,                  -- derived from data_quality_score
    completeness_pct        REAL,

    -- alert signal
    active_alert_flag       INTEGER DEFAULT 0,
    alert_level             TEXT,

    computed_at             TEXT    DEFAULT (datetime('now')),
    UNIQUE (lga_id, epi_year, epi_week, disease_category)
);

CREATE INDEX IF NOT EXISTS idx_fs_lga      ON feature_store(lga_id);
CREATE INDEX IF NOT EXISTS idx_fs_epiweek  ON feature_store(epi_year, epi_week);
CREATE INDEX IF NOT EXISTS idx_fs_disease  ON feature_store(disease_category);
CREATE INDEX IF NOT EXISTS idx_fs_zone     ON feature_store(zone);

-- ============================================================
-- VIEWS — convenience queries for the dashboard & model training
-- ============================================================

-- weekly incidence by zone and disease (for spatial clustering model)
CREATE VIEW IF NOT EXISTS v_weekly_zone_incidence AS
SELECT
    l.zone,
    f.epi_year,
    f.epi_week,
    f.disease_category,
    AVG(f.incidence_rate)       AS mean_incidence,
    MAX(f.incidence_rate)       AS max_incidence,
    SUM(f.active_alert_flag)    AS active_alerts,
    AVG(f.reporting_weight)     AS mean_data_quality,
    COUNT(*)                    AS lga_count
FROM feature_store f
JOIN lga l ON l.lga_id = f.lga_id
GROUP BY l.zone, f.epi_year, f.epi_week, f.disease_category;

-- low-quality LGAs (for bias flagging in validity controls)
CREATE VIEW IF NOT EXISTS v_low_quality_lgas AS
SELECT
    l.lga_name,
    l.state,
    l.zone,
    l.lga_type,
    dql.table_name,
    dql.completeness_pct,
    dql.urban_rural_gap,
    dql.bias_flags
FROM data_quality_log dql
JOIN lga l ON l.lga_id = dql.lga_id
WHERE dql.completeness_pct < 60
   OR dql.consistency_score < 0.5;

-- maternal risk summary by LGA (for maternal risk model)
CREATE VIEW IF NOT EXISTS v_maternal_risk_summary AS
SELECT
    l.lga_id,
    l.lga_name,
    l.state,
    l.zone,
    l.lga_type,
    mh.year,
    mh.anc_coverage_pct,
    mh.skilled_birth_pct,
    mh.maternal_mortality_ratio,
    mh.neonatal_mortality_rate,
    mh.stunting_rate_pct,
    se.poverty_headcount_pct,
    se.piped_water_pct,
    se.sanitation_pct,
    l.pop_density
FROM maternal_health mh
JOIN lga l  ON l.lga_id  = mh.lga_id
JOIN socioeconomic se ON se.lga_id = mh.lga_id AND se.year = mh.year;

