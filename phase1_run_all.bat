@echo off
REM phase1_run_all.bat — Run all Phase 1 fixes in order
REM Run this from your project root with venv activated:
REM   venv\Scripts\activate
REM   phase1_run_all.bat

echo.
echo ============================================================
echo   Nigeria Health AI — Phase 1 Data Refinery
echo ============================================================
echo.

echo [1/3] Fixing socioeconomic table (770 LGAs x 9 years)...
python fix_socioeconomic.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR in fix_socioeconomic.py — check output above
    pause
    exit /b 1
)
echo.

echo [2/3] Populating maternal_health table...
python ndhs_maternal_collector.py --source auto
if %ERRORLEVEL% NEQ 0 (
    echo ERROR in ndhs_maternal_collector.py — check output above
    pause
    exit /b 1
)
echo.

echo [3/3] Building feature_store...
python build_feature_store.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR in build_feature_store.py — check output above
    pause
    exit /b 1
)
echo.

echo ============================================================
echo   Phase 1 complete. Verifying row counts...
echo ============================================================
python check_db.py

echo.
echo All done. You are ready for Phase 2 (Streamlit dashboard).
pause
