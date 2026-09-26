@echo off
title ForecastBlend Dashboard
cd /d "%~dp0"
echo ============================================
echo  ForecastBlend - Hybrid AI-NWP Dashboard
echo ============================================
echo.
echo Starting Streamlit dashboard...
echo Open your browser at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server.
echo.
streamlit run app.py
pause
