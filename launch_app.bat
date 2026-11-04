@echo off
title PlasticityX Desktop Engine
cd /d "%~dp0"

:: Start the Streamlit backend minimized in the background
start /min "" "C:\Users\Bavin\AppData\Local\Programs\Python\Python312\python.exe" -m streamlit run app.py --server.headless=true --server.port=8501

:: Wait 2 seconds for the server to bind port 8501
timeout /t 2 /nobreak >nul

:: Launch Edge/Chrome in dedicated standalone app-window mode
start msedge --app="http://localhost:8501"
