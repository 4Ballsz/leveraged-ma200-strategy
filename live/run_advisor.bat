@echo off
REM Daily run of the MA200 leveraged advisor.
REM Double-click this, or point Windows Task Scheduler at it.

setlocal
set SCRIPT_DIR=%~dp0

REM Prefer the project venv if it exists, otherwise whatever python is on PATH.
set PY=%SCRIPT_DIR%.venv\Scripts\python.exe
if not exist "%PY%" set PY=C:\documents\coding\trading\testing\.venv\Scripts\python.exe
if not exist "%PY%" set PY=python

"%PY%" "%SCRIPT_DIR%daily_advisor.py" %*
set RC=%ERRORLEVEL%

REM Keep the window open when double-clicked; harmless under Task Scheduler.
if "%~1"=="" pause
exit /b %RC%
