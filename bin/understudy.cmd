@echo off
rem Understudy launcher for Windows. Prefers the project venv so the tool runs
rem straight from a clone, without the caller having to activate anything.
setlocal
set "here=%~dp0.."
pushd "%here%" || exit /b 1
if exist "%here%\.venv\Scripts\python.exe" (
    "%here%\.venv\Scripts\python.exe" -m understudy %*
) else (
    py -3 -m understudy %* 2>nul || python -m understudy %*
)
set "rc=%errorlevel%"
popd
exit /b %rc%
