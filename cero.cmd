@echo off
rem Lanzador para Windows. Desde esta carpeta:  .\cero <orden>     (por ejemplo:  .\cero todo --receta nano)
setlocal
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo   Falta instalar. Ejecuta primero:  instalar.cmd
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -X utf8 -m cero %*
