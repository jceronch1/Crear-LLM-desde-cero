@echo off
rem Menú de Windows: doble clic y elige una opción. No hace falta saber usar la consola.
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title Crear un LLM desde cero
set "PY=%~dp0.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo.
    echo   Todavía no está instalado. Se abre el instalador...
    echo.
    call "%~dp0instalar.cmd"
    if not exist "%PY%" exit /b 1
)

:menu
cls
echo.
echo   ==========================================================
echo     CREAR UN LLM DESDE CERO
echo   ==========================================================
echo.
echo     1. Comprobar el equipo (y cuánto tardará cada receta)
echo     2. Entrenar mi primer LLM con su chatbot (receta nano)
echo     3. Conversar con el chatbot
echo     4. Abrir el laboratorio en el navegador
echo     5. Abrir el curso (13 clases en Jupyter)
echo     6. Ver el estado de mis proyectos
echo     7. Evaluar qué aprendió el modelo
echo.
echo     0. Salir
echo.
set "opcion="
set /p "opcion=  Elige una opción y pulsa Intro: "

if "%opcion%"=="1" ( "%PY%" -X utf8 -m cero comprobar & goto pausa )
if "%opcion%"=="2" goto entrenar
if "%opcion%"=="3" ( "%PY%" -X utf8 -m cero chat & goto pausa )
if "%opcion%"=="4" (
    echo.
    echo   El laboratorio se abre en el navegador. Cierra esta ventana o pulsa Ctrl+C para apagarlo.
    "%PY%" -X utf8 -m cero lab
    goto pausa
)
if "%opcion%"=="5" ( "%PY%" -X utf8 -m cero curso & goto pausa )
if "%opcion%"=="6" ( "%PY%" -X utf8 -m cero proyectos & "%PY%" -X utf8 -m cero estado & goto pausa )
if "%opcion%"=="7" ( "%PY%" -X utf8 -m cero evaluar & goto pausa )
if "%opcion%"=="0" exit /b 0
goto menu

:entrenar
echo.
set "nombre=mi-primer-llm"
set /p "nombre=  Nombre del proyecto, sin espacios (Intro = mi-primer-llm): "
set "receta=nano"
set /p "receta=  Receta: nano, micro, mini o base (Intro = nano, la de unos minutos): "
"%PY%" -X utf8 -m cero todo "%nombre%" --receta "%receta%"

:pausa
echo.
pause
goto menu
