@echo off
rem Instalador para Windows: busca un Python adecuado y ejecuta instalar.py.
rem Doble clic, o desde una terminal:  instalar.cmd   (opciones: --cpu  --gpu  --sin-curso)
setlocal
cd /d "%~dp0"
chcp 65001 >nul

rem 1. El lanzador «py» de python.org: se prefiere 3.12 (la versión más probada), luego otras.
where py >nul 2>nul
if %errorlevel%==0 (
    for %%v in (3.12 3.13 3.11 3.14) do (
        py -%%v -c "import sys" >nul 2>nul && (
            py -%%v instalar.py %*
            goto fin
        )
    )
)

rem 2. Un «python» en el PATH (instalar.py comprueba que sea 3.11 o posterior).
where python >nul 2>nul
if %errorlevel%==0 (
    python instalar.py %*
    goto fin
)

echo.
echo   No se encuentra Python 3.11 o posterior en este equipo.
echo   Instala Python 3.12 desde https://www.python.org/downloads/
echo   (marca la casilla "Add python.exe to PATH") y vuelve a ejecutar instalar.cmd
set errorlevel=1

:fin
set codigo=%errorlevel%
echo.
pause
exit /b %codigo%
