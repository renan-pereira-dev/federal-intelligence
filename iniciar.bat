@echo off
setlocal
chcp 65001 >nul
title Federal Intelligence - Beta 0.1
cd /d "%~dp0"

echo.
echo  ==============================================
echo   FEDERAL INTELLIGENCE - BETA 0.1
echo   Laboratorio experimental - uso educacional
echo  ==============================================
echo.

rem 1. Verificar Python ----------------------------------------------
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (
    where python >nul 2>&1 && python --version >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo [ERRO] Python nao encontrado.
    echo Instale o Python 3.12 ou mais novo em https://www.python.org/downloads/
    echo e marque a opcao "Add python.exe to PATH" durante a instalacao.
    pause
    exit /b 1
)
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)"
if errorlevel 1 (
    echo [ERRO] E necessario Python 3.12 ou mais novo. Versao encontrada:
    %PY% --version
    pause
    exit /b 1
)
echo [OK] Python encontrado:
%PY% --version

rem 2. Criar ambiente virtual se necessario ----------------------------
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Criando ambiente virtual na pasta .venv ...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [ERRO] Nao foi possivel criar o ambiente virtual.
        pause
        exit /b 1
    )
)
set "VPY=.venv\Scripts\python.exe"

rem 3. Instalar dependencias ------------------------------------------
echo.
echo Verificando dependencias (na primeira vez pode levar alguns minutos)...
"%VPY%" -m pip install --disable-pip-version-check -q -r requirements.txt
if errorlevel 1 (
    echo [ERRO] Falha ao instalar as dependencias.
    echo Verifique a conexao com a internet e tente novamente.
    pause
    exit /b 1
)
echo [OK] Dependencias prontas.

rem 4 e 5. Abrir o navegador e iniciar o Streamlit ----------------------
rem Opcional: FI_PORTA muda a porta (padrao 8501); FI_SEM_NAVEGADOR=1 nao abre o navegador.
if not defined FI_PORTA set "FI_PORTA=8501"
echo.
echo Abrindo http://localhost:%FI_PORTA% no navegador...
echo Para encerrar, feche esta janela ou pressione Ctrl+C.
echo.
if not "%FI_SEM_NAVEGADOR%"=="1" start "" /b powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Sleep -Seconds 4; Start-Process 'http://localhost:%FI_PORTA%'"
"%VPY%" -m streamlit run app.py --server.port %FI_PORTA% --server.headless true

echo.
echo O aplicativo foi encerrado.
pause
endlocal
