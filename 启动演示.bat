@echo off
chcp 65001 >nul
title RAG Demo Launcher

REM ============================================
REM  一键启动 RAG 演示（Docker + Ollama + API + 浏览器）
REM  双击即可。可选参数：--backend llm  切换成 LLM 精排
REM ============================================

set PY=D:\Code_Tools\Miniforge3\envs\mini_pg_agent\python.exe
set PROJ=D:\Code\Work_Code\projects\mini_pg_agent

"%PY%" "%PROJ%\scripts\start_demo.py" %*

echo.
echo 按任意键关闭本窗口（服务会继续在后台运行）...
pause >nul
