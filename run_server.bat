@echo off
echo ========================================
echo   RGB 이미지 기반 과실 분석 플랫폼
echo ========================================
echo.
echo 서버 시작 중...
echo 브라우저에서 http://localhost:8000 접속하세요
echo 종료하려면 Ctrl+C
echo.

cd /d "%~dp0web"
python -m uvicorn app:app --host 0.0.0.0 --port 8000

pause
