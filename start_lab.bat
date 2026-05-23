@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo 正在启动深度学习交互式学习网站...
echo.
echo 使用浏览器页面时，请保持这个窗口打开。
echo 页面地址：
echo   http://127.0.0.1:8501
echo.

python -m streamlit run main.py --server.address 127.0.0.1 --server.port 8501

echo.
echo 网站已停止。按任意键关闭窗口。
pause >nul
