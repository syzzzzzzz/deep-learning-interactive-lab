@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "PORTS=8000 8001 8002 8003 4173 5173 5500"

for %%P in (%PORTS%) do (
  python -c "import socket,sys;s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',int(sys.argv[1])));s.close()" %%P >nul 2>nul
  if errorlevel 1 (
    echo 端口 %%P 不可用，继续尝试下一个端口。
  ) else (
    set "PORT=%%P"
    goto :serve
  )
)

echo 端口 %PORTS% 都无法启动，请先关闭占用这些端口的程序后再重试。
pause >nul
exit /b 1

:serve

echo 正在启动深度学习交互式学习网站...
echo.
echo 使用浏览器页面时，请保持这个窗口打开。
echo 如果某个端口被占用或被系统权限拒绝，脚本会自动尝试下一个端口。
echo 页面地址：
echo   http://127.0.0.1:%PORT%
echo.

python -m http.server %PORT% --bind 127.0.0.1

echo.
echo 网站已停止。按任意键关闭窗口。
pause >nul
