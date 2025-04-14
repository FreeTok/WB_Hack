@echo off
chcp 1251 > nul

cd /d "%~dp0"

if not exist app.py (
    echo Ошибка: Файл app.py не найден в текущей директории.
    echo Убедитесь, что батник находится в той же папке, что и app.py
    pause
    exit /b 1
)

echo Запуск приложения...
python app.py

if %errorlevel% neq 0 (
    echo Приложение завершилось с ошибкой (код %errorlevel%).
    pause
)