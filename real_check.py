
import sys
import os
import json

# Запись в лог-файл для отладки
def log(message):
    with open('debug_log.txt', 'a', encoding='utf-8') as f:
        f.write(str(message) + '\n')

log("\n--- Начало выполнения основного скрипта ---")

# Добавляем текущую директорию в путь поиска
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
    log(f"Добавлен путь: {current_dir}")

# Импортируем функцию check из test_model.py
try:
    log("Пытаемся импортировать test_model...")
    from test_model import check
    log("Импорт test_model успешен")
    
    # Запускаем проверку с указанным индексом и путем к данным
    log("Запуск проверки...")
    data_folder = r"N:/Hackatons/WB/data"
    log(f"Индекс: 2, Папка: {data_folder}")
    
    result = check(2, data_folder=data_folder)
    log("Проверка завершена")
    log(f"Результат: {result}")
    
    # Выводим результат в JSON формате
    json_result = json.dumps({"result": result}, ensure_ascii=False)
    print(json_result)
except Exception as e:
    import traceback
    error = traceback.format_exc()
    log(f"Ошибка: {e}")
    log(f"Трассировка: {error}")
    json_result = json.dumps({"error": str(e), "traceback": error}, ensure_ascii=False)
    print(json_result)
