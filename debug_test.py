
import sys
import os
import json

# Запись в лог-файл для отладки
def log(message):
    with open('debug_log.txt', 'a', encoding='utf-8') as f:
        f.write(message + '\n')

log("Тестовый скрипт запущен")

# Выводим информацию о среде
log(f"Python: {sys.version}")
log(f"Текущая директория: {os.getcwd()}")
log(f"sys.path: {sys.path}")

# Проверка наличия файлов
test_model_exists = os.path.exists('test_model.py')
test_comparison_exists = os.path.exists('test_comparison.py')
log(f"test_model.py существует: {test_model_exists}")
log(f"test_comparison.py существует: {test_comparison_exists}")

# Пытаемся импортировать модули
try:
    log("Пытаемся импортировать test_model...")
    import test_model
    log("Импорт test_model успешен")
    
    # Смотрим, какие функции есть в модуле
    log(f"Функции в test_model: {dir(test_model)}")
    
    # Проверяем наличие функции check
    if hasattr(test_model, 'check'):
        log("Функция check найдена")
    else:
        log("Функция check НЕ найдена!")
    
except Exception as e:
    log(f"Ошибка при импорте test_model: {e}")

# Возвращаем успешный результат для отладки
print(json.dumps({"result": "Отладочный запуск успешно выполнен. Проверьте файл debug_log.txt для деталей."}, ensure_ascii=False))
