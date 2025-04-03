from pywebio import start_server
from pywebio.output import *
from pywebio.input import input as pyinput
from pywebio.session import set_env, run_js
import tkinter as tk
from tkinter import filedialog
import threading
import webbrowser
import time
import queue
import json
import os
import subprocess
import sys

# Файл для сохранения путей к папкам
PATHS_FILE = "saved_paths.json"

# Загрузка сохраненных путей или создание пустого словаря
def load_paths():
    if os.path.exists(PATHS_FILE):
        try:
            with open(PATHS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"Ошибка чтения файла {PATHS_FILE}, создаем новый словарь")
    
    return {"folder1": "", "folder2": ""}

# Сохранение путей в файл
def save_paths(paths):
    try:
        with open(PATHS_FILE, 'w', encoding='utf-8') as f:
            json.dump(paths, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"Ошибка при сохранении путей: {e}")

# Загружаем сохраненные пути
folder_paths = load_paths()

# Очередь для передачи результатов между потоками
results_queue = queue.Queue()

def select_folder(folder_id):
    """Функция для выбора папки через диалог Windows"""
    def open_dialog():
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        path = filedialog.askdirectory()
        root.destroy()
        results_queue.put((folder_id, path))
    
    dialog_thread = threading.Thread(target=open_dialog)
    dialog_thread.daemon = True
    dialog_thread.start()

    toast("Пожалуйста, выберите папку в открывшемся диалоговом окне")

def check_results():
    """Проверяет очередь результатов и обновляет UI"""
    while True:
        try:
            folder_id, path = results_queue.get_nowait()
            if path:  # Если путь был выбран
                folder_paths[folder_id] = path
                # Сохраняем изменения в файл
                save_paths(folder_paths)
                # Обновляем UI
                run_js(f'document.getElementById("path_{folder_id}").textContent = {repr(path)}')
            results_queue.task_done()
        except queue.Empty:
            break

def run_check():
    """Запускает проверку с указанным индексом"""
    # Сначала получаем индекс от пользователя
    index = pyinput("Введите индекс записи для проверки:", type='number', value=0)
    
    # Проверяем, выбрана ли папка с данными
    if not folder_paths["folder1"]:
        put_error("Сначала выберите папку с данными!")
        return
    
    # Создаем область для результатов и показываем индикатор загрузки
    with use_scope("results", clear=True):
        put_loading(shape='grow')
        put_text("Запуск проверки, пожалуйста подождите...")

    # Получаем путь к папке с данными
    data_folder = folder_paths["folder1"]
    
    # Создаем именованный временный файл для логов
    log_file = "debug_log.txt"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Начало проверки с индексом {index} и папкой {data_folder}\n")
    
    # Создаем простой скрипт для теста базовой функциональности
    test_script = "debug_test.py"
    with open(test_script, 'w', encoding='utf-8') as f:
        f.write("""
import sys
import os
import json

# Запись в лог-файл для отладки
def log(message):
    with open('debug_log.txt', 'a', encoding='utf-8') as f:
        f.write(message + '\\n')

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
""")
    
    # Запускаем отладочный скрипт
    try:
        # Выводим информацию для отладки
        print(f"Запуск отладочного скрипта: {test_script}")
        
        # Запускаем процесс
        process = subprocess.Popen(
            [sys.executable, test_script],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True,
            encoding='utf-8'
        )
        
        # Получаем результаты
        stdout, stderr = process.communicate()
        
        # Добавляем результаты запуска в лог
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n--- Результат выполнения отладочного скрипта ---\n")
            f.write(f"stdout: {stdout}\n")
            f.write(f"stderr: {stderr}\n")
        
        # Теперь запускаем реальную проверку
        real_script = "real_check.py"
        with open(real_script, 'w', encoding='utf-8') as f:
            f.write(f"""
import sys
import os
import json

# Запись в лог-файл для отладки
def log(message):
    with open('debug_log.txt', 'a', encoding='utf-8') as f:
        f.write(str(message) + '\\n')

log("\\n--- Начало выполнения основного скрипта ---")

# Добавляем текущую директорию в путь поиска
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
    log(f"Добавлен путь: {{current_dir}}")

# Импортируем функцию check из test_model.py
try:
    log("Пытаемся импортировать test_model...")
    from test_model import check
    log("Импорт test_model успешен")
    
    # Запускаем проверку с указанным индексом и путем к данным
    log("Запуск проверки...")
    data_folder = r"{data_folder}"
    log(f"Индекс: {index}, Папка: {{data_folder}}")
    
    result = check({index}, data_folder=data_folder)
    log("Проверка завершена")
    log(f"Результат: {{result}}")
    
    # Выводим результат в JSON формате
    json_result = json.dumps({{"result": result}}, ensure_ascii=False)
    print(json_result)
except Exception as e:
    import traceback
    error = traceback.format_exc()
    log(f"Ошибка: {{e}}")
    log(f"Трассировка: {{error}}")
    json_result = json.dumps({{"error": str(e), "traceback": error}}, ensure_ascii=False)
    print(json_result)
""")
        
        # Запускаем основной скрипт проверки
        log_process = subprocess.Popen(
            [sys.executable, real_script],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True,
            encoding='utf-8'
        )
        
        # Получаем результаты с таймаутом
        try:
            real_stdout, real_stderr = log_process.communicate(timeout=120)  # Ждем результат максимум 120 секунд
            
            # Добавляем результаты запуска в лог
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n--- Результат выполнения основного скрипта ---\n")
                f.write(f"stdout: {real_stdout}\n")
                f.write(f"stderr: {real_stderr}\n")
            
            # Обрабатываем результаты
            with use_scope("results", clear=True):
                if real_stderr:
                    put_error("Произошла ошибка при выполнении проверки:")
                    put_code(real_stderr)
                
                # Проверяем, есть ли вывод
                if not real_stdout:
                    put_error("Скрипт не вернул никаких данных!")
                    put_text(f"Проверьте файл отладки {log_file} для деталей")
                    put_file('debug_log.txt', open(log_file, 'rb').read(), 'Скачать лог для отладки')
                    return
                
                # Пробуем прочитать JSON
                try:
                    result_data = json.loads(real_stdout)
                    
                    if "error" in result_data:
                        put_error("Ошибка при выполнении проверки:")
                        put_code(result_data["error"])
                        if "traceback" in result_data:
                            put_code(result_data["traceback"])
                    else:
                        put_markdown("## Результаты проверки")
                        put_code(result_data["result"])
                except json.JSONDecodeError:
                    # Если не удалось разобрать JSON, выводим как обычный текст
                    put_markdown("## Результаты проверки (не в формате JSON)")
                    put_code(real_stdout)
                
                # Добавляем ссылку на лог для отладки
                put_text("Для отладки доступен подробный лог:")
                put_file('debug_log.txt', open(log_file, 'rb').read(), 'Скачать лог')
        
        except subprocess.TimeoutExpired:
            log_process.kill()
            with use_scope("results", clear=True):
                put_error("Время выполнения проверки истекло (превышено 120 секунд)")
                put_text("Проверьте файл debug_log.txt для деталей")
                put_file('debug_log.txt', open(log_file, 'rb').read(), 'Скачать лог для отладки')
    
    except Exception as e:
        import traceback
        with use_scope("results", clear=True):
            put_error(f"Ошибка при запуске проверки: {str(e)}")
            put_code(traceback.format_exc())
            if os.path.exists(log_file):
                put_file('debug_log.txt', open(log_file, 'rb').read(), 'Скачать лог для отладки')

def main():
    """Основная функция приложения"""
    set_env(title="Проверка товаров в видео")
    
    put_markdown("# Проверка товаров в видео")
    
    # Информационное сообщение о сохранении путей
    put_info("Выберите папку с данными, содержащую файл videos.json и подпапки с изображениями.")
    
    # Кнопка и место для отображения пути для первой папки
    put_row([
        put_button("Выбрать папку с данными", onclick=lambda: select_folder("folder1"), color='primary'),
    ])
    
    # Используем HTML с установкой начального значения из сохраненных путей
    folder1_path = folder_paths["folder1"] or "Путь к папке с данными будет отображен здесь"
    html = f"""
    <div style="margin-top: 5px; padding: 10px; border: 1px solid #ddd; background-color: #f9f9f9;">
        <code id="path_folder1">{folder1_path}</code>
    </div>
    """
    put_html(html)
    
    put_markdown("---")
    
    # Поле для ввода индекса и кнопка запуска проверки
    put_markdown("## Запуск проверки")
    
    # Добавляем кнопку запуска, которая запросит индекс
    put_button("Запустить проверку", onclick=run_check, color='success')
    
    # Область для вывода результатов
    put_markdown("## Результаты")
    put_scope("results")
    
    # Кнопка очистки путей
    put_markdown("---")
    put_row([
        put_button("Очистить пути", onclick=clear_paths, color='warning'),
        put_button("Закрыть приложение", onclick=lambda: exit(), color='danger'),
    ])
    
    # Запускаем периодическую проверку результатов
    while True:
        check_results()
        time.sleep(0.5)

def clear_paths():
    """Очищает сохраненные пути к папкам"""
    global folder_paths
    folder_paths = {"folder1": "", "folder2": ""}
    save_paths(folder_paths)
    
    # Обновляем интерфейс
    run_js('''
        document.getElementById("path_folder1").textContent = "Путь к папке с данными будет отображен здесь";
    ''')
    
    toast("Пути были очищены")

def open_browser():
    """Открывает браузер по умолчанию"""
    time.sleep(1)
    webbrowser.open("http://localhost:8080/")

if __name__ == '__main__':
    # Открываем браузер в отдельном потоке
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Запускаем приложение
    start_server(main, port=8080, debug=True)