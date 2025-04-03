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
    
    # Получаем абсолютный путь к текущему скрипту
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Прямой вызов test_model.py через Python
    try:
        # Запускаем Python с аргументами для передачи в test_model.py
        cmd = [
            sys.executable,
            os.path.join(current_dir, "test_model.py"),
            str(index),
            data_folder
        ]
        
        # Выводим информацию для отладки
        print(f"Запуск команды: {cmd}")
        
        # Запускаем процесс
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Получаем результаты
        stdout, stderr = process.communicate()
        
        # Выводим результаты в интерфейс
        with use_scope("results", clear=True):
            if stderr:
                put_error("Произошла ошибка при выполнении проверки:")
                put_code(stderr)
            
            put_markdown("## Результаты проверки")
            put_code(stdout)
    
    except Exception as e:
        import traceback
        with use_scope("results", clear=True):
            put_error(f"Ошибка при запуске проверки: {str(e)}")
            put_code(traceback.format_exc())

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