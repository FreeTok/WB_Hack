from pywebio import start_server
from pywebio.output import *
from pywebio.input import input as pyinput
from pywebio.session import set_env, run_js, register_thread, info
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
import torch
import concurrent.futures
from functools import partial
import multiprocessing
import mimetypes
from pywebio.platform.tornado import start_server as tornado_start_server
from pywebio.platform.tornado import webio_handler
from tornado.web import StaticFileHandler, Application
import tornado.web
import urllib.parse

# Импортируем модуль для предварительной загрузки моделей
from model_preloader import start_preloading, get_loading_status

class FileHandler(tornado.web.RequestHandler):
    """Обработчик для доступа к файлам через URL"""
    
    def get(self):
        file_path = self.get_argument('file', None)
        if not file_path:
            self.set_status(400)
            self.write("Не указан путь к файлу")
            return
        
        # Декодируем путь
        file_path = urllib.parse.unquote(file_path)
        
        # Проверяем, существует ли файл
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            self.set_status(404)
            self.write(f"Файл не найден: {file_path}")
            return
        
        # Определяем MIME-тип
        content_type, _ = mimetypes.guess_type(file_path)
        if content_type:
            self.set_header('Content-Type', content_type)
        
        # Отправляем файл
        with open(file_path, 'rb') as f:
            self.write(f.read())

def create_video_html(video_path):
    """
    Создает HTML-код для отображения только видео с поддержкой перематывания.
    
    Args:
        video_path (str): Путь к видео-файлу
        
    Returns:
        str: HTML-код с видеоплеером
    """
    # Генерируем уникальный ID для плеера
    import uuid
    player_id = f"videoplayer_{uuid.uuid4().hex[:8]}"
    container_id = f"media_container_{uuid.uuid4().hex[:8]}"
    
    html = f"""
    <div id="{container_id}" class="media-container">
        <style>
            .media-container {{
                margin: 15px 0;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: #f9f9f9;
            }}
            .video-container {{
                margin-bottom: 15px;
            }}
            video {{
                max-width: 100%;
                max-height: 400px;
            }}
            .section-title {{
                margin: 10px 0 5px 0;
                font-weight: bold;
            }}
            /* Добавим явные стили для элементов управления */
            video::-webkit-media-controls-panel {{
                display: flex !important;
                opacity: 1 !important;
            }}
            video::-webkit-media-controls-play-button {{
                display: block !important;
            }}
            video::-webkit-media-controls-timeline {{
                display: block !important;
            }}
            video::-webkit-media-controls-current-time-display {{
                display: block !important;
            }}
            video::-webkit-media-controls-time-remaining-display {{
                display: block !important;
            }}
            video::-webkit-media-controls-mute-button {{
                display: block !important;
            }}
            video::-webkit-media-controls-volume-slider {{
                display: block !important;
            }}
        </style>
    """
    
    # Добавляем видеоплеер
    html += """
        <div class="section-title">Проверенное видео:</div>
        <div class="video-container">
    """
    
    # Формируем URL для видео, обеспечивающий доступ через веб
    video_url = f"/file?file={video_path}"
    
    # Добавляем видеоплеер с дополнительными атрибутами
    html += f"""
            <video id="{player_id}" controls preload="metadata" controlsList="nodownload" disablePictureInPicture disableRemotePlayback playsinline>
                <source src="{video_url}" type="video/mp4">
                Ваш браузер не поддерживает тег video.
            </video>
        </div>
    </div>
    
    <script>
    (function() {{
        // Функция для инициализации видеоплеера
        function initializePlayer() {{
            const player = document.getElementById('{player_id}');
            if (!player) return;
            
            // Обеспечиваем, что элементы управления всегда доступны
            player.controls = true;
            
            // Добавляем обработчики событий для перехвата кликов по элементам управления
            player.addEventListener('click', function(e) {{
                e.stopPropagation();
            }});
            
            // Предотвращаем всплытие событий для элементов управления
            const controls = player.querySelectorAll('*');
            controls.forEach(control => {{
                control.addEventListener('click', function(e) {{
                    e.stopPropagation();
                }});
            }});
        }}
        
        // Запускаем инициализацию сразу при загрузке DOM
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initializePlayer);
        }} else {{
            initializePlayer();
        }}
        
        // Также добавляем проверку через setTimeout на случай, если DOM обновляется асинхронно
        setTimeout(initializePlayer, 500);
        setTimeout(initializePlayer, 1000);
        setTimeout(initializePlayer, 2000);
    }})();
    </script>
    """
    
    return html

def create_images_html(image_paths, item_id=None):
    """
    Создает HTML-код для отображения только изображений товара.
    
    Args:
        image_paths (list): Список путей к изображениям
        item_id (str, optional): ID товара для заголовка
        
    Returns:
        str: HTML-код с изображениями
    """
    # Если нет изображений, возвращаем пустую строку
    if not image_paths or len(image_paths) == 0:
        return "<div>Изображения товара отсутствуют.</div>"
    
    # Создаем уникальный ID для контейнера, чтобы избежать конфликтов
    container_id = f"images_container_{item_id if item_id else 'main'}"
    
    html = f"""
    <div id="{container_id}" class="images-container-wrapper">
        <style>
            .images-container-wrapper {{
                margin: 10px 0;
                padding: 10px;
                border: 1px solid #eee;
                border-radius: 5px;
                background-color: #fafafa;
            }}
            .images-container {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-top: 10px;
            }}
            .image-item {{
                border: 1px solid #ccc;
                padding: 5px;
                text-align: center;
                background: white;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .image-item img {{
                max-width: 150px;
                max-height: 150px;
                object-fit: contain;
            }}
        </style>
    """
    
    html += """<div class="images-container">"""
    
    for i, img_path in enumerate(image_paths):
        # Формируем URL для изображения
        img_url = f"/file?file={img_path}"
        img_name = os.path.basename(img_path)
        
        html += f"""
        <div class="image-item">
            <img src="{img_url}" alt="Изображение товара {i+1}">
            <div>{img_name}</div>
        </div>
        """
    
    html += """</div></div>"""
    
    return html

def add_collapse_fix_script():
    """
    Добавляет JavaScript для исправления работы видео в спойлерах
    """
    script = """
    <script>
    (function() {
        // Функция для модификации поведения спойлеров с видео
        function fixVideoCollapse() {
            // Находим все .collapse-btn (кнопки спойлеров в PyWebIO)
            const collapseBtns = document.querySelectorAll('.collapse-btn');
            
            collapseBtns.forEach(btn => {
                // Проверяем, есть ли видео в связанном контенте
                const collapseId = btn.getAttribute('data-target');
                if (!collapseId) return;
                
                const collapseContent = document.querySelector(collapseId);
                if (!collapseContent) return;
                
                const hasVideo = collapseContent.querySelector('video');
                if (!hasVideo) return;
                
                // Модифицируем обработчик клика, чтобы предотвратить всплытие событий от видео
                btn.addEventListener('click', function() {
                    // Находим все видео внутри этого спойлера
                    const videos = collapseContent.querySelectorAll('video');
                    videos.forEach(video => {
                        // Устанавливаем обработчики событий для предотвращения всплытия
                        video.addEventListener('click', function(e) {
                            e.stopPropagation();
                        });
                        
                        // Обеспечиваем, что элементы управления видео работают
                        video.controls = true;
                    });
                });
            });
        }
        
        // Запускаем исправление спойлеров через небольшую задержку, чтобы DOM успел загрузиться
        setTimeout(fixVideoCollapse, 500);
        setTimeout(fixVideoCollapse, 2000);
        // Также запускаем при каждом изменении DOM
        const observer = new MutationObserver(fixVideoCollapse);
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    })();
    </script>
    """
    put_html(script)

# Файл для сохранения путей к папкам
PATHS_FILE = "saved_paths.json"

# Загрузка сохраненных путей или создание пустого словаря
def load_paths():
    if os.path.exists(PATHS_FILE):
        try:
            with open(PATHS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Ошибка чтения файла {PATHS_FILE} ({e}), создаем новый словарь")
    
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

def estimate_optimal_workers():
    """
    Оценивает оптимальное количество параллельных работников в зависимости от доступной памяти GPU.
    """
    if not torch.cuda.is_available():
        # Если GPU недоступен, используем количество ядер CPU
        return max(1, multiprocessing.cpu_count() - 1)
    
    try:
        # Получаем информацию о доступной памяти GPU
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        free_memory = total_memory - torch.cuda.memory_allocated(device) - torch.cuda.memory_reserved(device)
        
        # Примерный объем памяти, необходимый для одной проверки (оценка)
        # Можно настроить этот параметр на основе экспериментов
        memory_per_task = 0.1 * 1024 * 1024 * 1024  # 2 ГБ на задачу (примерно)
        
        # Рассчитываем максимальное количество параллельных задач
        max_workers = max(1, int(free_memory / memory_per_task))
        
        # Ограничиваем максимальное количество работников 
        # для предотвращения перегрузки системы
        return min(max_workers, 6)  # Не более 4 параллельных процессов
    except Exception as e:
        print(f"Ошибка при оценке оптимального количества воркеров: {e}")
        return 1  # В случае ошибки используем один воркер

def batch_check(start_index, end_index, data_folder):
    """
    Выполняет пакетную проверку для указанного диапазона индексов.
    
    Args:
        start_index (int): Начальный индекс
        end_index (int): Конечный индекс
        data_folder (str): Путь к папке с данными
        
    Returns:
        dict: Результаты проверки для всех индексов
    """
    # Время начала
    start_time = time.time()
    
    # Создаем словарь для результатов
    all_results = {
        "start_index": start_index,
        "end_index": end_index,
        "total_processed": 0,
        "results": {},
        "errors": [],
        "total_time": 0,
        "avg_time_per_item": 0
    }
    
    # Определяем количество воркеров
    workers = estimate_optimal_workers()
    print(f"Оптимальное количество параллельных задач: {workers}")
    
    # Функция для обработки одного индекса
    def process_single_index(index):
        try:
            # Импортируем функцию напрямую
            from test_model import check
            # Вызываем функцию проверки
            result = check(index, data_folder=data_folder)
            return index, result
        except Exception as e:
            import traceback
            return index, {
                "success": False, 
                "message": f"Ошибка при проверке индекса {index}: {str(e)}",
                "error_traceback": traceback.format_exc()
            }
    
    # Создаем диапазон индексов для обработки
    indices = range(start_index, end_index + 1)
    
    # Разбиваем индексы на батчи
    batch_size = max(1, workers * 2)  # Чтобы всегда были задачи для воркеров
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
    
    # Общее количество батчей
    total_batches = len(batches)
    
    # Обрабатываем каждый батч
    for batch_idx, batch in enumerate(batches):
        print(f"Обработка батча {batch_idx + 1}/{total_batches} (индексы {batch[0]}-{batch[-1]})")
        
        # Очищаем кэш CUDA перед началом нового батча, а не перед каждой проверкой
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        #     print(f"Кэш CUDA очищен перед батчем {batch_idx + 1}")
        
        # Запускаем параллельную обработку для текущего батча
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            batch_results = list(executor.map(process_single_index, batch))
        
        # Добавляем результаты в общий словарь
        for index, result in batch_results:
            all_results["results"][str(index)] = result
            all_results["total_processed"] += 1
            
            # Если есть ошибка, добавляем ее в список ошибок
            if not result.get("success", True):
                all_results["errors"].append({
                    "index": index,
                    "message": result.get("message", "Неизвестная ошибка")
                })
    
    # Рассчитываем общее время выполнения
    all_results["total_time"] = time.time() - start_time
    
    # Рассчитываем среднее время на проверку одного элемента
    if all_results["total_processed"] > 0:
        all_results["avg_time_per_item"] = all_results["total_time"] / all_results["total_processed"]
    
    return all_results

def run_check():
    """Запускает проверку с указанным индексом"""
    # Проверяем, готовы ли модели
    status = get_loading_status()
    if not status["loaded"]:
        put_error(f"Модели еще не загружены. Пожалуйста, подождите.\n{status['status']}")
        return
    
    # Сначала получаем индекс от пользователя
    index = pyinput("Введите индекс записи для проверки:", type='number', value=0)
    
    # Проверяем, выбрана ли папка с данными
    if not folder_paths["folder1"]:
        put_error("Сначала выберите папку с данными!")
        return
    
    # Создаем область для результатов и показываем индикатор загрузки
    with use_scope("results", clear=True):
        add_collapse_fix_script()
        put_loading(shape='grow')
        put_text("Запуск проверки, пожалуйста подождите...")
    
    # Получаем путь к папке с данными
    data_folder = folder_paths["folder1"]
    
    # Создаем именованный временный файл для логов
    log_file = "debug_log.txt"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Начало проверки с индексом {index} и папкой {data_folder}\n")
    
    try:
        # Импортируем функцию напрямую
        from test_model import check
        
        # Очищаем кэш CUDA перед запуском проверки
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        #     # Добавляем информацию о GPU в лог
        #     with open(log_file, 'a', encoding='utf-8') as f:
        #         f.write(f"\nИнформация о GPU:\n")
        #         f.write(f"CUDA доступна: {torch.cuda.is_available()}\n")
        #         if torch.cuda.is_available():
        #             f.write(f"Устройств CUDA: {torch.cuda.device_count()}\n")
        #             for i in range(torch.cuda.device_count()):
        #                 props = torch.cuda.get_device_properties(i)
        #                 f.write(f"GPU {i}: {props.name}\n")
        #                 f.write(f"  Общая память: {props.total_memory / 1024 / 1024 / 1024:.2f} ГБ\n")
        #                 f.write(f"  Вычислительная способность: {props.major}.{props.minor}\n")
        
        # Запускаем проверку в основном потоке
        # Это блокирует интерфейс, но зато надежно работает
        result = check(index, data_folder=data_folder)
        
        # Обновляем UI с результатами
        with use_scope("results", clear=True):
            # Если результат - словарь (новый формат)
            if isinstance(result, dict):
                # Формируем заголовок
                status_icon = "✅" if result.get("found_items", []) else "❌"
                header_text = f"Индекс {index} {status_icon} {'' if result.get('found_items', []) else '(Товары не найдены)'}"
                
                put_markdown(f"## {header_text}")
                
                # Информация о времени обработки
                if "processing_time" in result:
                    put_info(f"Время обработки: {result['processing_time']:.2f} секунд")
                
                # Статус проверки
                if not result["success"]:
                    put_error(f"Ошибка при проверке: {result['message']}")
                
                # Добавляем видеоплеер, если есть путь к видео (только один раз в начале)
                if "video_path" in result and os.path.exists(result["video_path"]):
                    # Создаем HTML для видеоплеера
                    video_html = create_video_html(result["video_path"])
                    put_html(video_html)
                
                # Информация о найденных товарах
                if "found_items" in result and result["found_items"]:
                    put_markdown("### Найденные товары в видео")
                    for item in result["found_items"]:
                        put_markdown(f"**[Товар {item['id']}](https://www.wildberries.ru/catalog/{item['id']}/detail.aspx)**")
                        put_text(f"Найден на таймкодах: {', '.join([str(t) for t in item['timecodes']])}")
                        put_text(f"Проверенные изображения: {', '.join(item['checked_images'])}")
                        
                        # Добавляем спойлер с изображениями товара
                        if "image_paths" in item:
                            images_html = create_images_html(item["image_paths"], item["id"])
                            put_collapse("Просмотреть изображения товара", put_html(images_html))
                
                # Информация о не найденных товарах
                if "not_found_items" in result and result["not_found_items"]:
                    put_markdown("### Товары, не найденные в видео")
                    for item in result["not_found_items"]:
                        put_markdown(f"**[Товар {item['id']}](https://www.wildberries.ru/catalog/{item['id']}/detail.aspx)**")
                        if "error" in item:
                            put_text(f"Причина: {item['error']}")
                        put_text(f"Проверенные изображения: {', '.join(item['checked_images'])}")
                        
                        # Добавляем спойлер с изображениями товара
                        if "image_paths" in item:
                            images_html = create_images_html(item["image_paths"], item["id"])
                            put_collapse("Просмотреть изображения товара", put_html(images_html))
                
                # Подробный лог
                if "raw_log" in result:
                    put_collapse("Подробный лог", put_code(result["raw_log"]))
            else:
                # Если результат - строка (старый формат)
                put_markdown("## Результаты проверки")
                put_code(result)
            
            # Добавляем ссылку на лог для отладки
            put_text("Для отладки доступен подробный лог:")
            put_file('debug_log.txt', open(log_file, 'rb').read(), 'Скачать лог')
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        
        # Записываем ошибку в лог
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\nОшибка при выполнении проверки:\n{str(e)}\n")
            f.write(error_trace)
        
        # Обновляем UI с сообщением об ошибке
        with use_scope("results", clear=True):
            put_error("Произошла ошибка при выполнении проверки:")
            put_code(str(e))
            put_code(error_trace)
            put_file('debug_log.txt', open(log_file, 'rb').read(), 'Скачать лог для отладки')

def run_batch_check():
    """
    Функция для запуска пакетной проверки.
    Запрашивает у пользователя начальный и конечный индексы и запускает проверку.
    """
    # Проверяем, готовы ли модели
    status = get_loading_status()
    if not status["loaded"]:
        put_error(f"Модели еще не загружены. Пожалуйста, подождите.\n{status['status']}")
        return
    
    # Проверяем, выбрана ли папка с данными
    if not folder_paths["folder1"]:
        put_error("Сначала выберите папку с данными!")
        return
    
    # Получаем начальный и конечный индексы от пользователя
    start_index = pyinput("Введите начальный индекс:", type='number', value=0)
    end_index = pyinput("Введите конечный индекс:", type='number', value=10)
    
    # Проверяем корректность индексов
    if start_index > end_index:
        put_error("Начальный индекс не может быть больше конечного!")
        return
    
    # Создаем область для результатов и показываем индикатор загрузки
    with use_scope("results", clear=True):
        add_collapse_fix_script()
        put_loading(shape='grow')
        put_text(f"Запуск пакетной проверки индексов {start_index}-{end_index}, пожалуйста подождите...")
    
    # Получаем путь к папке с данными
    data_folder = folder_paths["folder1"]
    
    # Создаем именованный временный файл для логов
    log_file = "batch_check_log.txt"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Начало пакетной проверки с индексами {start_index}-{end_index} и папкой {data_folder}\n")
    
    try:
        # Запускаем пакетную проверку
        results = batch_check(start_index, end_index, data_folder)
        
        # Обновляем UI с результатами
        with use_scope("results", clear=True):
            put_markdown("## Результаты пакетной проверки")
            
            # Информация о времени обработки
            put_info(f"Всего проверено: {results['total_processed']} индексов")
            put_info(f"Общее время: {results['total_time']:.2f} секунд")
            put_info(f"Среднее время на элемент: {results['avg_time_per_item']:.2f} секунд")
            
            # Если есть ошибки
            if results["errors"]:
                put_markdown("### Ошибки при проверке")
                for error in results["errors"]:
                    put_error(f"Индекс {error['index']}: {error['message']}")
            
            # Результаты для каждого индекса
            put_markdown("### Результаты по индексам")
            
            # Создаем аккордеон для результатов
            for index, result in results["results"].items():
                # Формируем заголовок аккордеона
                header = f"Индекс {index}"
                
                # Добавляем индикатор в зависимости от наличия найденных товаров
                if result.get("success", True):
                    if "found_items" in result and result["found_items"]:
                        header += f" ✅ (Найдено товаров: {len(result['found_items'])})"
                    else:
                        header += f" ❌ (Товары не найдены)"
                else:
                    header += " ⚠️ (Ошибка проверки)"
                
                # Создаем содержимое для аккордеона
                content = []
                
                # Добавляем видеоплеер, если есть путь к видео (только один раз в начале)
                if "video_path" in result and os.path.exists(result["video_path"]):
                    # Создаем HTML для видеоплеера
                    video_html = create_video_html(result["video_path"])
                    content.append(put_html(video_html))
                
                # Информация о найденных товарах
                if "found_items" in result and result["found_items"]:
                    content.append(put_markdown("#### Найденные товары в видео"))
                    for item in result["found_items"]:
                        content.append(put_markdown(f"**[Товар {item['id']}](https://www.wildberries.ru/catalog/{item['id']}/detail.aspx)**"))
                        content.append(put_text(f"Найден на таймкодах: {', '.join([str(t) for t in item['timecodes']])}"))
                        content.append(put_text(f"Проверенные изображения: {', '.join(item['checked_images'])}"))
                        
                        # Добавляем спойлер с изображениями товара
                        if "image_paths" in item:
                            images_html = create_images_html(item["image_paths"], item["id"])
                            content.append(put_collapse("Просмотреть изображения товара", put_html(images_html)))
                
                # Информация о не найденных товарах
                if "not_found_items" in result and result["not_found_items"]:
                    content.append(put_markdown("#### Товары, не найденные в видео"))
                    for item in result["not_found_items"]:
                        content.append(put_markdown(f"**[Товар {item['id']}](https://www.wildberries.ru/catalog/{item['id']}/detail.aspx)**")) #put_markdown(f"**Товар {item['id']}**")
                        if "error" in item:
                            content.append(put_text(f"Причина: {item['error']}"))
                        content.append(put_text(f"Проверенные изображения: {', '.join(item['checked_images'])}"))
                        
                        # Добавляем спойлер с изображениями товара
                        if "image_paths" in item:
                            images_html = create_images_html(item["image_paths"], item["id"])
                            content.append(put_collapse("Просмотреть изображения товара", put_html(images_html)))
                
                # Для ошибки
                if not result.get("success", True):
                    content.append(put_error(result.get("message", "Неизвестная ошибка")))
                
                # Добавляем подробный лог
                if "raw_log" in result:
                    content.append(put_collapse("Подробный лог", put_code(result["raw_log"])))
                
                # Добавляем аккордеон с результатами
                put_collapse(header, content)
            
            # Добавляем ссылку на лог для отладки
            put_text("Для отладки доступен подробный лог:")
            put_file('batch_check_log.txt', open(log_file, 'rb').read(), 'Скачать лог')
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        
        # Записываем ошибку в лог
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\nОшибка при выполнении пакетной проверки:\n{str(e)}\n")
            f.write(error_trace)
        
        # Обновляем UI с сообщением об ошибке
        with use_scope("results", clear=True):
            put_error("Произошла ошибка при выполнении пакетной проверки:")
            put_code(str(e))
            put_code(error_trace)
            put_file('batch_check_log.txt', open(log_file, 'rb').read(), 'Скачать лог для отладки')

# Функция для обновления состояния загрузки на странице
def update_loading_status():
    """Обновляет статус загрузки моделей на странице"""
    status = get_loading_status()
    
    # Получаем процент загрузки
    progress = status["progress"]
    
    # Обновляем статус загрузки
    run_js(f'''
        document.getElementById("loading_status").textContent = "{status['status']}";
        document.getElementById("loading_indicator").style.width = "{progress}%";
        document.getElementById("progress_text").textContent = "Загружено {status['current']} из {status['total']} моделей ({progress}%)";
        
        if ({str(status['loaded']).lower()}) {{
            // Убираем анимацию индикатора после завершения загрузки
            document.getElementById("loading_indicator").classList.remove("progress-bar-striped");
            document.getElementById("loading_indicator").classList.remove("progress-bar-animated");
            
            // Обновляем сообщение о статусе
            document.getElementById("loading_container").className = "alert alert-success";
        }} else {{
            // Обновляем сообщение о статусе
            document.getElementById("loading_container").className = "alert alert-info";
        }}
    ''')
    
    # Если модели загружены, обновляем кнопки запуска
    if status['loaded']:
        with use_scope('check_button_area', clear=True):
            put_row([
                put_button(
                    label="Проверить один индекс", 
                    onclick=run_check, 
                    color='primary',
                    disabled=False,
                    scope='check_button_area'
                ),
                put_button(
                    label="Пакетная проверка", 
                    onclick=run_batch_check, 
                    color='success',
                    disabled=False,
                    scope='check_button_area'
                )
            ])
    # Если произошла ошибка, тоже обновляем кнопки, но предупреждаем об ошибке
    elif status['error']:
        with use_scope('check_button_area', clear=True):
            put_row([
                put_button(
                    label="Модели не загружены (ошибка)", 
                    onclick=lambda: toast("Необходимо перезапустить приложение"), 
                    color='danger',
                    disabled=False,
                    scope='check_button_area'
                )
            ])
    
    # Если есть ошибка, показываем её
    if status['error']:
        with use_scope("loading_error", clear=True):
            put_error("Ошибка при загрузке моделей:")
            put_code(status['error'])
    
    # Если модели загружены, останавливаем таймер
    if status['loaded']:
        return False
    
    # Иначе продолжаем проверку каждую секунду
    return True

def check_gpu_status():
    """Проверяет статус GPU и выводит информацию"""
    info_str = ""
    
    if torch.cuda.is_available():
        info_str += "GPU доступен: ДА\n"
        info_str += f"Количество устройств CUDA: {torch.cuda.device_count()}\n"
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info_str += f"GPU {i}: {props.name}\n"
            info_str += f"  Общая память: {props.total_memory / 1024 / 1024 / 1024:.2f} ГБ\n"
            info_str += f"  Вычислительная способность: {props.major}.{props.minor}\n"
        
        # Информация о текущем использовании памяти
        info_str += "\nТекущее использование памяти:\n"
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            info_str += f"  GPU {i}: Выделено {allocated:.2f} МБ, Зарезервировано {reserved:.2f} МБ\n"
    else:
        info_str += "GPU доступен: НЕТ\n"
        info_str += "Будет использован CPU, что значительно замедлит работу.\n"
    
    return info_str

def main():
    """Основная функция приложения"""
    set_env(title="Проверка товаров в видео")
    
    # Проверяем статус GPU
    gpu_info = check_gpu_status()
    
    # Запускаем загрузку моделей
    preload_thread = start_preloading()
    
    put_markdown("# Проверка товаров в видео")
    
    # Выводим информацию о GPU
    put_collapse("Информация о GPU", put_code(gpu_info))
    
    # Добавляем контейнер для индикатора загрузки моделей
    html_loading = """
    <div id="loading_container" class="alert alert-info">
        <div style="margin-bottom: 10px;">
            <strong>Состояние системы:</strong> <span id="loading_status">Загрузка моделей...</span>
        </div>
        <div class="progress" style="height: 20px;">
            <div id="loading_indicator" class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" 
                 style="width: 0%;" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100"></div>
        </div>
        <div style="text-align: center; margin-top: 5px;">
            <small id="progress_text">Загружено 0 из 2 моделей (0%)</small>
        </div>
    </div>
    <style>
        .alert {
            padding: 15px;
            margin-bottom: 20px;
            border: 1px solid transparent;
            border-radius: 4px;
        }
        .alert-info {
            color: #31708f;
            background-color: #d9edf7;
            border-color: #bce8f1;
        }
        .alert-success {
            color: #3c763d;
            background-color: #dff0d8;
            border-color: #d6e9c6;
        }
        .progress {
            overflow: hidden;
            height: 20px;
            margin-bottom: 5px;
            background-color: #f5f5f5;
            border-radius: 4px;
            box-shadow: inset 0 1px 2px rgba(0,0,0,.1);
        }
        .progress-bar {
            float: left;
            width: 0;
            height: 100%;
            font-size: 12px;
            line-height: 20px;
            color: #fff;
            text-align: center;
            background-color: #337ab7;
            box-shadow: inset 0 -1px 0 rgba(0,0,0,.15);
            transition: width .6s ease;
        }
        .progress-bar-striped {
            background-image: linear-gradient(45deg,rgba(255,255,255,.15) 25%,transparent 25%,transparent 50%,rgba(255,255,255,.15) 50%,rgba(255,255,255,.15) 75%,transparent 75%,transparent);
            background-size: 40px 40px;
        }
        .progress-bar-animated {
            animation: progress-bar-stripes 2s linear infinite;
        }
        @keyframes progress-bar-stripes {
            from { background-position: 40px 0; }
            to { background-position: 0 0; }
        }
        button.disabled {
            opacity: 0.65;
            cursor: not-allowed;
        }
        button.success {
            color: #fff;
            background-color: #5cb85c;
            border-color: #4cae4c;
        }
    </style>
    """
    put_html(html_loading)
    
    # Область для вывода ошибок загрузки
    put_scope("loading_error")
    
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
    
    # Добавляем кнопки запуска с использованием PyWebIO API
    with use_scope('check_button_area'):
        # Кнопки будут неактивны при старте, и станут активными после загрузки моделей
        put_row([
            put_button(
                label="Проверить один индекс", 
                onclick=run_check, 
                color='primary',
                disabled=True,
                scope='check_button_area'
            ),
            put_button(
                label="Пакетная проверка", 
                onclick=run_batch_check, 
                color='success',
                disabled=True,
                scope='check_button_area'
            )
        ])
    
    # Область для вывода результатов
    put_markdown("## Результаты")
    put_scope("results")
    
    # Кнопка очистки путей и закрытия приложения
    put_markdown("---")
    put_row([
        put_button("Очистить кэш CUDA", onclick=clear_cuda, color='primary'),
        put_button("Очистить пути", onclick=clear_paths, color='warning'),
        put_button("Закрыть приложение", onclick=lambda: exit(), color='danger'),
    ])
    
    # Запускаем таймер для обновления статуса загрузки каждую секунду
    register_thread(info.task_id)
    
    # Обновляем статус загрузки каждую секунду
    while update_loading_status():
        time.sleep(1)
    
    # Запускаем периодическую проверку результатов в основном цикле
    while True:
        check_results()
        time.sleep(0.5)

def clear_cuda():
    """Очищает кэш CUDA"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        info_str = "Кэш CUDA очищен!\n\n"
        
        # Информация о текущем использовании памяти
        info_str += "Текущее использование памяти:\n"
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            info_str += f"GPU {i}: Выделено {allocated:.2f} МБ, Зарезервировано {reserved:.2f} МБ\n"
        
        toast(info_str)
    else:
        toast("CUDA недоступна в системе.")

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

def start_app():
    """Запускает приложение с кастомным обработчиком файлов"""
    import tornado.ioloop
    import tornado.web
    
    # Получаем обработчик для WebIO
    webio_handler_instance = webio_handler(main)
    
    # Создаем приложение с нашими обработчиками
    app = Application([
        # Обработчик для веб-интерфейса
        (r"/", webio_handler_instance),
        # Обработчик для /file=... URL
        (r"/file", FileHandler),
        # Обработчик для статических файлов (если нужно)
        (r"/static/(.*)", StaticFileHandler, {"path": os.path.abspath("./static")})
    ])
    
    # Запускаем сервер
    app.listen(8080)
    print(f"Сервер запущен на http://localhost:8080/")
    
    # Запускаем цикл событий
    tornado.ioloop.IOLoop.current().start()

if __name__ == '__main__':
    # Открываем браузер в отдельном потоке
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Создаем папку для статических файлов, если нужно
    os.makedirs("static", exist_ok=True)
    
    # Запускаем приложение
    start_app()