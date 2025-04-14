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

from model_preloader import start_preloading, get_loading_status
from test_comparison import optimized_check as check

class FileHandler(tornado.web.RequestHandler):
    
    def get(self):
        file_path = self.get_argument('file', None)
        if not file_path:
            self.set_status(400)
            self.write("Не указан путь к файлу")
            return
        
        file_path = urllib.parse.unquote(file_path)
        
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            self.set_status(404)
            self.write(f"Файл не найден: {file_path}")
            return
        
        content_type, _ = mimetypes.guess_type(file_path)
        if content_type:
            self.set_header('Content-Type', content_type)
        
        with open(file_path, 'rb') as f:
            self.write(f.read())

def create_video_html(video_path):
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
    
    html += """
        <div class="section-title">Проверенное видео:</div>
        <div class="video-container">
    """
    
    video_url = f"/file?file={video_path}"
    
    html += f"""
            <video id="{player_id}" controls preload="metadata" controlsList="nodownload" disablePictureInPicture disableRemotePlayback playsinline>
                <source src="{video_url}" type="video/mp4">
                Ваш браузер не поддерживает тег video.
            </video>
        </div>
    </div>
    
    <script>
    (function() {{
        function initializePlayer() {{
            const player = document.getElementById('{player_id}');
            if (!player) return;
            
            player.controls = true;
            
            player.addEventListener('click', function(e) {{
                e.stopPropagation();
            }});
            
            const controls = player.querySelectorAll('*');
            controls.forEach(control => {{
                control.addEventListener('click', function(e) {{
                    e.stopPropagation();
                }});
            }});
        }}
        
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initializePlayer);
        }} else {{
            initializePlayer();
        }}
        
        setTimeout(initializePlayer, 500);
        setTimeout(initializePlayer, 1000);
        setTimeout(initializePlayer, 2000);
    }})();
    </script>
    """
    
    return html

def create_images_html(image_paths, item_id=None):
    if not image_paths or len(image_paths) == 0:
        return "<div>Изображения товара отсутствуют.</div>"
    
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
    script = """
    <script>
    (function() {
        function fixVideoCollapse() {
            const collapseBtns = document.querySelectorAll('.collapse-btn');
            
            collapseBtns.forEach(btn => {
                const collapseId = btn.getAttribute('data-target');
                if (!collapseId) return;
                
                const collapseContent = document.querySelector(collapseId);
                if (!collapseContent) return;
                
                const hasVideo = collapseContent.querySelector('video');
                if (!hasVideo) return;
                
                btn.addEventListener('click', function() {
                    const videos = collapseContent.querySelectorAll('video');
                    videos.forEach(video => {
                        video.addEventListener('click', function(e) {
                            e.stopPropagation();
                        });
                        
                        video.controls = true;
                    });
                });
            });
        }
        
        setTimeout(fixVideoCollapse, 500);
        setTimeout(fixVideoCollapse, 2000);
        const observer = new MutationObserver(fixVideoCollapse);
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    })();
    </script>
    """
    put_html(script)

PATHS_FILE = "saved_paths.json"

def load_paths():
    if os.path.exists(PATHS_FILE):
        try:
            with open(PATHS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Ошибка чтения файла {PATHS_FILE} ({e}), создаем новый словарь")
    
    return {"folder1": "", "folder2": ""}

def save_paths(paths):
    try:
        with open(PATHS_FILE, 'w', encoding='utf-8') as f:
            json.dump(paths, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"Ошибка при сохранении путей: {e}")

folder_paths = load_paths()

results_queue = queue.Queue()

def select_folder(folder_id):
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
    while True:
        try:
            folder_id, path = results_queue.get_nowait()
            if path:  
                folder_paths[folder_id] = path
                save_paths(folder_paths)
                run_js(f'document.getElementById("path_{folder_id}").textContent = {repr(path)}')
            results_queue.task_done()
        except queue.Empty:
            break

def estimate_optimal_workers():
    if not torch.cuda.is_available():
        return max(1, multiprocessing.cpu_count() - 1)
    
    try:
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        free_memory = total_memory - torch.cuda.memory_allocated(device) - torch.cuda.memory_reserved(device)
        
        memory_per_task = 0.1 * 1024 * 1024 * 1024  
        
        max_workers = max(1, int(free_memory / memory_per_task))
        
        return min(max_workers, 6)  
    except Exception as e:
        print(f"Ошибка при оценке оптимального количества воркеров: {e}")
        return 1  

def batch_check(start_index, end_index, data_folder):
    start_time = time.time()
    
    all_results = {
        "start_index": start_index,
        "end_index": end_index,
        "total_processed": 0,
        "results": {},
        "errors": [],
        "total_time": 0,
        "avg_time_per_item": 0
    }
    
    workers = estimate_optimal_workers()
    print(f"Оптимальное количество параллельных задач: {workers}")
    
    def process_single_index(index):
        try:
            result = check(index, data_folder=data_folder)
            return index, result
        except Exception as e:
            import traceback
            return index, {
                "success": False, 
                "message": f"Ошибка при проверке индекса {index}: {str(e)}",
                "error_traceback": traceback.format_exc()
            }
    
    indices = range(start_index, end_index + 1)
    
    batch_size = max(1, workers * 2)  
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
    
    total_batches = len(batches)
    
    for batch_idx, batch in enumerate(batches):
        print(f"Обработка батча {batch_idx + 1}/{total_batches} (индексы {batch[0]}-{batch[-1]})")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            batch_results = list(executor.map(process_single_index, batch))
        
        for index, result in batch_results:
            all_results["results"][str(index)] = result
            all_results["total_processed"] += 1
            
            if not result.get("success", True):
                all_results["errors"].append({
                    "index": index,
                    "message": result.get("message", "Неизвестная ошибка")
                })
    
    all_results["total_time"] = time.time() - start_time
    
    if all_results["total_processed"] > 0:
        all_results["avg_time_per_item"] = all_results["total_time"] / all_results["total_processed"]
    
    return all_results

def run_check():
    status = get_loading_status()
    if not status["loaded"]:
        put_error(f"Модели еще не загружены. Пожалуйста, подождите.\n{status['status']}")
        return
    
    index = pyinput("Введите индекс записи для проверки:", type='number', value=0)
    
    if not folder_paths["folder1"]:
        put_error("Сначала выберите папку с данными!")
        return
    
    with use_scope("results", clear=True):
        add_collapse_fix_script()
        put_loading(shape='grow')
        put_text("Запуск проверки, пожалуйста подождите...")
    
    data_folder = folder_paths["folder1"]
    
    try:
        result = check(index, data_folder=data_folder)
        
        with use_scope("results", clear=True):
            if isinstance(result, dict):
                status_icon = "✅" if result.get("found_items", []) else "❌"
                header_text = f"Индекс {index} {status_icon} {'' if result.get('found_items', []) else '(Товары не найдены)'}"
                
                put_markdown(f"## {header_text}")
                
                if "processing_time" in result:
                    put_info(f"Время обработки: {result['processing_time']:.2f} секунд")
                
                if not result["success"]:
                    put_error(f"Ошибка при проверке: {result['message']}")
                
                if "video_path" in result and os.path.exists(result["video_path"]):
                    video_html = create_video_html(result["video_path"])
                    put_html(video_html)
                
                if "found_items" in result and result["found_items"]:
                    put_markdown("### Найденные товары в видео")
                    for item in result["found_items"]:
                        put_markdown(f"**[Товар {item['id']}](https://www.wildberries.ru/catalog/{item['id']}/detail.aspx)**")
                        put_text(f"Найден на таймкодах: {', '.join([str(t) for t in item['timecodes']])}")
                        put_text(f"Проверенные изображения: {', '.join(item['checked_images'])}")
                        
                        if "image_paths" in item:
                            images_html = create_images_html(item["image_paths"], item["id"])
                            put_collapse("Просмотреть изображения товара", put_html(images_html))
                
                if "not_found_items" in result and result["not_found_items"]:
                    put_markdown("### Товары, не найденные в видео")
                    for item in result["not_found_items"]:
                        put_markdown(f"**[Товар {item['id']}](https://www.wildberries.ru/catalog/{item['id']}/detail.aspx)**")
                        if "error" in item:
                            put_text(f"Причина: {item['error']}")
                        put_text(f"Проверенные изображения: {', '.join(item['checked_images'])}")
                        
                        if "image_paths" in item:
                            images_html = create_images_html(item["image_paths"], item["id"])
                            put_collapse("Просмотреть изображения товара", put_html(images_html))
                
                if "raw_log" in result:
                    put_collapse("Подробный лог", put_code(result["raw_log"]))
            else:
                put_markdown("## Результаты проверки")
                put_code(result)
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        
        with use_scope("results", clear=True):
            put_error("Произошла ошибка при выполнении проверки:")
            put_code(str(e))
            put_code(error_trace)

def run_batch_check():
    status = get_loading_status()
    if not status["loaded"]:
        put_error(f"Модели еще не загружены. Пожалуйста, подождите.\n{status['status']}")
        return
    
    if not folder_paths["folder1"]:
        put_error("Сначала выберите папку с данными!")
        return
    
    start_index = pyinput("Введите начальный индекс:", type='number', value=0)
    end_index = pyinput("Введите конечный индекс:", type='number', value=10)
    
    if start_index > end_index:
        put_error("Начальный индекс не может быть больше конечного!")
        return
    
    with use_scope("results", clear=True):
        add_collapse_fix_script()
        put_loading(shape='grow')
        put_text(f"Запуск пакетной проверки индексов {start_index}-{end_index}, пожалуйста подождите...")
    
    data_folder = folder_paths["folder1"]
    
    try:
        results = batch_check(start_index, end_index, data_folder)
        
        with use_scope("results", clear=True):
            put_markdown("## Результаты пакетной проверки")
            
            put_info(f"Всего проверено: {results['total_processed']} индексов")
            put_info(f"Общее время: {results['total_time']:.2f} секунд")
            put_info(f"Среднее время на элемент: {results['avg_time_per_item']:.2f} секунд")
            
            if results["errors"]:
                put_markdown("### Ошибки при проверке")
                for error in results["errors"]:
                    put_error(f"Индекс {error['index']}: {error['message']}")
            
            put_markdown("### Результаты по индексам")
            
            for index, result in results["results"].items():
                header = f"Индекс {index}"
                
                if result.get("success", True):
                    if "found_items" in result and result["found_items"]:
                        header += f" ✅ (Найдено товаров: {len(result['found_items'])})"
                    else:
                        header += f" ❌ (Товары не найдены)"
                else:
                    header += " ⚠️ (Ошибка проверки)"
                
                content = []
                
                if "video_path" in result and os.path.exists(result["video_path"]):
                    video_html = create_video_html(result["video_path"])
                    content.append(put_html(video_html))
                
                if "found_items" in result and result["found_items"]:
                    content.append(put_markdown("#### Найденные товары в видео"))
                    for item in result["found_items"]:
                        content.append(put_markdown(f"**[Товар {item['id']}](https://www.wildberries.ru/catalog/{item['id']}/detail.aspx)**"))
                        content.append(put_text(f"Найден на таймкодах: {', '.join([str(t) for t in item['timecodes']])}"))
                        content.append(put_text(f"Проверенные изображения: {', '.join(item['checked_images'])}"))
                        
                        if "image_paths" in item:
                            images_html = create_images_html(item["image_paths"], item["id"])
                            content.append(put_collapse("Просмотреть изображения товара", put_html(images_html)))
                
                if "not_found_items" in result and result["not_found_items"]:
                    content.append(put_markdown("#### Товары, не найденные в видео"))
                    for item in result["not_found_items"]:
                        content.append(put_markdown(f"**[Товар {item['id']}](https://www.wildberries.ru/catalog/{item['id']}/detail.aspx)**"))
                        if "error" in item:
                            content.append(put_text(f"Причина: {item['error']}"))
                        content.append(put_text(f"Проверенные изображения: {', '.join(item['checked_images'])}"))
                        
                        if "image_paths" in item:
                            images_html = create_images_html(item["image_paths"], item["id"])
                            content.append(put_collapse("Просмотреть изображения товара", put_html(images_html)))
                
                if not result.get("success", True):
                    content.append(put_error(result.get("message", "Неизвестная ошибка")))
                
                if "raw_log" in result:
                    content.append(put_collapse("Подробный лог", put_code(result["raw_log"])))
                
                put_collapse(header, content)
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        
        with use_scope("results", clear=True):
            put_error("Произошла ошибка при выполнении пакетной проверки:")
            put_code(str(e))
            put_code(error_trace)

def update_loading_status():
    status = get_loading_status()
    
    progress = status["progress"]
    
    run_js(f'''
        document.getElementById("loading_status").textContent = "{status['status']}";
        document.getElementById("loading_indicator").style.width = "{progress}%";
        document.getElementById("progress_text").textContent = "Загружено {status['current']} из {status['total']} моделей ({progress}%)";
        
        if ({str(status['loaded']).lower()}) {{
            document.getElementById("loading_indicator").classList.remove("progress-bar-striped");
            document.getElementById("loading_indicator").classList.remove("progress-bar-animated");
            
            document.getElementById("loading_container").className = "alert alert-success";
        }} else {{
            document.getElementById("loading_container").className = "alert alert-info";
        }}
    ''')
    
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
    
    if status['error']:
        with use_scope("loading_error", clear=True):
            put_error("Ошибка при загрузке моделей:")
            put_code(status['error'])
    
    if status['loaded']:
        return False
    
    return True

def check_gpu_status():
    info_str = ""
    
    if torch.cuda.is_available():
        info_str += "GPU доступен: ДА\n"
        info_str += f"Количество устройств CUDA: {torch.cuda.device_count()}\n"
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info_str += f"GPU {i}: {props.name}\n"
            info_str += f"  Общая память: {props.total_memory / 1024 / 1024 / 1024:.2f} ГБ\n"
            info_str += f"  Вычислительная способность: {props.major}.{props.minor}\n"
        
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

def clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        info_str = "Кэш CUDA очищен!\n\n"
        
        info_str += "Текущее использование памяти:\n"
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            info_str += f"GPU {i}: Выделено {allocated:.2f} МБ, Зарезервировано {reserved:.2f} МБ\n"
        
        toast(info_str)
    else:
        toast("CUDA недоступна в системе.")

def main():
    set_env(title="Проверка товаров в видео")
    
    gpu_info = check_gpu_status()
    
    preload_thread = start_preloading()
    
    put_markdown("# Проверка товаров в видео")
    
    put_collapse("Информация о GPU", put_code(gpu_info))
    
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
    
    put_scope("loading_error")
    
    put_info("Выберите папку с данными, содержащую файл videos.json и подпапки с изображениями.")
    
    put_row([
        put_button("Выбрать папку с данными", onclick=lambda: select_folder("folder1"), color='primary'),
    ])
    
    folder1_path = folder_paths["folder1"] or "Путь к папке с данными будет отображен здесь"
    html = f"""
    <div style="margin-top: 5px; padding: 10px; border: 1px solid #ddd; background-color: #f9f9f9;">
        <code id="path_folder1">{folder1_path}</code>
    </div>
    """
    put_html(html)
    
    put_markdown("---")
    
    put_markdown("## Запуск проверки")
    
    with use_scope('check_button_area'):
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
    
    put_markdown("## Проверка для экспертов")
    put_info("Модуль для расширенной проверки, требует настройки экспертом.")
    put_button("Запустить экспертную проверку", onclick=expert_check, color='warning')
    
    put_markdown("## Результаты")
    put_scope("results")
    
    register_thread(info.task_id)
    
    while update_loading_status():
        time.sleep(1)
    
    while True:
        check_results()
        time.sleep(0.5)

def open_browser():
    time.sleep(1)
    webbrowser.open("http://localhost:8080/")

def start_app():
    import tornado.ioloop
    import tornado.web
    
    webio_handler_instance = webio_handler(main)
    
    app = Application([
        (r"/", webio_handler_instance),
        (r"/file", FileHandler),
        (r"/static/(.*)", StaticFileHandler, {"path": os.path.abspath("./static")})
    ])
    
    app.listen(8080)
    print(f"Сервер запущен на http://localhost:8080/")
    
    tornado.ioloop.IOLoop.current().start()

def expert_check():
    status = get_loading_status()
    if not status["loaded"]:
        put_error("Модели еще не загружены. Пожалуйста, подождите.")
        return
    
    if not folder_paths["folder1"]:
        put_error("Сначала выберите папку с данными!")
        return
    
    with use_scope("results", clear=True):
        put_loading(shape='grow')
        put_text("Запуск экспертной проверки, пожалуйста подождите...")
    
    try:
        from test_comparison import expert_custom_check
        
        result = expert_custom_check(folder_paths["folder1"])
        
        with use_scope("results", clear=True):
            if isinstance(result, str):
                put_text(result)
            else:
                put_code(str(result))
                
    except ImportError:
        with use_scope("results", clear=True):
            put_error("Функция expert_custom_check не найдена в файле test_comparison.py")
    except Exception as e:
        with use_scope("results", clear=True):
            put_error(f"Ошибка при выполнении экспертной проверки: {str(e)}")

if __name__ == '__main__':
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    os.makedirs("static", exist_ok=True)
    
    start_app()