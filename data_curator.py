import os
import json
import torch
import numpy as np
import cv2
import time
import argparse
from tqdm import tqdm
from pathlib import Path
import random
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import shutil

# Настройка трансформации для модели
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class DataCurator:
    def __init__(self, data_root, videos_json, output_dir="curated_data", num_frames=10, gpu=True):
        """
        Инициализация куратора данных
        
        Args:
            data_root: Корневая директория с данными
            videos_json: Путь к JSON-файлу с видео
            output_dir: Директория для сохранения кураторских данных 
            num_frames: Количество ключевых кадров для извлечения из каждого видео
            gpu: Использовать ли GPU
        """
        self.data_root = os.path.abspath(data_root)
        self.videos_json = videos_json
        self.output_dir = os.path.abspath(output_dir)
        self.num_frames = num_frames
        
        # Создаем директорию для сохранения кадров
        self.frames_dir = os.path.join(self.output_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        
        # Создаем директорию для разметки
        self.labels_dir = os.path.join(self.output_dir, "labels")
        os.makedirs(self.labels_dir, exist_ok=True)
        
        # Устройство для инференса
        self.device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
        print(f"Используемое устройство: {self.device}")
        
        # Загружаем модель
        self.model = self._load_model()
        
        # Загружаем данные о видео
        self.videos_data = self._load_json(videos_json)
        print(f"Загружено {len(self.videos_data)} видео из JSON")
        
        # Словарь для хранения информации о кадрах
        self.frames_info = {}
    
    def _load_model(self):
        """Загрузка предобученной модели для фильтрации кадров"""
        print("Загрузка предобученной модели...")
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.eval()
        model.to(self.device)
        return model
    
    def _load_json(self, path):
        """Загрузка JSON-файла"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Ошибка при загрузке JSON из {path}: {e}")
            return []
    
    def extract_key_frames(self, video_path, video_id, num_frames=10):
        """
        Извлечение ключевых кадров из видео
        
        Args:
            video_path: Путь к видео
            video_id: ID видео
            num_frames: Количество ключевых кадров
            
        Returns:
            frames_data: Список путей к сохраненным кадрам и их характеристики
        """
        try:
            # Создаем директорию для кадров этого видео
            video_frames_dir = os.path.join(self.frames_dir, f"video_{video_id}")
            os.makedirs(video_frames_dir, exist_ok=True)
            
            # Проверяем, если уже есть извлеченные кадры
            existing_frames = [f for f in os.listdir(video_frames_dir) if f.endswith('.jpg')]
            if len(existing_frames) >= num_frames:
                print(f"Для видео {video_id} уже извлечены кадры ({len(existing_frames)})")
                frames_data = []
                for frame_file in existing_frames:
                    frame_path = os.path.join(video_frames_dir, frame_file)
                    frame_path = os.path.normpath(frame_path)
                    timestamp = float(frame_file.split('_')[1].split('.')[0])
                    frames_data.append({
                        'path': frame_path,
                        'timestamp': timestamp,
                        'video_id': video_id
                    })
                return frames_data
            
            # Загружаем видео
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Не удалось открыть видео {video_path}")
                return []
            
            # Получаем информацию о видео
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 25  # Стандартное значение
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            # Вычисляем шаг для равномерного распределения кадров
            if total_frames <= num_frames:
                # Если кадров меньше, чем нужно, берем все
                frame_indices = list(range(total_frames))
            else:
                # Иначе равномерно распределяем
                step = total_frames // num_frames
                frame_indices = [i * step for i in range(num_frames)]
            
            frames_data = []
            
            # Извлекаем кадры
            for i in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Вычисляем временную метку
                timestamp = i / fps
                
                # Сохраняем кадр
                frame_path = os.path.join(video_frames_dir, f"frame_{timestamp:.2f}.jpg")
                frame_path = os.path.normpath(frame_path)
                cv2.imwrite(frame_path, frame)
                
                frames_data.append({
                    'path': frame_path,
                    'timestamp': timestamp,
                    'video_id': video_id
                })
            
            cap.release()
            print(f"Извлечено {len(frames_data)} кадров из видео {video_id}")
            return frames_data
            
        except Exception as e:
            print(f"Ошибка при извлечении кадров из видео {video_id}: {e}")
            return []
    
    def calculate_frame_score(self, frame_path):
        """
        Расчет "интересности" кадра с помощью модели
        
        Args:
            frame_path: Путь к кадру
            
        Returns:
            score: Оценка кадра (0-1)
        """
        try:
            # Загружаем изображение
            image = Image.open(frame_path).convert('RGB')
            
            # Применяем трансформации
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Получаем предсказания модели
            with torch.no_grad():
                output = self.model(image_tensor)
            
            # Вычисляем "интересность" как энтропию предсказаний
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            entropy_normalized = entropy / np.log2(1000)  # Нормализуем к диапазону [0, 1]
            
            return float(entropy_normalized)
            
        except Exception as e:
            print(f"Ошибка при расчете оценки кадра {frame_path}: {e}")
            return 0.0
    
    def filter_frames(self, frames_data, min_score=0.5):
        """
        Фильтрация кадров по оценке "интересности"
        
        Args:
            frames_data: Список с информацией о кадрах
            min_score: Минимальная оценка для сохранения кадра
            
        Returns:
            filtered_frames: Отфильтрованные кадры
        """
        # Рассчитываем оценки для всех кадров
        for frame in tqdm(frames_data, desc="Оценка кадров"):
            frame['score'] = self.calculate_frame_score(frame['path'])
        
        # Сортируем кадры по оценке
        frames_data.sort(key=lambda x: x['score'], reverse=True)
        
        # Берем лучшие кадры
        filtered_frames = [f for f in frames_data if f['score'] >= min_score]
        
        # Если кадров мало, берем минимум 3 лучших кадра
        if len(filtered_frames) < 3:
            filtered_frames = frames_data[:min(3, len(frames_data))]
        
        print(f"Отфильтровано {len(filtered_frames)} кадров из {len(frames_data)}")
        return filtered_frames
    
    def get_product_images(self, nm_id):
        """Получение путей к изображениям товара"""
        image_paths = []
        
        # Проверяем наличие основного изображения
        img1_path = os.path.join(self.data_root, f"images_1/{nm_id}/1.webp")
        img1_path = os.path.normpath(img1_path)
        if os.path.exists(img1_path):
            image_paths.append(img1_path)
        
        # Проверяем наличие дополнительных изображений
        for img_idx in [2, 3]:
            img_path = os.path.join(self.data_root, f"images_2-3/{nm_id}/{img_idx}.webp")
            img_path = os.path.normpath(img_path)
            if os.path.exists(img_path):
                image_paths.append(img_path)
        
        # Проверяем наличие дополнительных изображений 4-5
        for img_idx in [4, 5]:
            img_path = os.path.join(self.data_root, f"images_4-5/{nm_id}/{img_idx}.webp")
            img_path = os.path.normpath(img_path)
            if os.path.exists(img_path):
                image_paths.append(img_path)
        
        return image_paths
    
    def download_video(self, url, output_path):
        """
        Загрузка видео с помощью ffmpeg
        
        Args:
            url: URL видео
            output_path: Путь для сохранения
            
        Returns:
            success: Успешность загрузки
        """
        try:
            output_path = os.path.normpath(output_path)
            
            # Проверяем, существует ли уже файл
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
                print(f"Видео уже загружено: {output_path}")
                return True
            
            # Создаем директорию для сохранения
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Загружаем видео с помощью ffmpeg
            import subprocess
            command = [
                'ffmpeg',
                '-i', url,
                '-c', 'copy',
                '-v', 'error',
                output_path
            ]
            
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Ошибка при загрузке видео: {result.stderr}")
                return False
            
            return os.path.exists(output_path)
            
        except Exception as e:
            print(f"Ошибка при загрузке видео: {e}")
            return False
    
    def process_video(self, video_data, temp_dir="temp"):
        """
        Обработка одного видео - загрузка, извлечение и фильтрация кадров
        
        Args:
            video_data: Данные о видео
            temp_dir: Директория для временных файлов
            
        Returns:
            frames_info: Информация о кадрах
        """
        try:
            video_id = video_data.get('video_id')
            video_url = video_data.get('path_url')
            nm_ids = video_data.get('nm_ids', [])
            
            if not nm_ids:
                print(f"Пропуск видео {video_id} - нет связанных товаров")
                return None
            
            # Создаем временную директорию
            temp_dir = os.path.abspath(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
            
            # Загружаем видео
            video_path = os.path.join(temp_dir, f"video_{video_id}.mp4")
            video_path = os.path.normpath(video_path)
            if not self.download_video(video_url, video_path):
                print(f"Не удалось загрузить видео {video_id}")
                return None
            
            # Извлекаем ключевые кадры
            frames_data = self.extract_key_frames(video_path, video_id, self.num_frames)
            
            # Фильтруем кадры
            filtered_frames = self.filter_frames(frames_data)
            
            # Получаем информацию о товарах
            products_info = {}
            for nm_id in nm_ids:
                product_images = self.get_product_images(nm_id)
                if product_images:
                    products_info[nm_id] = product_images
            
            # Сохраняем информацию о кадрах и товарах
            result = {
                'video_id': video_id,
                'video_path': video_path,
                'frames': filtered_frames,
                'products': products_info
            }
            
            # Удаляем видео для экономии места
            # os.remove(video_path)
            
            return result
            
        except Exception as e:
            print(f"Ошибка при обработке видео {video_data.get('video_id')}: {e}")
            return None
    
    def process_all_videos(self, max_videos=None, num_workers=4):
        """
        Обработка всех видео
        
        Args:
            max_videos: Максимальное количество видео для обработки
            num_workers: Количество параллельных потоков
            
        Returns:
            all_frames_info: Информация о всех кадрах
        """
        # Ограничиваем количество видео, если указано
        videos_to_process = self.videos_data
        if max_videos is not None and max_videos > 0:
            videos_to_process = videos_to_process[:max_videos]
        
        print(f"Обработка {len(videos_to_process)} видео...")
        
        # Обрабатываем видео параллельно
        all_results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.process_video, video) for video in videos_to_process]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Обработка видео"):
                result = future.result()
                if result:
                    all_results.append(result)
        
        # Сохраняем результаты в файл
        results_file = os.path.join(self.output_dir, "frames_info.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"Обработано {len(all_results)} видео. Результаты сохранены в {results_file}")
        return all_results

    def run_manual_curation(self, frames_info):
        """
        Запуск графического интерфейса для ручной разметки
        
        Args:
            frames_info: Информация о кадрах
        """
        app = FrameCuratorApp(frames_info, self.output_dir)
        app.run()


class FrameCuratorApp:
    def __init__(self, frames_info, output_dir):
        """
        Инициализация приложения для разметки кадров
        
        Args:
            frames_info: Информация о кадрах
            output_dir: Директория для сохранения результатов
        """
        self.frames_info = frames_info
        self.output_dir = os.path.abspath(output_dir)
        self.current_video_index = 0
        self.current_frame_index = 0
        self.current_product_index = 0
        
        # Директория для сохранения разметки
        self.labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(self.labels_dir, exist_ok=True)
        
        # Директория для позитивных пар
        self.positive_pairs_dir = os.path.join(output_dir, "positive_pairs")
        os.makedirs(self.positive_pairs_dir, exist_ok=True)
        
        # Словарь для хранения меток
        self.labels = {}
        
        # Загружаем сохраненные метки, если есть
        self.load_labels()

        self.find_first_unmarked_pair()
        
        # Создаем окно
        self.root = tk.Tk()
        self.root.title("Разметка кадров")
        self.root.geometry("1200x800")
        
        # Создаем основной интерфейс
        self.create_widgets()

    def find_first_unmarked_pair(self):
        if not self.frames_info:
            return
        
        for video_idx, video in enumerate(self.frames_info):
            for frame_idx, frame in enumerate(video.get('frames', [])):
                frame_path = frame['path']
                
                product_ids = list(video.get('products', {}).keys())
                for product_idx, product_id in enumerate(product_ids):
                    product_images = video['products'][product_id]
                    if not product_images:
                        continue
                    
                    product_path = product_images[0]
                    
                    # Проверяем, есть ли метка для этой пары
                    pair_id = self.generate_pair_id(frame_path, product_path)
                    if pair_id not in self.labels:
                        # Нашли неразмеченную пару
                        self.current_video_index = video_idx
                        self.current_frame_index = frame_idx
                        self.current_product_index = product_idx
                        return
        
        # Если не нашли неразмеченные пары, оставляем текущие индексы
    
    def create_widgets(self):
        """Создание виджетов интерфейса"""
        # Верхняя панель с информацией
        self.info_frame = ttk.Frame(self.root, padding=10)
        self.info_frame.pack(fill="x")
        
        self.video_label = ttk.Label(self.info_frame, text="Видео: 0/0")
        self.video_label.pack(side="left", padx=5)
        
        self.frame_label = ttk.Label(self.info_frame, text="Кадр: 0/0")
        self.frame_label.pack(side="left", padx=5)
        
        self.product_label = ttk.Label(self.info_frame, text="Товар: 0/0")
        self.product_label.pack(side="left", padx=5)
        
        # Фрейм для отображения изображений
        self.images_frame = ttk.Frame(self.root, padding=10)
        self.images_frame.pack(fill="both", expand=True)
        
        # Левая панель - кадр видео
        self.frame_panel = ttk.LabelFrame(self.images_frame, text="Кадр видео", padding=10)
        self.frame_panel.pack(side="left", fill="both", expand=True)
        
        self.frame_image_label = ttk.Label(self.frame_panel)
        self.frame_image_label.pack(fill="both", expand=True)
        
        # Правая панель - изображение товара
        self.product_panel = ttk.LabelFrame(self.images_frame, text="Товар", padding=10)
        self.product_panel.pack(side="right", fill="both", expand=True)
        
        self.product_image_label = ttk.Label(self.product_panel)
        self.product_image_label.pack(fill="both", expand=True)
        
        # Нижняя панель с кнопками
        self.buttons_frame = ttk.Frame(self.root, padding=10)
        self.buttons_frame.pack(fill="x")
        
        self.yes_button = ttk.Button(self.buttons_frame, text="Да (товар есть на кадре)", command=self.mark_positive)
        self.yes_button.pack(side="left", padx=5, fill="x", expand=True)
        
        self.no_button = ttk.Button(self.buttons_frame, text="Нет (товара нет на кадре)", command=self.mark_negative)
        self.no_button.pack(side="left", padx=5, fill="x", expand=True)
        
        self.skip_button = ttk.Button(self.buttons_frame, text="Пропустить", command=self.skip_current)
        self.skip_button.pack(side="left", padx=5, fill="x", expand=True)
        
        self.save_button = ttk.Button(self.buttons_frame, text="Сохранить и выйти", command=self.save_and_exit)
        self.save_button.pack(side="left", padx=5, fill="x", expand=True)
        
        # Статус-бар
        self.status_var = tk.StringVar()
        self.status_var.set("Готово к разметке")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        self.status_bar.pack(fill="x", side="bottom", pady=5)
        
        # Прогресс-бар
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.root, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", side="bottom", pady=5)
        
        # Привязываем горячие клавиши
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<Up>", lambda e: self.prev_product())
        self.root.bind("<Down>", lambda e: self.next_product())
        self.root.bind("<y>", lambda e: self.mark_positive())
        self.root.bind("<n>", lambda e: self.mark_negative())
        self.root.bind("<s>", lambda e: self.skip_current())
        self.root.bind("<Escape>", lambda e: self.save_and_exit())
    
    def generate_pair_id(self, frame_path, product_path):
        """
        Генерирует уникальный идентификатор для пары кадр-товар
        Вместо соединения путей подчеркиванием, используем хеш от обоих путей
        """
        # Нормализуем пути
        frame_path = os.path.normpath(frame_path)
        product_path = os.path.normpath(product_path)
        
        # Создаем уникальный идентификатор
        pair_id = f"{hash(frame_path)}_{hash(product_path)}"
        return pair_id
    
    def load_labels(self):
        """Загрузка сохраненных меток"""
        labels_file = os.path.join(self.labels_dir, "labels.json")
        if os.path.exists(labels_file):
            try:
                with open(labels_file, 'r', encoding='utf-8') as f:
                    labels_data = json.load(f)
                    
                # В новом формате метки хранятся в виде структуры с frame_path и product_path
                if isinstance(labels_data, list):
                    for item in labels_data:
                        frame_path = item.get('frame_path')
                        product_path = item.get('product_path')
                        label = item.get('label')
                        
                        if frame_path and product_path and label is not None:
                            pair_id = self.generate_pair_id(frame_path, product_path)
                            self.labels[pair_id] = {
                                'frame_path': frame_path,
                                'product_path': product_path,
                                'label': label
                            }
                # Совместимость со старым форматом
                elif isinstance(labels_data, dict):
                    # Преобразуем старый формат в новый
                    for pair_key, label_value in labels_data.items():
                        parts = pair_key.split('_')
                        if len(parts) >= 2:
                            # Предполагаем, что последняя часть - это путь к товару
                            product_path = parts[-1]
                            # Остальные части составляют путь к кадру
                            frame_path = '_'.join(parts[:-1])
                            
                            pair_id = self.generate_pair_id(frame_path, product_path)
                            self.labels[pair_id] = {
                                'frame_path': frame_path,
                                'product_path': product_path,
                                'label': label_value
                            }
                
                print(f"Загружено {len(self.labels)} меток")
            except Exception as e:
                print(f"Ошибка при загрузке меток: {e}")
                self.labels = {}
    
    def save_labels(self):
        """Сохранение меток"""
        labels_file = os.path.join(self.labels_dir, "labels.json")
        try:
            # Создаем список меток в новом формате
            labels_list = []
            for pair_id, label_data in self.labels.items():
                labels_list.append({
                    'frame_path': label_data['frame_path'],
                    'product_path': label_data['product_path'],
                    'label': label_data['label']
                })
            
            with open(labels_file, 'w', encoding='utf-8') as f:
                json.dump(labels_list, f, ensure_ascii=False, indent=2)
            print(f"Сохранено {len(labels_list)} меток в {labels_file}")
        except Exception as e:
            print(f"Ошибка при сохранении меток: {e}")
    
    def generate_training_data(self):
        """Генерация данных для обучения из разметки"""
        try:
            positive_pairs = []
            negative_pairs = []
            
            # Собираем положительные и отрицательные пары
            for pair_id, label_data in self.labels.items():
                frame_path = label_data['frame_path']
                product_path = label_data['product_path']
                label = label_data['label']
                
                # Проверяем существование файлов
                if not os.path.exists(frame_path):
                    print(f"Предупреждение: Файл кадра не найден: {frame_path}")
                    continue
                
                if not os.path.exists(product_path):
                    print(f"Предупреждение: Файл товара не найден: {product_path}")
                    continue
                
                # Создаем пару
                if label == 1:
                    positive_pairs.append({
                        "frame": frame_path,
                        "product": product_path
                    })
                else:
                    negative_pairs.append({
                        "frame": frame_path,
                        "product": product_path
                    })
            
            # Сохраняем JSON с путями к парам
            pairs_data = {
                "positive_pairs": positive_pairs,
                "negative_pairs": negative_pairs
            }
            
            pairs_file = os.path.join(self.output_dir, "training_pairs.json")
            with open(pairs_file, 'w', encoding='utf-8') as f:
                json.dump(pairs_data, f, ensure_ascii=False, indent=2)
            
            # Создаем плоский список для использования в training_with_curated_data.py
            training_data = []
            
            for pair in positive_pairs:
                training_data.append({
                    'frame_path': pair['frame'],
                    'product_path': pair['product'],
                    'label': 1
                })
            
            for pair in negative_pairs:
                training_data.append({
                    'frame_path': pair['frame'],
                    'product_path': pair['product'],
                    'label': 0
                })
            
            # Сохраняем файл в новом формате
            training_data_file = os.path.join(self.output_dir, "training_data.json")
            with open(training_data_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, ensure_ascii=False, indent=2)
            
            print(f"Сгенерированы данные для обучения: {len(positive_pairs)} положительных, {len(negative_pairs)} отрицательных пар")
            
            return pairs_file, training_data_file
            
        except Exception as e:
            print(f"Ошибка при генерации данных для обучения: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def update_display(self):
        """Обновление отображения текущей пары изображений"""
        if not self.frames_info:
            self.status_var.set("Нет данных для отображения")
            return
        
        try:
            # Получаем текущее видео
            current_video = self.frames_info[self.current_video_index]
            
            # Получаем текущий кадр
            if not current_video.get('frames'):
                self.status_var.set(f"Нет кадров для видео {current_video.get('video_id')}")
                return
            
            current_frame = current_video['frames'][self.current_frame_index]
            frame_path = current_frame['path']
            
            # Получаем текущий товар
            product_ids = list(current_video.get('products', {}).keys())
            if not product_ids:
                self.status_var.set(f"Нет товаров для видео {current_video.get('video_id')}")
                return
            
            current_product_id = product_ids[self.current_product_index]
            product_images = current_video['products'][current_product_id]
            
            if not product_images:
                self.status_var.set(f"Нет изображений для товара {current_product_id}")
                return
            
            product_path = product_images[0]
            
            # Загружаем и отображаем изображения
            frame_image = Image.open(frame_path).convert('RGB')
            product_image = Image.open(product_path).convert('RGB')
            
            # Изменяем размер для отображения
            frame_image = self.resize_image(frame_image, (500, 400))
            product_image = self.resize_image(product_image, (500, 400))
            
            # Преобразуем в формат для Tkinter
            frame_tk = ImageTk.PhotoImage(frame_image)
            product_tk = ImageTk.PhotoImage(product_image)
            
            # Обновляем изображения на экране
            self.frame_image_label.configure(image=frame_tk)
            self.frame_image_label.image = frame_tk
            
            self.product_image_label.configure(image=product_tk)
            self.product_image_label.image = product_tk
            
            # Обновляем метки
            self.video_label.configure(text=f"Видео: {self.current_video_index+1}/{len(self.frames_info)} (ID: {current_video['video_id']})")
            self.frame_label.configure(text=f"Кадр: {self.current_frame_index+1}/{len(current_video['frames'])} (Время: {current_frame['timestamp']:.2f}с)")
            self.product_label.configure(text=f"Товар: {self.current_product_index+1}/{len(product_ids)} (ID: {current_product_id})")
            
            # Проверяем, есть ли уже метка для этой пары
            pair_id = self.generate_pair_id(frame_path, product_path)
            if pair_id in self.labels:
                label = self.labels[pair_id]['label']
                if label == 1:
                    self.status_var.set("Эта пара уже помечена как ПОЛОЖИТЕЛЬНАЯ (товар есть на кадре)")
                else:
                    self.status_var.set("Эта пара уже помечена как ОТРИЦАТЕЛЬНАЯ (товара нет на кадре)")
            else:
                self.status_var.set("Ожидание разметки...")
            
            # Обновляем прогресс-бар
            total_pairs = self.calculate_total_pairs()
            processed_pairs = len(self.labels)
            progress = (processed_pairs / total_pairs) * 100 if total_pairs > 0 else 0
            self.progress_var.set(progress)
            
        except Exception as e:
            self.status_var.set(f"Ошибка при обновлении отображения: {e}")
    
    def resize_image(self, image, max_size):
        """Изменение размера изображения с сохранением пропорций"""
        width, height = image.size
        
        # Вычисляем соотношение сторон
        ratio = min(max_size[0] / width, max_size[1] / height)
        
        # Вычисляем новые размеры
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        # Изменяем размер
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def calculate_total_pairs(self):
        """Расчет общего количества пар для разметки"""
        total = 0
        for video in self.frames_info:
            num_frames = len(video.get('frames', []))
            num_products = len(video.get('products', {}))
            total += num_frames * num_products
        return total
    
    def mark_positive(self):
        """Пометить текущую пару как положительную"""
        if not self.frames_info:
            return
        
        try:
            # Получаем текущее видео
            current_video = self.frames_info[self.current_video_index]
            
            # Получаем текущий кадр
            if not current_video.get('frames'):
                return
            
            current_frame = current_video['frames'][self.current_frame_index]
            frame_path = current_frame['path']
            
            # Получаем текущий товар
            product_ids = list(current_video.get('products', {}).keys())
            if not product_ids:
                return
            
            current_product_id = product_ids[self.current_product_index]
            product_images = current_video['products'][current_product_id]
            
            if not product_images:
                return
            
            product_path = product_images[0]
            
            # Сохраняем метку
            pair_id = self.generate_pair_id(frame_path, product_path)
            self.labels[pair_id] = {
                'frame_path': os.path.normpath(frame_path),
                'product_path': os.path.normpath(product_path),
                'label': 1
            }
            
            # Сохраняем метки после каждой разметки
            self.save_labels()
            
            # Копируем пару в директорию положительных пар
            pair_dir = os.path.join(self.positive_pairs_dir, f"pair_{len(self.labels)}")
            os.makedirs(pair_dir, exist_ok=True)
            
            # Копируем изображения
            shutil.copy(frame_path, os.path.join(pair_dir, "frame.jpg"))
            shutil.copy(product_path, os.path.join(pair_dir, "product.jpg"))
            
            # Переходим к следующей паре
            self.next_pair()
            
        except Exception as e:
            self.status_var.set(f"Ошибка при разметке: {e}")
    
    def mark_negative(self):
        """Пометить текущую пару как отрицательную"""
        if not self.frames_info:
            return
        
        try:
            # Получаем текущее видео
            current_video = self.frames_info[self.current_video_index]
            
            # Получаем текущий кадр
            if not current_video.get('frames'):
                return
            
            current_frame = current_video['frames'][self.current_frame_index]
            frame_path = current_frame['path']
            
            # Получаем текущий товар
            product_ids = list(current_video.get('products', {}).keys())
            if not product_ids:
                return
            
            current_product_id = product_ids[self.current_product_index]
            product_images = current_video['products'][current_product_id]
            
            if not product_images:
                return
            
            product_path = product_images[0]
            
            # Сохраняем метку
            pair_id = self.generate_pair_id(frame_path, product_path)
            self.labels[pair_id] = {
                'frame_path': os.path.normpath(frame_path),
                'product_path': os.path.normpath(product_path),
                'label': 0
            }
            
            # Сохраняем метки после каждой разметки
            self.save_labels()
            
            # Переходим к следующей паре
            self.next_pair()
            
        except Exception as e:
            self.status_var.set(f"Ошибка при разметке: {e}")
    
    def skip_current(self):
        """Пропустить текущую пару"""
        self.next_pair()
    
    def next_pair(self):
        """Переход к следующей паре"""
        if not self.frames_info:
            return
        
        # Переходим к следующему товару
        current_video = self.frames_info[self.current_video_index]
        product_ids = list(current_video.get('products', {}).keys())
        
        if self.current_product_index < len(product_ids) - 1:
            self.current_product_index += 1
        else:
            # Если это последний товар, переходим к следующему кадру
            self.current_product_index = 0
            
            if self.current_frame_index < len(current_video.get('frames', [])) - 1:
                self.current_frame_index += 1
            else:
                # Если это последний кадр, переходим к следующему видео
                self.current_frame_index = 0
                
                if self.current_video_index < len(self.frames_info) - 1:
                    self.current_video_index += 1
                else:
                    # Если это последнее видео, начинаем сначала
                    self.current_video_index = 0
                    messagebox.showinfo("Информация", "Все видео просмотрены. Начинаем сначала.")
        
        # Обновляем отображение
        self.update_display()
    
    def prev_image(self):
        """Переход к предыдущему кадру"""
        if not self.frames_info:
            return
        
        current_video = self.frames_info[self.current_video_index]
        
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
        else:
            # Если это первый кадр, переходим к предыдущему видео
            if self.current_video_index > 0:
                self.current_video_index -= 1
                current_video = self.frames_info[self.current_video_index]
                self.current_frame_index = len(current_video.get('frames', [])) - 1
        
        # Сбрасываем индекс товара
        self.current_product_index = 0
        
        # Обновляем отображение
        self.update_display()
    
    def next_image(self):
        """Переход к следующему кадру"""
        if not self.frames_info:
            return
        
        current_video = self.frames_info[self.current_video_index]
        
        if self.current_frame_index < len(current_video.get('frames', [])) - 1:
            self.current_frame_index += 1
        else:
            # Если это последний кадр, переходим к следующему видео
            if self.current_video_index < len(self.frames_info) - 1:
                self.current_video_index += 1
                self.current_frame_index = 0
        
        # Сбрасываем индекс товара
        self.current_product_index = 0
        
        # Обновляем отображение
        self.update_display()
    
    def prev_product(self):
        """Переход к предыдущему товару"""
        if not self.frames_info:
            return
        
        current_video = self.frames_info[self.current_video_index]
        product_ids = list(current_video.get('products', {}).keys())
        
        if self.current_product_index > 0:
            self.current_product_index -= 1
        else:
            # Если это первый товар, оставляем как есть
            pass
        
        # Обновляем отображение
        self.update_display()
    
    def next_product(self):
        """Переход к следующему товару"""
        if not self.frames_info:
            return
        
        current_video = self.frames_info[self.current_video_index]
        product_ids = list(current_video.get('products', {}).keys())
        
        if self.current_product_index < len(product_ids) - 1:
            self.current_product_index += 1
        else:
            # Если это последний товар, оставляем как есть
            pass
        
        # Обновляем отображение
        self.update_display()
    
    def save_and_exit(self):
        """Сохранение меток и выход"""
        self.save_labels()
        
        # Генерируем данные для обучения
        pairs_file, training_data_file = self.generate_training_data()
        
        if pairs_file and training_data_file:
            messagebox.showinfo("Информация", f"Метки сохранены. Данные для обучения сгенерированы:\n- {pairs_file}\n- {training_data_file}")
        else:
            messagebox.showwarning("Предупреждение", "Не удалось сгенерировать данные для обучения")
        
        self.root.destroy()
    
    def run(self):
        """Запуск интерфейса"""
        # Обновляем отображение
        self.update_display()
        
        # Запускаем главный цикл
        self.root.mainloop()


class TrainingDataLoader:
    def __init__(self, output_dir, labels_json_path=None, training_pairs_json=None):
        """
        Класс для загрузки и подготовки данных для обучения
        
        Args:
            output_dir: Директория с результатами кураторства
            labels_json_path: Путь к файлу labels.json (приоритетнее, чем training_pairs_json)
            training_pairs_json: Путь к JSON-файлу с парами для обучения
        """
        self.output_dir = os.path.abspath(output_dir)
        
        # Если передан путь к labels.json, генерируем training_pairs.json
        if labels_json_path and os.path.exists(labels_json_path):
            print(f"Генерация training_pairs.json из labels.json...")
            training_pairs_path = os.path.join(output_dir, "training_pairs_generated.json")
            self._generate_pairs_from_labels(labels_json_path, training_pairs_path)
            self.training_pairs_json = training_pairs_path
        elif training_pairs_json:
            self.training_pairs_json = training_pairs_json
        else:
            # Ищем файл в директории
            default_pairs = os.path.join(output_dir, "training_pairs.json")
            if os.path.exists(default_pairs):
                self.training_pairs_json = default_pairs
            else:
                # Ищем в новом формате
                training_data_file = os.path.join(output_dir, "training_data.json")
                if os.path.exists(training_data_file):
                    self.training_pairs_json = training_data_file
                    self.is_new_format = True
                    print(f"Обнаружен файл данных в новом формате: {training_data_file}")
                else:
                    self.training_pairs_json = None
                    print("ПРЕДУПРЕЖДЕНИЕ: Не найден файл с парами для обучения!")
                    return
        
        # Загружаем данные о парах
        self.pairs_data = self._load_json(self.training_pairs_json)
        
        if self.pairs_data:
            # Проверяем формат данных
            if isinstance(self.pairs_data, list):
                # Новый формат - уже готовый список пар
                self.is_new_format = True
                positive_count = sum(1 for item in self.pairs_data if item.get('label') == 1)
                negative_count = sum(1 for item in self.pairs_data if item.get('label') == 0)
                print(f"Загружено {positive_count} положительных пар, {negative_count} отрицательных пар (новый формат)")
            else:
                # Старый формат с positive_pairs и negative_pairs
                self.is_new_format = False
                print(f"Загружено {len(self.pairs_data.get('positive_pairs', []))} положительных пар, "
                      f"{len(self.pairs_data.get('negative_pairs', []))} отрицательных пар (старый формат)")
    
    def _generate_pairs_from_labels(self, labels_file, output_file):
        """Генерирует файл training_pairs.json из labels.json"""
        try:
            # Загрузка файла с метками
            with open(labels_file, 'r', encoding='utf-8') as f:
                labels_data = json.load(f)
            
            # Проверяем формат данных
            if isinstance(labels_data, list):
                # Новый формат - просто копируем в training_data.json
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(labels_data, f, ensure_ascii=False, indent=2)
                
                positive_count = sum(1 for item in labels_data if item.get('label') == 1)
                negative_count = sum(1 for item in labels_data if item.get('label') == 0)
                
                print(f"Сгенерировано {positive_count} положительных и {negative_count} отрицательных пар (новый формат)")
                return True
            
            # Старый формат - обрабатываем каждую пару
            positive_pairs = []
            negative_pairs = []
            
            for pair_key, label in labels_data.items():
                # Разделяем ключ на путь к кадру и путь к товару
                try:
                    parts = pair_key.split('_')
                    
                    # Последняя часть - путь к изображению товара
                    product_path = parts[-1]
                    
                    # Все предыдущие части составляют путь к кадру
                    frame_path = '_'.join(parts[:-1])
                    
                    # Добавляем пару в соответствующий список
                    pair = {
                        "frame": frame_path,
                        "product": product_path
                    }
                    
                    if label == 1:
                        positive_pairs.append(pair)
                    else:
                        negative_pairs.append(pair)
                except Exception as e:
                    print(f"Ошибка при обработке пары {pair_key}: {e}")
            
            # Формируем итоговый JSON
            result = {
                "positive_pairs": positive_pairs,
                "negative_pairs": negative_pairs
            }
            
            # Сохраняем в файл
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"Сгенерировано {len(positive_pairs)} положительных и {len(negative_pairs)} отрицательных пар")
            
            return True
        
        except Exception as e:
            print(f"Ошибка при генерации training_pairs.json: {e}")
            return False
    
    def _load_json(self, path):
        """Загрузка JSON-файла"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Ошибка при загрузке JSON из {path}: {e}")
            return {}
    
    def prepare_training_data(self, output_json):
        """
        Подготовка данных для обучения в формате, пригодном для improved_training.py
        
        Args:
            output_json: Путь для сохранения JSON-файла
            
        Returns:
            str: Путь к сгенерированному JSON-файлу
        """
        try:
            if not self.pairs_data:
                print("Нет данных о парах для обучения!")
                return None
            
            # Формируем данные для обучения
            training_data = []
            valid_count = 0
            skipped_count = 0
            
            # Проверяем формат данных
            if self.is_new_format:
                # Новый формат - уже готовый список пар
                for item in self.pairs_data:
                    frame_path = item.get('frame_path')
                    product_path = item.get('product_path')
                    label = item.get('label')
                    
                    # Проверяем существование файлов
                    if os.path.exists(frame_path) and os.path.exists(product_path):
                        training_data.append({
                            'frame_path': os.path.normpath(frame_path),
                            'product_path': os.path.normpath(product_path),
                            'label': label
                        })
                        valid_count += 1
                    else:
                        skipped_count += 1
                        if not os.path.exists(frame_path):
                            print(f"Не найден файл кадра: {frame_path}")
                        if not os.path.exists(product_path):
                            print(f"Не найден файл товара: {product_path}")
            else:
                # Старый формат с positive_pairs и negative_pairs
                # Добавляем положительные пары
                for pair in self.pairs_data.get('positive_pairs', []):
                    frame_path = pair.get('frame')
                    product_path = pair.get('product')
                    
                    # Проверяем существование файлов
                    if os.path.exists(frame_path) and os.path.exists(product_path):
                        training_data.append({
                            'frame_path': os.path.normpath(frame_path),
                            'product_path': os.path.normpath(product_path),
                            'label': 1
                        })
                        valid_count += 1
                    else:
                        skipped_count += 1
                        if not os.path.exists(frame_path):
                            print(f"Не найден файл кадра: {frame_path}")
                        if not os.path.exists(product_path):
                            print(f"Не найден файл товара: {product_path}")
                
                # Добавляем отрицательные пары
                for pair in self.pairs_data.get('negative_pairs', []):
                    frame_path = pair.get('frame')
                    product_path = pair.get('product')
                    
                    # Проверяем существование файлов
                    if os.path.exists(frame_path) and os.path.exists(product_path):
                        training_data.append({
                            'frame_path': os.path.normpath(frame_path),
                            'product_path': os.path.normpath(product_path),
                            'label': 0
                        })
                        valid_count += 1
                    else:
                        skipped_count += 1
            
            # Перемешиваем данные
            random.shuffle(training_data)
            
            # Сохраняем данные в JSON
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, ensure_ascii=False, indent=2)
            
            print(f"Подготовлены данные для обучения: {valid_count} валидных пар, {skipped_count} пропущено.")
            print(f"Данные сохранены в {output_json}")
            return output_json
            
        except Exception as e:
            print(f"Ошибка при подготовке данных для обучения: {e}")
            return None
        
def main():
    parser = argparse.ArgumentParser(description="Куратор данных для обучения модели детектирования товаров")
    parser.add_argument("--data_root", type=str, required=True, help="Корневая директория с данными")
    parser.add_argument("--videos_json", type=str, required=True, help="Путь к JSON-файлу с видео")
    parser.add_argument("--output_dir", type=str, default="curated_data", help="Директория для сохранения результатов")
    parser.add_argument("--num_frames", type=int, default=10, help="Количество кадров для извлечения из каждого видео")
    parser.add_argument("--max_videos", type=int, default=None, help="Максимальное количество видео для обработки")
    parser.add_argument("--num_workers", type=int, default=4, help="Количество параллельных потоков")
    parser.add_argument("--mode", type=str, choices=["process", "curate", "both"], default="both", 
                       help="Режим работы: process - только обработка, curate - только разметка, both - обработка и разметка")
    parser.add_argument("--frames_info", type=str, default=None, help="Путь к файлу с информацией о кадрах (для режима curate)")
    parser.add_argument("--generate_training_data", action="store_true", help="Сгенерировать данные для обучения")
    parser.add_argument("--training_data_output", type=str, default="training_data.json", help="Путь для сохранения данных для обучения")
    parser.add_argument("--gpu", action="store_true", help="Использовать GPU для обработки")
    
    args = parser.parse_args()
    
    # Проверяем наличие файлов
    if not os.path.exists(args.data_root):
        print(f"Ошибка: Директория {args.data_root} не существует")
        return
    
    if not os.path.exists(args.videos_json):
        print(f"Ошибка: Файл {args.videos_json} не существует")
        return
    
    # Создаем куратора
    curator = DataCurator(
        data_root=args.data_root,
        videos_json=args.videos_json,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        gpu=args.gpu
    )
    
    # Режим обработки
    if args.mode in ["process", "both"]:
        # Обрабатываем видео
        frames_info = curator.process_all_videos(
            max_videos=args.max_videos,
            num_workers=args.num_workers
        )
    
    # Режим ручной разметки
    if args.mode in ["curate", "both"]:
        # Загружаем информацию о кадрах
        if args.frames_info:
            with open(args.frames_info, 'r', encoding='utf-8') as f:
                frames_info = json.load(f)
        else:
            frames_info_path = os.path.join(args.output_dir, "frames_info.json")
            if os.path.exists(frames_info_path):
                with open(frames_info_path, 'r', encoding='utf-8') as f:
                    frames_info = json.load(f)
            else:
                print(f"Ошибка: Файл {frames_info_path} не существует")
                return
        
        # Запускаем ручную разметку
        curator.run_manual_curation(frames_info)
    
    # Генерация данных для обучения
    if args.generate_training_data:
        # Загружаем данные о парах
        labels_json_path = os.path.join(args.output_dir, "labels", "labels.json")
        if os.path.exists(labels_json_path):
            # Создаем загрузчик данных
            loader = TrainingDataLoader(args.output_dir, labels_json_path=labels_json_path)
            
            # Подготавливаем данные для обучения
            loader.prepare_training_data(args.training_data_output)
        else:
            print(f"Ошибка: Файл {labels_json_path} не существует")

if __name__ == "__main__":
    main()