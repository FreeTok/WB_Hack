import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageFile
import cv2
import numpy as np
import time
from tqdm import tqdm
import logging
import argparse
from pathlib import Path
from collections import defaultdict
import shutil

# Для обработки поврежденных изображений
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Импортируем модель из файла обучения
from improved_training import ProductMatcher

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"inference_{time.strftime('%Y%m%d-%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ProductDetector")

class ProductDetector:
    def __init__(self, model_path, threshold=0.5, device=None, cache_dir="cache"):
        """
        Инициализация детектора товаров на видео
        
        Args:
            model_path: Путь к обученной модели
            threshold: Порог сходства для определения совпадения (0.0-1.0)
            device: Устройство для инференса ('cuda' или 'cpu')
            cache_dir: Директория для кэширования
        """
        self.threshold = threshold
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Отдельная папка для кадров
        self.frames_dir = os.path.join(self.cache_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        
        # Папка для эмбеддингов
        self.embeddings_dir = os.path.join(self.cache_dir, "embeddings")
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        # Загрузка модели
        logger.info(f"Загрузка модели из {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Определение параметров модели из checkpoint
            backbone = checkpoint.get('backbone', 'efficientnet_b0')
            embedding_dim = checkpoint.get('embedding_dim', 512)
            
            logger.info(f"Backbone: {backbone}, Embedding dim: {embedding_dim}")
            
            # Создание модели
            self.model = ProductMatcher(
                backbone_name=backbone,
                embedding_dim=embedding_dim
            ).to(self.device)
            
            # Загрузка весов
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            logger.info("Модель загружена успешно")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise
        
        # Трансформации для изображений
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_embedding(self, image_path):
        """
        Получение эмбеддинга для изображения
        
        Args:
            image_path: Путь к изображению или PIL Image или numpy array
            
        Returns:
            embedding: numpy array
        """
        try:
            # Проверяем кэш эмбеддингов
            if isinstance(image_path, str) and os.path.exists(image_path):
                embed_cache_path = os.path.join(
                    self.embeddings_dir, 
                    f"embed_{abs(hash(image_path)) % 10000}.npy"
                )
                
                if os.path.exists(embed_cache_path):
                    return np.load(embed_cache_path)
            
            # Загружаем изображение
            if isinstance(image_path, str):
                image = Image.open(image_path).convert('RGB')
            elif isinstance(image_path, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))
            elif isinstance(image_path, Image.Image):
                image = image_path.convert('RGB')
            else:
                raise ValueError("Неподдерживаемый тип изображения")
            
            # Применяем трансформации
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Получаем эмбеддинг
            with torch.no_grad():
                embedding = self.model.forward_one(image_tensor)
            
            # Преобразуем в numpy для дальнейшей работы
            embedding_np = embedding.cpu().numpy()[0]
            
            # Кэшируем эмбеддинг
            if isinstance(image_path, str) and os.path.exists(image_path):
                np.save(embed_cache_path, embedding_np)
            
            return embedding_np
        
        except Exception as e:
            logger.error(f"Ошибка при получении эмбеддинга: {e}")
            return None
    
    def extract_frames(self, video_path, fps=1):
        """
        Извлечение кадров из видео с заданной частотой
        Использует OpenCV вместо ffmpeg для более надежной работы
        
        Args:
            video_path: Путь к видео
            fps: Частота извлечения кадров (кадров в секунду)
            
        Returns:
            frames: Список путей к сохраненным кадрам
            timestamps: Список временных меток для каждого кадра
        """
        try:
            # Проверяем существование видео
            if not os.path.exists(video_path):
                logger.error(f"Видео не найдено: {video_path}")
                return [], []
            
            # Создаем уникальную директорию для кадров этого видео
            video_id = os.path.basename(video_path).split('.')[0]
            frames_dir = os.path.join(self.frames_dir, f"{video_id}")
            
            # Проверяем, если кадры уже извлечены
            if os.path.exists(frames_dir) and len(os.listdir(frames_dir)) > 0:
                frames = sorted([
                    os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
                    if f.endswith('.jpg')
                ])
                
                # Извлекаем временные метки из имен файлов
                timestamps = []
                for frame in frames:
                    try:
                        time_str = os.path.basename(frame).split('_')[1].split('.')[0]
                        timestamps.append(float(time_str))
                    except:
                        timestamps.append(0.0)
                
                logger.info(f"Найдено {len(frames)} кэшированных кадров для видео {video_id}")
                return frames, timestamps
            
            # Если кадров еще нет, создаем директорию и извлекаем их
            os.makedirs(frames_dir, exist_ok=True)
            
            # Открываем видео
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Не удалось открыть видео: {video_path}")
                return [], []
            
            # Получаем информацию о видео
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0:
                video_fps = 25  # Стандартное значение, если не удалось определить
            
            # Вычисляем шаг в кадрах
            frame_step = int(video_fps / fps)
            if frame_step <= 0:
                frame_step = 1
            
            frames = []
            timestamps = []
            
            # Извлекаем кадры
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_step == 0:
                    # Вычисляем временную метку
                    time_sec = frame_idx / video_fps
                    
                    # Сохраняем кадр
                    frame_path = os.path.join(frames_dir, f"frame_{time_sec:.2f}.jpg")
                    cv2.imwrite(frame_path, frame)
                    
                    frames.append(frame_path)
                    timestamps.append(time_sec)
                
                frame_idx += 1
            
            # Освобождаем ресурсы
            cap.release()
            
            logger.info(f"Извлечено {len(frames)} кадров из видео {video_id}")
            return frames, timestamps
        
        except Exception as e:
            logger.error(f"Ошибка при извлечении кадров: {e}")
            return [], []
    
    def find_product_in_video(self, product_images, video_path):
        """
        Поиск товара на видео
        
        Args:
            product_images: Список путей к изображениям товара
            video_path: Путь к видео
            
        Returns:
            matches: Список словарей с информацией о совпадениях
        """
        try:
            start_time = time.time()
            
            # Извлекаем кадры из видео
            frames, timestamps = self.extract_frames(video_path)
            
            if not frames:
                logger.warning(f"Не удалось извлечь кадры из видео {video_path}")
                return []
            
            # Получаем эмбеддинги для всех изображений товара
            product_embeddings = []
            valid_product_images = []
            
            for img_path in product_images:
                if not os.path.exists(img_path):
                    logger.warning(f"Изображение товара не найдено: {img_path}")
                    continue
                
                embedding = self.get_embedding(img_path)
                if embedding is not None:
                    product_embeddings.append(embedding)
                    valid_product_images.append(img_path)
            
            if not product_embeddings:
                logger.warning(f"Не удалось получить эмбеддинги для товара")
                return []
            
            # Ищем совпадения с каждым кадром видео
            matches = []
            
            for frame_path, timestamp in tqdm(zip(frames, timestamps), desc="Поиск товара", total=len(frames)):
                # Получаем эмбеддинг для кадра
                frame_embedding = self.get_embedding(frame_path)
                if frame_embedding is None:
                    continue
                
                # Проверяем сходство с каждым изображением товара
                for i, product_emb in enumerate(product_embeddings):
                    # Косинусное сходство
                    similarity = np.dot(frame_embedding, product_emb)
                    
                    if similarity >= self.threshold:
                        matches.append({
                            "timestamp": timestamp,
                            "similarity": float(similarity),
                            "frame_path": frame_path,
                            "product_path": valid_product_images[i]
                        })
                        # Если нашли совпадение для этого кадра, переходим к следующему кадру
                        break
            
            # Сортируем совпадения по времени
            matches.sort(key=lambda x: x["timestamp"])
            
            # Группируем близкие таймкоды
            if matches:
                grouped_matches = []
                current_group = [matches[0]]
                
                for match in matches[1:]:
                    if match["timestamp"] - current_group[-1]["timestamp"] <= 1.0:
                        current_group.append(match)
                    else:
                        # Берем лучшее совпадение из группы
                        best_match = max(current_group, key=lambda x: x["similarity"])
                        grouped_matches.append(best_match)
                        current_group = [match]
                
                # Добавляем последнюю группу
                if current_group:
                    best_match = max(current_group, key=lambda x: x["similarity"])
                    grouped_matches.append(best_match)
                
                logger.info(f"Найдено {len(grouped_matches)} совпадений (после группировки)")
                
                # Вычисляем затраченное время
                elapsed_time = time.time() - start_time
                logger.info(f"Поиск товара занял {elapsed_time:.2f} секунд")
                
                return grouped_matches
            else:
                logger.info("Товар не найден в видео")
                return []
        
        except Exception as e:
            logger.error(f"Ошибка при поиске товара: {e}")
            return []
    
    def process_video_by_nm_id(self, nm_id, video_path, data_root):
        """
        Обработка видео и поиск товара по его nm_id
        
        Args:
            nm_id: ID товара
            video_path: Путь к видео
            data_root: Корневая директория с данными
            
        Returns:
            result: Результаты обнаружения
        """
        try:
            # Получаем пути к изображениям товара
            product_images = self._get_product_image_paths(nm_id, data_root)
            
            if not product_images:
                logger.warning(f"Не найдено изображений для товара {nm_id}")
                return {
                    'nm_id': nm_id,
                    'video_path': video_path,
                    'found': False,
                    'timestamps': [],
                    'error': "Не найдено изображений товара"
                }
            
            # Ищем товар на видео
            matches = self.find_product_in_video(product_images, video_path)
            
            # Формируем результат
            return {
                'nm_id': nm_id,
                'video_path': video_path,
                'found': len(matches) > 0,
                'timestamps': [match["timestamp"] for match in matches],
                'similarities': [match["similarity"] for match in matches] if matches else [],
                'matches': matches
            }
        
        except Exception as e:
            logger.error(f"Ошибка при обработке товара {nm_id}: {e}")
            return {
                'nm_id': nm_id,
                'video_path': video_path,
                'found': False,
                'timestamps': [],
                'error': str(e)
            }
    
    def _get_product_image_paths(self, nm_id, data_root):
        """Получение путей к изображениям товара"""
        image_paths = []
        
        # Проверяем наличие основного изображения
        img1_path = os.path.join(data_root, f"images_1/{nm_id}/1.webp")
        if os.path.exists(img1_path):
            image_paths.append(img1_path)
        
        # Проверяем наличие дополнительных изображений
        for img_idx in [2, 3]:
            img_path = os.path.join(data_root, f"images_2-3/{nm_id}/{img_idx}.webp")
            if os.path.exists(img_path):
                image_paths.append(img_path)
        
        # Проверяем наличие дополнительных изображений 4-5
        for img_idx in [4, 5]:
            img_path = os.path.join(data_root, f"images_4-5/{nm_id}/{img_idx}.webp")
            if os.path.exists(img_path):
                image_paths.append(img_path)
        
        return image_paths
    
    def download_video(self, url, output_path):
        """
        Загрузка видео по URL
        Используем OpenCV для более надежной загрузки, если это возможно
        
        Args:
            url: URL видео
            output_path: Путь для сохранения
            
        Returns:
            success: Успешность загрузки
        """
        try:
            # Проверяем, существует ли уже загруженный файл
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                if file_size > 1024:  # Файл больше 1KB, считаем его корректным
                    logger.info(f"Используется существующее видео: {output_path}")
                    return True
                else:
                    # Файл слишком мал, удаляем его
                    os.remove(output_path)
            
            # Создаем директорию, если не существует
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Для m3u8 и других потоковых форматов используем ffmpeg
            if "m3u8" in url or "https://" in url:
                import subprocess
                logger.info(f"Загрузка видео через ffmpeg: {url}")
                
                command = [
                    'ffmpeg',
                    '-i', url,
                    '-c', 'copy',
                    '-v', 'error',
                    output_path
                ]
                
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                
                if process.returncode != 0:
                    logger.error(f"Ошибка ffmpeg: {stderr.decode()}")
                    return False
                
                return os.path.exists(output_path)
            
            # Для локальных файлов просто копируем
            elif os.path.exists(url):
                logger.info(f"Копирование локального файла: {url}")
                shutil.copy(url, output_path)
                return True
            
            # Прямая загрузка через OpenCV
            else:
                logger.info(f"Попытка загрузки через OpenCV: {url}")
                cap = cv2.VideoCapture(url)
                if not cap.isOpened():
                    logger.error(f"Не удалось открыть видео: {url}")
                    return False
                
                # Создаем VideoWriter
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                # Читаем и записываем кадры
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    writer.write(frame)
                
                # Освобождаем ресурсы
                cap.release()
                writer.release()
                
                return os.path.exists(output_path)
        
        except Exception as e:
            logger.error(f"Ошибка при загрузке видео: {e}")
            return False
    
    def process_from_json(self, videos_json, data_root, temp_dir="temp", result_file="results.json", max_videos=None):
        """
        Обработка видео из JSON-файла с опцией ограничения количества видео
        
        Args:
            videos_json: Путь к JSON-файлу с данными о видео
            data_root: Корневая директория с данными
            temp_dir: Директория для временных файлов
            result_file: Файл для сохранения результатов
            max_videos: Максимальное количество видео для обработки (None = все)
        """
        # Создаем директорию для временных файлов
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Загружаем данные о видео
            with open(videos_json, 'r', encoding='utf-8') as f:
                videos_data = json.load(f)
            
            # Ограничиваем количество видео
            if max_videos is not None and max_videos > 0:
                videos_data = videos_data[:max_videos]
                logger.info(f"Ограничение обработки до первых {max_videos} видео")
            
            logger.info(f"Будет обработано {len(videos_data)} видео из JSON")
            
            # Результаты для каждого видео
            results = []
            
            # Счетчики для статистики
            total_found = 0
            total_not_found = 0
            
            # Обрабатываем каждое видео
            for i, video in enumerate(videos_data):
                video_id = video.get('video_id')
                video_url = video.get('path_url')
                nm_ids = video.get('nm_ids', [])
                
                logger.info(f"Обработка видео {i+1}/{len(videos_data)}: {video_id}")
                
                if not nm_ids:
                    logger.warning(f"Видео {video_id} не имеет связанных товаров")
                    results.append({
                        'video_id': video_id,
                        'detections': {},
                        'error': "Нет связанных товаров"
                    })
                    continue
                
                # Загружаем видео
                temp_video_path = os.path.join(temp_dir, f"video_{video_id}.mp4")
                
                # Пропускаем, если видео уже загружено
                if not os.path.exists(temp_video_path) or os.path.getsize(temp_video_path) < 1024:
                    if not self.download_video(video_url, temp_video_path):
                        logger.error(f"Не удалось загрузить видео {video_id}")
                        results.append({
                            'video_id': video_id,
                            'detections': {},
                            'error': "Ошибка загрузки видео"
                        })
                        continue
                
                # Создаем результат для этого видео
                video_result = {
                    'video_id': video_id,
                    'detections': {},
                    'found_items': [],
                    'not_found_items': []
                }
                
                # Обрабатываем каждый товар
                for nm_id in nm_ids:
                    detection = self.process_video_by_nm_id(nm_id, temp_video_path, data_root)
                    
                    # Сохраняем результат
                    video_result['detections'][str(nm_id)] = detection
                    
                    # Обновляем счетчики и списки
                    if detection.get('found', False):
                        video_result['found_items'].append(str(nm_id))
                        total_found += 1
                    else:
                        video_result['not_found_items'].append(str(nm_id))
                        total_not_found += 1
                
                # Добавляем результат в общий список
                results.append(video_result)
                
                # Сохраняем промежуточные результаты каждые 5 видео
                if (i + 1) % 5 == 0 or (i + 1) == len(videos_data):
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            'videos': results,
                            'stats': {
                                'total_videos': len(results),
                                'total_found': total_found,
                                'total_not_found': total_not_found,
                                'accuracy': total_found / (total_found + total_not_found) if (total_found + total_not_found) > 0 else 0
                            }
                        }, f, ensure_ascii=False, indent=2)
                    
                    logger.info(f"Обработано {i+1}/{len(videos_data)} видео")
                    logger.info(f"Найдено: {total_found}, не найдено: {total_not_found}")
            
            # Удаляем временные видео для экономии места
            for video in videos_data:
                video_id = video.get('video_id')
                temp_video_path = os.path.join(temp_dir, f"video_{video_id}.mp4")
                
                if os.path.exists(temp_video_path):
                    try:
                        os.remove(temp_video_path)
                    except:
                        pass
            
            logger.info(f"Обработка завершена. Результаты сохранены в {result_file}")
            
            # Возвращаем финальные результаты
            return {
                'videos': results,
                'stats': {
                    'total_videos': len(results),
                    'total_found': total_found,
                    'total_not_found': total_not_found,
                    'accuracy': total_found / (total_found + total_not_found) if (total_found + total_not_found) > 0 else 0
                }
            }
        
        except Exception as e:
            logger.error(f"Ошибка при обработке JSON: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def evaluate_model(self, videos_json, data_root, output_file="evaluation.json"):
        """
        Оценка качества модели на тестовом наборе данных
        
        Args:
            videos_json: Путь к JSON-файлу с данными о видео (где nm_ids - истинные товары)
            data_root: Корневая директория с данными
            output_file: Файл для сохранения результатов оценки
            
        Returns:
            metrics: Метрики качества модели
        """
        try:
            # Загружаем данные
            with open(videos_json, 'r', encoding='utf-8') as f:
                videos_data = json.load(f)
            
            logger.info(f"Оценка модели на {len(videos_data)} видео")
            
            # Счетчики для метрик
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            
            # Обрабатываем все видео
            results = self.process_from_json(videos_json, data_root)
            
            # Подсчитываем метрики для каждого видео
            for video_result in results.get('videos', []):
                found_items = set(video_result.get('found_items', []))
                not_found_items = set(video_result.get('not_found_items', []))
                
                # Получаем истинные nm_ids для этого видео
                video_id = video_result.get('video_id')
                original_video = next((v for v in videos_data if v.get('video_id') == video_id), None)
                
                if original_video:
                    true_items = set(str(item) for item in original_video.get('nm_ids', []))
                    
                    # Обновляем счетчики
                    for item in found_items:
                        if item in true_items:
                            true_positives += 1
                        else:
                            false_positives += 1
                    
                    for item in true_items:
                        if item not in found_items:
                            false_negatives += 1
            
            # Вычисляем метрики
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Формируем результат
            metrics = {
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            # Сохраняем результаты
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Оценка завершена. Метрики: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Ошибка при оценке модели: {e}")
            return {'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description="Детектирование товаров на видео")
    parser.add_argument("--model", type=str, required=True, help="Путь к обученной модели")
    parser.add_argument("--data", type=str, required=True, help="Путь к корневой директории с данными")
    parser.add_argument("--videos_json", type=str, required=True, help="Путь к JSON-файлу с видео")
    parser.add_argument("--output", type=str, default="detection_results.json", help="Файл для сохранения результатов")
    parser.add_argument("--threshold", type=float, default=0.5, help="Порог сходства (0.0-1.0)")
    parser.add_argument("--max_videos", type=int, default=None, help="Максимальное количество видео для обработки")
    parser.add_argument("--mode", type=str, default="process", choices=["process", "evaluate"], 
                       help="Режим работы: process - обработка видео, evaluate - оценка качества модели")
    parser.add_argument("--clear_cache", action="store_true", help="Очистить кэш перед запуском")
    
    args = parser.parse_args()
    
    # Проверяем наличие файлов
    if not os.path.exists(args.model):
        logger.error(f"Модель не найдена: {args.model}")
        return
    
    if not os.path.exists(args.videos_json):
        logger.error(f"JSON-файл с видео не найден: {args.videos_json}")
        return
    
    if not os.path.exists(args.data):
        logger.error(f"Директория с данными не найдена: {args.data}")
        return
    
    # Инициализируем детектор
    detector = ProductDetector(
        model_path=args.model,
        threshold=args.threshold,
        cache_dir="cache"
    )
    
    # Очищаем кэш, если требуется
    if args.clear_cache:
        logger.info("Очистка кэша...")
        shutil.rmtree("cache", ignore_errors=True)
        os.makedirs("cache", exist_ok=True)
    
    # Запускаем обработку или оценку
    if args.mode == "process":
        results = detector.process_from_json(
            videos_json=args.videos_json,
            data_root=args.data,
            temp_dir="temp",
            result_file=args.output,
            max_videos=args.max_videos
        )
        
        # Выводим короткую сводку
        print("\nРезультаты обработки:")
        print(f"Всего видео: {results.get('stats', {}).get('total_videos', 0)}")
        print(f"Найдено товаров: {results.get('stats', {}).get('total_found', 0)}")
        print(f"Не найдено товаров: {results.get('stats', {}).get('total_not_found', 0)}")
        print(f"Точность: {results.get('stats', {}).get('accuracy', 0)*100:.2f}%")
        
    elif args.mode == "evaluate":
        metrics = detector.evaluate_model(
            videos_json=args.videos_json,
            data_root=args.data,
            output_file=args.output
        )
        
        # Выводим метрики
        print("\nМетрики качества модели:")
        print(f"Precision: {metrics.get('precision', 0)*100:.2f}%")
        print(f"Recall: {metrics.get('recall', 0)*100:.2f}%")
        print(f"F1 Score: {metrics.get('f1_score', 0)*100:.2f}%")

if __name__ == "__main__":
    main()