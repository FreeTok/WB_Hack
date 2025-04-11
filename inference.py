import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import subprocess
import os
import json
from pathlib import Path
import logging
from tqdm import tqdm
import time

# Импортируем класс нашей модели из обучающего скрипта
from learning import ProductMatcher

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ProductDetector")

class ProductDetector:
    def __init__(self, model_path, threshold=0.75, device=None):
        """
        Инициализация детектора товаров на видео
        
        Args:
            model_path: Путь к обученной модели
            threshold: Порог сходства для определения совпадения
            device: Устройство для инференса ('cuda' или 'cpu')
        """
        self.threshold = threshold
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Загрузка модели
        logger.info(f"Загрузка модели из {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Определение параметров модели из checkpoint
        backbone = checkpoint.get('backbone', 'efficientnet_b0')
        embedding_dim = checkpoint.get('embedding_dim', 512)
        
        # Создание модели
        self.model = ProductMatcher(
            backbone=backbone,
            embedding_dim=embedding_dim,
            pretrained=False
        ).to(self.device)
        
        # Загрузка весов
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Трансформации для изображений
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("Модель загружена успешно")
    
    def get_embedding(self, image):
        """Получение эмбеддинга для изображения"""
        if isinstance(image, str):
            # Если передан путь к изображению
            image = Image.open(image).convert('RGB')
        
        if isinstance(image, np.ndarray):
            # Если передан numpy массив (например, кадр из OpenCV)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Применяем трансформации
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Получаем эмбеддинг
        with torch.no_grad():
            embedding = self.model.forward_one(image_tensor)
        
        return embedding.cpu().numpy()[0]
    
    def _extract_frames(self, video_path, frame_rate=1):
        """Извлечение кадров из видео с заданной частотой"""
        # Проверка существования видео
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Видео не найдено: {video_path}")
        
        # Открываем видео
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
        
        # Получаем информацию о видео
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        logger.info(f"Обработка видео: {video_path}")
        logger.info(f"FPS: {fps}, Длительность: {duration:.2f} сек")
        
        # Шаг в кадрах для выборки
        frame_step = max(1, int(fps * frame_rate))
        
        # Извлекаем кадры
        frames = []
        timestamps = []
        
        for frame_idx in tqdm(range(0, total_frames, frame_step), desc="Извлечение кадров"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Сохраняем кадр и его таймкод
            frames.append(frame)
            timestamps.append(frame_idx / fps)
        
        cap.release()
        
        return frames, timestamps
    
    def _download_m3u8_video(self, m3u8_url, output_path):
        """Загрузка видео из m3u8-ссылки"""
        try:
            # Убедимся, что директория существует
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Команда для ffmpeg
            command = [
                'ffmpeg',
                '-i', m3u8_url,
                '-c', 'copy',
                '-v', 'error',
                output_path
            ]
            
            # Запускаем ffmpeg
            logger.info(f"Загрузка видео из {m3u8_url}")
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Ошибка загрузки видео: {result.stderr}")
                return None
            
            return output_path
        except Exception as e:
            logger.error(f"Исключение при загрузке видео: {e}")
            return None
    
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
    
    def find_product_in_video(self, product_images, video_path, frame_rate=1):
        """
        Поиск товара на видео
        
        Args:
            product_images: Список путей к изображениям товара или PIL изображения
            video_path: Путь к видео
            frame_rate: Частота выборки кадров (в секундах)
            
        Returns:
            matches: Список таймкодов с обнаружениями
        """
        # Получаем эмбеддинги для всех изображений товара
        product_embeddings = []
        for img in product_images:
            embedding = self.get_embedding(img)
            product_embeddings.append(embedding)
        
        # Извлекаем кадры из видео
        frames, timestamps = self._extract_frames(video_path, frame_rate)
        
        # Ищем совпадения
        matches = []
        
        for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            # Получаем эмбеддинг для кадра
            frame_embedding = self.get_embedding(frame)
            
            # Проверяем сходство с изображениями товара
            for product_emb in product_embeddings:
                similarity = np.dot(frame_embedding, product_emb)
                
                if similarity >= self.threshold:
                    matches.append(timestamp)
                    break  # Если обнаружен хотя бы один товар, переходим к следующему кадру
        
        # Группируем близкие таймкоды
        if matches:
            grouped_matches = []
            current_group = [matches[0]]
            
            for t in matches[1:]:
                if t - current_group[-1] <= 1.0:
                    current_group.append(t)
                else:
                    # Берем среднее время для группы
                    grouped_matches.append(sum(current_group) / len(current_group))
                    current_group = [t]
            
            # Добавляем последнюю группу
            if current_group:
                grouped_matches.append(sum(current_group) / len(current_group))
                
            return grouped_matches
        
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
        # Получаем пути к изображениям товара
        product_images = self._get_product_image_paths(nm_id, data_root)
        
        if not product_images:
            logger.warning(f"Не найдено изображений для товара {nm_id}")
            return None
        
        # Ищем товар на видео
        matches = self.find_product_in_video(product_images, video_path)
        
        return {
            'nm_id': nm_id,
            'video_path': video_path,
            'timestamps': matches,
            'found': len(matches) > 0
        }
    
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
        
        # Загружаем данные о видео
        with open(videos_json, 'r', encoding='utf-8') as f:
            videos_data = json.load(f)
        
        # Ограничиваем количество видео, если указано
        if max_videos is not None and max_videos > 0:
            videos_data = videos_data[:max_videos]
            logger.info(f"Ограничение обработки до первых {max_videos} видео")
        
        logger.info(f"Будет обработано {len(videos_data)} видео из JSON")
        
        results = []
        
        for i, video in enumerate(videos_data):
            video_id = video['video_id']
            video_url = video['path_url']
            nm_ids = video.get('nm_ids', [])
            
            if not nm_ids:
                logger.warning(f"Видео {video_id} не имеет связанных товаров")
                continue
            
            # Загружаем видео
            temp_video_path = os.path.join(temp_dir, f"video_{video_id}.mp4")
            
            if not os.path.exists(temp_video_path):
                downloaded_path = self._download_m3u8_video(video_url, temp_video_path)
                if not downloaded_path:
                    logger.error(f"Не удалось загрузить видео {video_id}")
                    continue
            
            # Обрабатываем каждый товар
            video_results = {
                'video_id': video_id,
                'detections': {}
            }
            
            for nm_id in nm_ids:
                result = self.process_video_by_nm_id(nm_id, temp_video_path, data_root)
                if result:
                    video_results['detections'][str(nm_id)] = result
            
            results.append(video_results)
            
            # Удаляем временное видео
            try:
                os.remove(temp_video_path)
            except:
                pass
            
            # Сохраняем промежуточные результаты
            if (i + 1) % 10 == 0 or (i + 1) == len(videos_data):
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Обработано {i+1}/{len(videos_data)} видео")
        
        # Сохраняем финальные результаты
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Обработка завершена. Результаты сохранены в {result_file}")
        
        return results

# Пример использования
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Детектирование товаров на видео")
    parser.add_argument("--model", type=str, required=True, help="Путь к обученной модели")
    parser.add_argument("--data", type=str, required=True, help="Путь к корневой директории с данными")
    parser.add_argument("--videos_json", type=str, required=True, help="Путь к JSON-файлу с видео")
    parser.add_argument("--output", type=str, default="detection_results.json", help="Файл для сохранения результатов")
    parser.add_argument("--threshold", type=float, default=0.75, help="Порог сходства (0.0-1.0)")
    parser.add_argument("--max_videos", type=int, default=None, help="Максимальное количество видео для обработки")
    
    args = parser.parse_args()
    
    # Создаем детектор
    detector = ProductDetector(
        model_path=args.model,
        threshold=args.threshold
    )
    
    # Запускаем обработку
    detector.process_from_json(
        videos_json=args.videos_json,
        data_root=args.data,
        temp_dir="temp",
        result_file=args.output,
        max_videos=args.max_videos
    )