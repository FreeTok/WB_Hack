import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image, ImageFile
import cv2
import logging
from pathlib import Path
import time
from tqdm import tqdm

# Чтобы избежать проблем с поврежденными изображениями
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ProductDataset")

class ProductVideoDataset(Dataset):
    def __init__(self, data_root, videos_json, transform=None, max_pairs=5000, frame_interval=1.0):
        """
        Датасет для пар товар-видео для обучения модели сопоставления
        
        Args:
            data_root: Корневая директория с данными
            videos_json: Путь к JSON-файлу с видео и товарами
            transform: Трансформации изображений
            max_pairs: Максимальное количество пар для обучения
            frame_interval: Интервал между извлекаемыми кадрами в секундах
        """
        self.data_root = data_root
        self.frame_interval = frame_interval
        self.transform = transform if transform else transforms.ToTensor()
        
        # Создаем временные директории
        self.temp_dir = os.path.join(data_root, "temp_frames")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Загружаем данные о видео
        self.videos_data = self._load_json(videos_json)
        
        # Создаем списки для модели обучения
        self.pairs = []
        self._prepare_pairs(max_pairs)
        
        logger.info(f"Создано {len(self.pairs)} пар объектов для обучения")
    
    def _load_json(self, path):
        """Загрузка JSON-файла"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Ошибка при загрузке JSON из {path}: {e}")
            return []
    
    def _prepare_pairs(self, max_pairs):
        """Подготовка пар товар-видео для обучения"""
        count = 0
        
        # Перемешиваем данные для более равномерной выборки
        random.shuffle(self.videos_data)
        
        for video in self.videos_data:
            # Пропускаем видео без товаров
            if not video.get('nm_ids'):
                continue
            
            # Собираем изображения товаров для этого видео
            products_info = {}
            for nm_id in video.get('nm_ids', []):
                product_images = self._get_product_image_paths(nm_id)
                if product_images:
                    products_info[nm_id] = product_images
            
            # Пропускаем видео без найденных изображений товаров
            if not products_info:
                continue
            
            # Предварительно извлекаем кадры из видео
            video_frames = self._extract_frames_from_mp4(video.get('path_url'))
            if not video_frames:
                continue
            
            # Создаем положительные пары (товар, кадр из видео)
            for nm_id, img_paths in products_info.items():
                for img_path in img_paths[:2]:  # Берем не более 2 изображений каждого товара
                    # Берем случайный кадр из видео для положительного примера
                    if video_frames:
                        frame_path = random.choice(video_frames)
                        
                        # Добавляем положительную пару
                        self.pairs.append({
                            'product_img': img_path,
                            'frame_img': frame_path,
                            'label': 1,  # 1 = товар есть на видео
                            'product_id': nm_id,
                            'video_id': video.get('video_id')
                        })
                        
                        count += 1
                        
                        # Создаем отрицательную пару, где товар точно отсутствует на видео
                        # Берем случайное видео без этого товара
                        negative_videos = [v for v in self.videos_data 
                                       if v.get('video_id') != video.get('video_id') 
                                       and nm_id not in v.get('nm_ids', [])]
                        
                        if negative_videos:
                            neg_video = random.choice(negative_videos)
                            neg_frames = self._extract_frames_from_mp4(neg_video.get('path_url'))
                            
                            if neg_frames:
                                neg_frame = random.choice(neg_frames)
                                
                                # Добавляем отрицательную пару (тот же товар, но на другом видео без него)
                                self.pairs.append({
                                    'product_img': img_path,
                                    'frame_img': neg_frame,
                                    'label': 0,  # 0 = товара нет на видео
                                    'product_id': nm_id,
                                    'video_id': neg_video.get('video_id')
                                })
                                
                                count += 1
                
                # Если достигли максимального количества пар
                if count >= max_pairs:
                    logger.info(f"Достигнуто максимальное количество пар: {max_pairs}")
                    return
    
    def _get_product_image_paths(self, nm_id):
        """Получение путей к изображениям товара"""
        image_paths = []
        
        # Проверяем наличие основного изображения
        img1_path = os.path.join(self.data_root, f"images_1/{nm_id}/1.webp")
        if os.path.exists(img1_path):
            image_paths.append(img1_path)
        
        # Проверяем наличие дополнительных изображений
        for img_idx in [2, 3]:
            img_path = os.path.join(self.data_root, f"images_2-3/{nm_id}/{img_idx}.webp")
            if os.path.exists(img_path):
                image_paths.append(img_path)
        
        # Проверяем наличие дополнительных изображений 4-5
        for img_idx in [4, 5]:
            img_path = os.path.join(self.data_root, f"images_4-5/{nm_id}/{img_idx}.webp")
            if os.path.exists(img_path):
                image_paths.append(img_path)
        
        return image_paths
    
    def _extract_frames_from_mp4(self, video_path):
        """
        Извлекаем кадры из видео напрямую с помощью OpenCV вместо ffmpeg
        Это должно устранить ошибки с ffmpeg
        """
        try:
            # Создаем уникальное имя директории для этого видео
            video_hash = abs(hash(video_path)) % 10000
            out_dir = os.path.join(self.temp_dir, f"video_{video_hash}")
            os.makedirs(out_dir, exist_ok=True)
            
            # Проверяем кэш - если кадры уже извлечены
            frames = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith('.jpg')]
            if frames:
                return frames
            
            # Если кадров еще нет, извлекаем их с помощью OpenCV
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Не удалось открыть видео: {video_path}")
                return []
            
            # Получаем FPS и вычисляем шаг кадров
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 25  # Если не удалось определить FPS, используем стандартное значение
            
            # Шаг для извлечения кадров (каждые frame_interval секунд)
            frame_step = int(fps * self.frame_interval)
            
            frames_list = []
            count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if count % frame_step == 0:
                    # Сохраняем кадр как изображение
                    frame_path = os.path.join(out_dir, f"frame_{count}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frames_list.append(frame_path)
                
                count += 1
                
                # Ограничиваем количество кадров для экономии памяти
                if len(frames_list) >= 10:
                    break
            
            cap.release()
            return frames_list
        
        except Exception as e:
            logger.error(f"Ошибка при извлечении кадров: {e}")
            return []
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """Получение пары изображений и метки"""
        try:
            item = self.pairs[idx]
            
            # Загружаем изображение товара
            product_img = Image.open(item['product_img']).convert('RGB')
            
            # Загружаем кадр видео
            frame_img = Image.open(item['frame_img']).convert('RGB')
            
            # Применяем трансформации
            if self.transform:
                product_img = self.transform(product_img)
                frame_img = self.transform(frame_img)
            
            return {
                'product': product_img, 
                'frame': frame_img, 
                'label': torch.tensor(item['label'], dtype=torch.float32)
            }
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных для индекса {idx}: {e}")
            # Возвращаем пустые тензоры в случае ошибки
            dummy_img = torch.zeros(3, 224, 224)
            return {'product': dummy_img, 'frame': dummy_img, 'label': torch.tensor(0, dtype=torch.float32)}


class HardTripletDataset(Dataset):
    """
    Датасет для обучения с использованием триплетов (якорь, положительный, отрицательный)
    """
    def __init__(self, data_root, videos_json, transform=None, max_triplets=5000):
        self.data_root = data_root
        self.transform = transform if transform else transforms.ToTensor()
        
        # Создаем временные директории
        self.temp_dir = os.path.join(data_root, "temp_frames")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Загружаем данные о видео
        self.videos_data = self._load_json(videos_json)
        
        # Создаем списки триплетов для обучения
        self.triplets = []
        self._prepare_triplets(max_triplets)
        
        logger.info(f"Создано {len(self.triplets)} триплетов для обучения")
    
    def _load_json(self, path):
        """Загрузка JSON-файла"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Ошибка при загрузке JSON из {path}: {e}")
            return []
    
    def _prepare_triplets(self, max_triplets):
        """Подготовка триплетов для обучения"""
        # Создаем словарь {nm_id: [image_paths]}
        all_products = {}
        
        # Сначала собираем информацию о всех товарах
        for video in self.videos_data:
            for nm_id in video.get('nm_ids', []):
                if nm_id not in all_products:
                    image_paths = self._get_product_image_paths(nm_id)
                    if image_paths:
                        all_products[nm_id] = {
                            'images': image_paths,
                            'videos': [video.get('video_id')]
                        }
                else:
                    if video.get('video_id') not in all_products[nm_id]['videos']:
                        all_products[nm_id]['videos'].append(video.get('video_id'))
        
        # Создаем кэш для кадров видео
        video_frames_cache = {}
        
        count = 0
        # Для каждого товара создаем триплеты
        for nm_id, data in all_products.items():
            # Пропускаем товары без видео
            if not data['videos']:
                continue
            
            # Для каждого видео с этим товаром
            for video_id in data['videos']:
                video = next((v for v in self.videos_data if v.get('video_id') == video_id), None)
                if not video:
                    continue
                
                # Получаем кадры видео (из кэша или извлекаем)
                if video_id not in video_frames_cache:
                    frames = self._extract_frames_from_video(video.get('path_url'))
                    if not frames:
                        continue
                    video_frames_cache[video_id] = frames
                
                frames = video_frames_cache[video_id]
                
                # Для каждой пары (изображение товара, кадр с товаром)
                for img_path in data['images'][:2]:  # Берем не более 2 изображений товара
                    if not frames:
                        continue
                    
                    # Берем случайный кадр как положительный пример
                    pos_frame = random.choice(frames)
                    
                    # Ищем отрицательный пример (видео без этого товара)
                    neg_videos = [v for v in self.videos_data if nm_id not in v.get('nm_ids', [])]
                    
                    if not neg_videos:
                        continue
                    
                    neg_video = random.choice(neg_videos)
                    neg_video_id = neg_video.get('video_id')
                    
                    # Получаем кадры негативного видео
                    if neg_video_id not in video_frames_cache:
                        neg_frames = self._extract_frames_from_video(neg_video.get('path_url'))
                        if not neg_frames:
                            continue
                        video_frames_cache[neg_video_id] = neg_frames
                    
                    neg_frames = video_frames_cache[neg_video_id]
                    
                    if not neg_frames:
                        continue
                    
                    # Берем случайный кадр как отрицательный пример
                    neg_frame = random.choice(neg_frames)
                    
                    # Добавляем триплет
                    self.triplets.append({
                        'anchor': img_path,
                        'positive': pos_frame,
                        'negative': neg_frame,
                        'anchor_id': nm_id,
                        'positive_id': video_id,
                        'negative_id': neg_video_id
                    })
                    
                    count += 1
                    
                    # Если достигли лимита
                    if count >= max_triplets:
                        return
    
    def _get_product_image_paths(self, nm_id):
        """Получение путей к изображениям товара"""
        image_paths = []
        
        # Проверяем наличие основного изображения
        img1_path = os.path.join(self.data_root, f"images_1/{nm_id}/1.webp")
        if os.path.exists(img1_path):
            image_paths.append(img1_path)
        
        # Проверяем наличие дополнительных изображений
        for img_idx in [2, 3]:
            img_path = os.path.join(self.data_root, f"images_2-3/{nm_id}/{img_idx}.webp")
            if os.path.exists(img_path):
                image_paths.append(img_path)
        
        # Проверяем наличие дополнительных изображений 4-5
        for img_idx in [4, 5]:
            img_path = os.path.join(self.data_root, f"images_4-5/{nm_id}/{img_idx}.webp")
            if os.path.exists(img_path):
                image_paths.append(img_path)
        
        return image_paths
    
    def _extract_frames_from_video(self, video_path):
        """Извлечение кадров из видео с помощью OpenCV"""
        try:
            # Создаем уникальное имя директории для этого видео
            video_hash = abs(hash(video_path)) % 10000
            out_dir = os.path.join(self.temp_dir, f"video_{video_hash}")
            os.makedirs(out_dir, exist_ok=True)
            
            # Проверяем кэш - если кадры уже извлечены
            frames = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith('.jpg')]
            if frames:
                return frames
            
            # Открываем видеофайл
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Не удалось открыть видео: {video_path}")
                return []
            
            # Получаем информацию о видео
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 25  # Стандартное значение, если не удалось определить
            
            # Интервал между кадрами (1 кадр в секунду)
            frame_step = int(fps)
            
            frames_list = []
            count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if count % frame_step == 0:
                    # Сохраняем кадр
                    frame_path = os.path.join(out_dir, f"frame_{count}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frames_list.append(frame_path)
                
                count += 1
                
                # Ограничиваем количество кадров
                if len(frames_list) >= 10:
                    break
            
            cap.release()
            return frames_list
        
        except Exception as e:
            logger.error(f"Ошибка при извлечении кадров: {e}")
            return []
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        """Получение триплета (якорь, положительный, отрицательный)"""
        try:
            triplet = self.triplets[idx]
            
            # Загружаем изображения
            anchor = Image.open(triplet['anchor']).convert('RGB')
            positive = Image.open(triplet['positive']).convert('RGB')
            negative = Image.open(triplet['negative']).convert('RGB')
            
            # Применяем трансформации
            if self.transform:
                anchor = self.transform(anchor)
                positive = self.transform(positive)
                negative = self.transform(negative)
            
            return {
                'anchor': anchor,
                'positive': positive,
                'negative': negative
            }
        
        except Exception as e:
            logger.error(f"Ошибка при загрузке триплета {idx}: {e}")
            # Возвращаем пустые тензоры в случае ошибки
            dummy_img = torch.zeros(3, 224, 224)
            return {
                'anchor': dummy_img,
                'positive': dummy_img,
                'negative': dummy_img
            }