import os
import numpy as np
import torch
from PIL import Image
import cv2
import time
from pathlib import Path
from tqdm import tqdm
import faiss
import pickle
from typing import List, Tuple, Dict, Optional, Union

# Для распараллеливания
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

# Загрузка моделей
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

class ProductVideoMatcher:
    def __init__(self, 
                 detection_model_path: str = "yolov8n.pt",
                 embedding_model: str = "openai/clip-vit-base-patch32",
                 similarity_threshold: float = 0.7,
                 frame_sample_rate: int = 1,
                 use_gpu: bool = torch.cuda.is_available(),
                 cache_dir: Optional[str] = 'cache'):
        """
        Инициализация системы сравнения товаров с видео
        
        Args:
            detection_model_path: Путь к модели YOLO
            embedding_model: Название модели для эмбеддингов
            similarity_threshold: Порог сходства для сопоставления
            frame_sample_rate: Частота выборки кадров (в секундах)
            use_gpu: Использовать ли GPU
            cache_dir: Директория для кэширования
        """
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.similarity_threshold = similarity_threshold
        self.frame_sample_rate = frame_sample_rate
        
        # Настройка кэширования
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir
        
        print(f"Инициализация на устройстве: {self.device}")
        
        # Загрузка моделей
        print("Загрузка моделей...")
        self.detector = YOLO(detection_model_path)  # Детектор объектов
        
        # Модель для извлечения признаков
        self.clip_model = CLIPModel.from_pretrained(embedding_model).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(embedding_model)
        
        # Индекс для быстрого поиска похожих векторов
        self.faiss_index = None
        self.product_ids = []
        
        print("Модели загружены успешно")
    
    def _get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Извлечение вектора признаков из изображения"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        inputs = self.clip_processor(
            images=image, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            
        # Нормализация вектора для косинусного сходства
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features.cpu().numpy()[0]
    
    def _detect_objects(self, image: Image.Image) -> List[Image.Image]:
        """Обнаружение и вырезание объектов с изображения"""
        # Выполняем детекцию
        results = self.detector(image, verbose=False)
        if not results:
            return []
            
        cropped_images = []
        for result in results[0].boxes.data:
            # Извлекаем координаты и достоверность
            xmin, ymin, xmax, ymax, conf, class_id = result
            
            # Отфильтровываем детекции с низкой достоверностью
            if conf < 0.5:
                continue
                
            # Обрезаем объект
            try:
                cropped = image.crop((int(xmin), int(ymin), int(xmax), int(ymax)))
                if cropped.size[0] > 20 and cropped.size[1] > 20:  # Минимальный размер
                    cropped_images.append(cropped)
            except Exception as e:
                print(f"Ошибка при обрезке: {e}")
                
        return cropped_images
    
    def _get_cache_path(self, image_path: str) -> str:
        """Получение пути для кэшированных эмбеддингов"""
        if not self.cache_dir:
            return None
        
        # Создаем уникальное имя файла на основе пути
        filename = Path(image_path).stem
        return os.path.join(self.cache_dir, f"{filename}_embedding.pkl")
    
    def index_product(self, product_id: str, product_images: List[str]) -> None:
        """
        Индексирование товара по его изображениям
        
        Args:
            product_id: Идентификатор товара
            product_images: Список путей к изображениям товара
        """
        embeddings = []
        
        for img_path in product_images:
            cache_path = self._get_cache_path(img_path)
            
            # Проверяем наличие кэша
            if cache_path and os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        embedding_data = pickle.load(f)
                        embeddings.extend(embedding_data)
                    continue
                except Exception as e:
                    print(f"Ошибка загрузки кэша: {e}")
            
            # Обрабатываем изображение
            try:
                image = Image.open(img_path)
                object_images = self._detect_objects(image)
                
                # Если не найдено объектов, используем всё изображение
                if not object_images:
                    object_images = [image]
                
                # Извлекаем эмбеддинги всех объектов
                image_embeddings = [self._get_image_embedding(obj) for obj in object_images]
                embeddings.extend(image_embeddings)
                
                # Сохраняем в кэш
                if cache_path:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(image_embeddings, f)
                        
            except Exception as e:
                print(f"Ошибка при обработке {img_path}: {e}")
        
        if not embeddings:
            print(f"Предупреждение: Не удалось получить эмбеддинги для товара {product_id}")
            return
            
        # Добавляем идентификатор для каждого эмбеддинга
        for _ in range(len(embeddings)):
            self.product_ids.append(product_id)
        
        # Обновляем индекс FAISS
        embeddings_array = np.array(embeddings).astype('float32')
        
        if self.faiss_index is None:
            dimension = embeddings_array.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Индекс для косинусного сходства
            
        self.faiss_index.add(embeddings_array)
    
    def batch_index_products(self, products_data: Dict[str, List[str]]) -> None:
        """
        Пакетное индексирование нескольких товаров
        
        Args:
            products_data: Словарь {product_id: [image_paths]}
        """
        print(f"Индексирование {len(products_data)} товаров...")
        
        for product_id, image_paths in tqdm(products_data.items()):
            self.index_product(product_id, image_paths)
            
        print(f"Индексирование завершено. Всего эмбеддингов: {len(self.product_ids)}")
    
    def _process_video_chunk(self, video_path: str, start_time: float, 
                           end_time: float) -> List[Tuple[str, float]]:
        """
        Обработка части видео для распараллеливания
        
        Returns:
            List[Tuple[product_id, timestamp]]
        """
        matches = []
        cap = cv2.VideoCapture(video_path)
        
        # Устанавливаем начальную позицию
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_count = start_frame
        end_frame = int(end_time * fps)
        
        skip_frames = int(fps * self.frame_sample_rate)
        skip_frames = max(1, skip_frames)  # Минимум 1
        
        while cap.isOpened() and frame_count < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Обрабатываем только каждый N-й кадр
            if (frame_count - start_frame) % skip_frames == 0:
                timestamp = frame_count / fps
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Обнаруживаем объекты на кадре
                objects = self._detect_objects(frame_pil)
                
                # Если объекты найдены, проверяем совпадения
                if objects:
                    for obj in objects:
                        embedding = self._get_image_embedding(obj)
                        
                        # Быстрый поиск по индексу FAISS
                        D, I = self.faiss_index.search(
                            np.array([embedding]).astype('float32'), 
                            k=5  # Ищем 5 ближайших соседей
                        )
                        
                        # Проверяем совпадения по порогу сходства
                        for i, (similarity, idx) in enumerate(zip(D[0], I[0])):
                            if similarity > self.similarity_threshold:
                                product_id = self.product_ids[idx]
                                matches.append((product_id, timestamp))
                                # После первого совпадения прекращаем поиск для этого объекта
                                break
            
            frame_count += 1
            
        cap.release()
        return matches
        
    def find_product_in_video(self, video_path: str, num_workers: int = 4) -> Dict[str, List[float]]:
        """
        Поиск всех проиндексированных товаров в видео
        
        Args:
            video_path: Путь к видео
            num_workers: Количество параллельных потоков
            
        Returns:
            Dictionary {product_id: [timestamps]}
        """
        if not self.faiss_index or len(self.product_ids) == 0:
            raise ValueError("Нет проиндексированных товаров. Сначала выполните индексацию.")
            
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Видео не найдено: {video_path}")
            
        # Получаем информацию о видео
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        cap.release()
        
        print(f"Анализ видео ({duration:.2f} сек)...")
        
        # Разделяем видео на части для параллельной обработки
        chunk_size = duration / num_workers
        chunks = [(i * chunk_size, min((i + 1) * chunk_size, duration)) 
                 for i in range(num_workers)]
        
        # Параллельная обработка частей видео
        all_matches = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(self._process_video_chunk, video_path, start, end)
                for start, end in chunks
            ]
            
            for future in tqdm(futures):
                chunk_matches = future.result()
                all_matches.extend(chunk_matches)
        
        # Группируем результаты по товарам
        results = {}
        for product_id, timestamp in all_matches:
            if product_id not in results:
                results[product_id] = []
            results[product_id].append(timestamp)
            
        # Постобработка: объединяем близкие таймкоды и сортируем
        for product_id in results:
            # Сортируем таймкоды
            timestamps = sorted(results[product_id])
            
            # Объединяем близкие таймкоды (в пределах 1 секунды)
            merged_timestamps = []
            if timestamps:
                current_group = [timestamps[0]]
                
                for t in timestamps[1:]:
                    if t - current_group[-1] <= 1.0:
                        current_group.append(t)
                    else:
                        # Берем среднее время для группы
                        merged_timestamps.append(sum(current_group) / len(current_group))
                        current_group = [t]
                
                # Добавляем последнюю группу
                if current_group:
                    merged_timestamps.append(sum(current_group) / len(current_group))
            
            results[product_id] = merged_timestamps
        
        return results
    
    def find_specific_product_in_video(self, 
                                      product_images: List[str], 
                                      video_path: str) -> List[float]:
        """
        Поиск конкретного товара в видео без предварительной индексации
        
        Args:
            product_images: Список путей к изображениям товара
            video_path: Путь к видео
            
        Returns:
            List[float]: Список таймкодов
        """
        # Создаем временный ID для товара
        temp_id = "temp_product"
        
        # Индексируем товар
        self.index_product(temp_id, product_images)
        
        # Ищем товар в видео
        results = self.find_product_in_video(video_path)
        
        # Получаем таймкоды для этого товара
        timestamps = results.get(temp_id, [])
        
        # Очищаем временные данные
        # Тут должен быть код для удаления временных данных из индекса,
        # но для простоты оставим как есть
        
        return timestamps
    
    def save_model(self, path: str) -> None:
        """Сохранение модели и индекса"""
        data = {
            'product_ids': self.product_ids,
            'similarity_threshold': self.similarity_threshold,
            'frame_sample_rate': self.frame_sample_rate
        }
        
        os.makedirs(path, exist_ok=True)
        
        # Сохраняем параметры
        with open(os.path.join(path, 'params.pkl'), 'wb') as f:
            pickle.dump(data, f)
        
        # Сохраняем индекс FAISS
        if self.faiss_index:
            faiss.write_index(self.faiss_index, os.path.join(path, 'index.faiss'))
            
        print(f"Модель сохранена в {path}")
    
    def load_model(self, path: str) -> None:
        """Загрузка сохраненной модели и индекса"""
        # Загружаем параметры
        with open(os.path.join(path, 'params.pkl'), 'rb') as f:
            data = pickle.load(f)
            
        self.product_ids = data['product_ids']
        self.similarity_threshold = data['similarity_threshold']
        self.frame_sample_rate = data['frame_sample_rate']
        
        # Загружаем индекс FAISS
        if os.path.exists(os.path.join(path, 'index.faiss')):
            self.faiss_index = faiss.read_index(os.path.join(path, 'index.faiss'))
            
        print(f"Модель загружена из {path}")




def testDef(product_images, video_path):
    matcher = ProductVideoMatcher(
        detection_model_path="yolov8n.pt",
        similarity_threshold=0.7,
        frame_sample_rate=1,
        cache_dir="cache"
    )
    
    # Пример 1: Поиск конкретного товара на видео
    product_images = ["путь/к/товару/img1.jpg", "путь/к/товару/img2.jpg"]
    video_path = "путь/к/видео.mp4"
    
    timestamps = matcher.find_specific_product_in_video(product_images, video_path)
    print(f"Товар обнаружен в следующих таймкодах: {timestamps}")


# Пример использования
# if __name__ == "__main__":
#     # Создаем экземпляр класса
#     matcher = ProductVideoMatcher(
#         detection_model_path="yolov8n.pt",
#         similarity_threshold=0.7,
#         frame_sample_rate=1,
#         cache_dir="cache"
#     )
    
#     # Пример 1: Поиск конкретного товара на видео
#     product_images = ["путь/к/товару/img1.jpg", "путь/к/товару/img2.jpg"]
#     video_path = "путь/к/видео.mp4"
    
#     timestamps = matcher.find_specific_product_in_video(product_images, video_path)
#     print(f"Товар обнаружен в следующих таймкодах: {timestamps}")
    
#     # Пример 2: Индексация нескольких товаров
#     products_data = {
#         "product1": ["путь/к/товару1/img1.jpg", "путь/к/товару1/img2.jpg"],
#         "product2": ["путь/к/товару2/img1.jpg", "путь/к/товару2/img2.jpg"],
#         # ...
#     }
    
#     matcher.batch_index_products(products_data)
    
#     # Поиск всех товаров на видео
#     results = matcher.find_product_in_video(video_path)
    
#     for product_id, timestamps in results.items():
#         print(f"Товар {product_id} обнаружен в таймкодах: {timestamps}")
    
#     # Сохранение модели для последующего использования
#     matcher.save_model("saved_model")