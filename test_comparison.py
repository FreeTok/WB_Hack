import os
import torch
from PIL import Image
from PIL import ImageEnhance  # Правильный импорт ImageEnhance
import numpy as np
import cv2
from ultralytics import YOLO
from torchvision import transforms
import torchvision.models as models
from sklearn.metrics.pairwise import cosine_similarity
import time
import json
from pathlib import Path
import subprocess
from collections import defaultdict

# Глобальные переменные для моделей
global_yolo_model = None
global_feature_model = None
global_transform = None

# Проверяем доступность CUDA и устанавливаем устройство
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")

# Константы для настройки алгоритма
CONFIDENCE_THRESHOLD = 0.25  # Порог уверенности YOLO
IOU_THRESHOLD = 0.45  # Порог IoU для NMS
SIMILARITY_THRESHOLD_DEFAULT = 0.7  # Базовый порог сходства
MULTI_SCALE = True  # Включить обработку в нескольких масштабах
FRAME_RATE = 2  # Частота извлечения кадров (кадров в секунду)
MAX_IMAGE_SIZE = 640  # Максимальный размер изображения для обработки

# Категории товаров и их пороги сходства
PRODUCT_CATEGORIES = {
    "одежда": 0.65,
    "электроника": 0.75,
    "косметика": 0.70,
    "аксессуары": 0.65,
    "обувь": 0.65,
    "default": 0.70
}

def get_yolo_model():
    """Получение модели YOLO с дополнительными оптимизациями"""
    global global_yolo_model
    if global_yolo_model is None:
        try:
            print(f"Загрузка модели YOLO на {device}...")
            # Загружаем модель YOLO
            global_yolo_model = YOLO("yolo11n.pt", verbose=False)
            
            # Настраиваем параметры модели
            if hasattr(global_yolo_model, 'conf'):
                global_yolo_model.conf = CONFIDENCE_THRESHOLD
            if hasattr(global_yolo_model, 'iou'):
                global_yolo_model.iou = IOU_THRESHOLD
            
            # Проверка и принудительное использование GPU
            if device.type == 'cuda':
                if hasattr(global_yolo_model, 'to'):
                    global_yolo_model.to(device)
                print("Модель YOLO использует GPU")
            
            print("Модель YOLO загружена успешно!")
        except Exception as e:
            print(f"Ошибка при загрузке модели YOLO: {str(e)}")
            import traceback
            print(traceback.format_exc())
            # В случае ошибки используем базовую конфигурацию
            global_yolo_model = YOLO("yolo11n.pt", verbose=False)
    
    return global_yolo_model

def get_feature_model():
    """Получение улучшенной модели извлечения признаков"""
    global global_feature_model, global_transform
    if global_feature_model is None:
        try:
            print(f"Загрузка модели для извлечения признаков на {device}...")
            
            # Используем более продвинутую модель для извлечения признаков
            global_feature_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
            # Удаляем последний слой (классификатор)
            global_feature_model = torch.nn.Sequential(*list(global_feature_model.children())[:-1])
            # Переносим модель на GPU
            global_feature_model = global_feature_model.to(device)
            # Переключаем модель в режим вывода
            global_feature_model.eval()
            
            # Улучшенное преобразование изображений
            global_transform = transforms.Compose([
                transforms.Resize((300, 300)),  # Увеличенный размер для лучшего качества
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            print("Модель для извлечения признаков загружена успешно!")
        except Exception as e:
            print(f"Ошибка при загрузке модели признаков: {str(e)}")
            try:
                # Пробуем загрузить EfficientNet B0 в случае ошибки
                print("Попытка загрузить EfficientNet B0...")
                global_feature_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
                global_feature_model = torch.nn.Sequential(*list(global_feature_model.children())[:-1])
                global_feature_model = global_feature_model.to(device)
                global_feature_model.eval()
                
                # Стандартное преобразование
                global_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                print("Модель EfficientNet B0 загружена успешно!")
            except Exception as e2:
                print(f"Ошибка при загрузке запасной модели: {str(e2)}")
                # В крайнем случае используем MobileNet
                global_feature_model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
                global_feature_model = torch.nn.Sequential(*list(global_feature_model.children())[:-1])
                global_feature_model = global_feature_model.to('cpu')
                global_feature_model.eval()
                print("Модель загружена на CPU")
    
    return global_feature_model

def get_transform():
    """Получение трансформации для изображений"""
    global global_transform
    if global_transform is None:
        global_transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return global_transform

class ProcessedFrame:
    """Класс для хранения обработанного кадра с улучшенной структурой"""
    def __init__(self, frame_index, time_code, detected_objects=None, feature_vectors=None, 
                 confidence_scores=None, detected_classes=None):
        self.frame_index = frame_index
        self.time_code = time_code
        self.detected_objects = detected_objects or []
        self.feature_vectors = feature_vectors or []
        self.confidence_scores = confidence_scores or []
        self.detected_classes = detected_classes or []

def preprocess_image(image):
    """Улучшенная предобработка изображения"""
    # Проверка размера изображения и изменение размера при необходимости
    if max(image.size) > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    
    # Проверка и коррекция цветового пространства
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Улучшение контраста и яркости для лучшего обнаружения
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)  # Увеличиваем контраст на 20%
    
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.1)  # Увеличиваем яркость на 10%
    
    return image

def get_detections(results):
    """Извлечение координат и данных обнаруженных объектов с дополнительной информацией"""
    try:
        if not results or len(results) == 0: 
            return None, None, None

        detections = []
        confidence_scores = []
        class_labels = []
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    if box.xyxy is not None and len(box.xyxy) > 0:
                        # Переводим координаты на CPU и в список
                        if torch.is_tensor(box.xyxy):
                            box_coords = box.xyxy.cpu().tolist()[0]
                        else:
                            box_coords = box.xyxy[0]
                        detections.append(box_coords)
                        
                        # Получаем уверенность модели
                        if hasattr(box, 'conf') and box.conf is not None:
                            if torch.is_tensor(box.conf):
                                conf = box.conf.cpu().item()
                            else:
                                conf = box.conf
                            confidence_scores.append(conf)
                        else:
                            confidence_scores.append(1.0)  # По умолчанию
                        
                        # Получаем класс объекта, если доступен
                        if hasattr(box, 'cls') and box.cls is not None:
                            if torch.is_tensor(box.cls):
                                cls = int(box.cls.cpu().item())
                            else:
                                cls = int(box.cls)
                            class_labels.append(cls)
                        else:
                            class_labels.append(-1)  # Неизвестный класс
        
        return detections, confidence_scores, class_labels if detections else (None, None, None)
    except Exception as e:
        print(f"Ошибка в get_detections: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, None, None

def crop_objects(image, detections):
    """Улучшенная функция для вырезания объектов с динамическим расширением границ"""
    if not detections: 
        return None

    cropped_images = []
    for detection in detections:
        try:
            xmin, ymin, xmax, ymax = [int(coord) for coord in detection]
            
            # Расширяем границы объекта на 5%, чтобы захватить контекст
            width, height = xmax - xmin, ymax - ymin
            padding_x = int(width * 0.05)
            padding_y = int(height * 0.05)
            
            xmin = max(0, xmin - padding_x)
            ymin = max(0, ymin - padding_y)
            xmax = min(image.width, xmax + padding_x)
            ymax = min(image.height, ymax + padding_y)
            
            if xmax > xmin and ymax > ymin:  # Убеждаемся, что область корректна
                cropped_image = image.crop((xmin, ymin, xmax, ymax))
                if cropped_image.size[0] > 0 and cropped_image.size[1] > 0:
                    # Применяем предобработку к вырезанному объекту
                    cropped_image = preprocess_image(cropped_image)
                    cropped_images.append(cropped_image)
        except Exception as e:
            print(f"Ошибка при кадрировании: {str(e)}")
            continue

    return cropped_images if cropped_images else None

def get_feature_vector(image):
    """Извлечение векторов признаков из изображения с улучшенным подходом"""
    try:
        # Получаем модель и трансформацию
        model = get_feature_model()
        transform = get_transform()
        
        # Применяем трансформацию и перемещаем на нужное устройство
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Извлекаем признаки
        with torch.no_grad():
            features = model(image_tensor)
            # Нормализуем вектор признаков
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            # Перемещаем результат на CPU для дальнейшей обработки
            features = features.cpu()
        
        return features.squeeze().numpy()
    except Exception as e:
        print(f"Ошибка в get_feature_vector: {str(e)}")
        return None

def compare_features(features1, features2, threshold=None, confidence_scores=None):
    """
    Улучшенное сравнение векторов признаков с динамическим порогом и учетом уверенности
    
    Args:
        features1: Векторы признаков первого набора объектов
        features2: Векторы признаков второго набора объектов
        threshold: Порог сходства (если None, используется SIMILARITY_THRESHOLD_DEFAULT)
        confidence_scores: Уверенность в обнаружении объектов (влияет на итоговую оценку)
        
    Returns:
        bool: Найдено ли сходство
        float: Максимальное значение сходства
    """
    try:
        if threshold is None:
            threshold = SIMILARITY_THRESHOLD_DEFAULT
            
        max_similarity = 0.0
        found_match = False
        
        for i, f1 in enumerate(features1):
            for j, f2 in enumerate(features2):
                # Базовое сходство по косинусной мере
                similarity = cosine_similarity(f1.reshape(1, -1), f2.reshape(1, -1))[0][0]
                
                # Корректировка сходства с учетом уверенности (если предоставлена)
                if confidence_scores is not None and j < len(confidence_scores):
                    conf_weight = min(1.0, confidence_scores[j] * 1.2)  # Увеличиваем вес уверенности
                    similarity = similarity * (0.8 + 0.2 * conf_weight)  # Взвешенное сходство
                
                max_similarity = max(max_similarity, similarity)
                
                if similarity > threshold:
                    print(f"Найдено сходство: {similarity:.4f} (порог: {threshold:.4f})")
                    found_match = True
        
        return found_match, max_similarity
    except Exception as e:
        print(f"Ошибка в compare_features: {str(e)}")
        return False, 0.0

def determine_product_category(product_id, product_name=None):
    """
    Определение категории товара для настройки параметров сравнения
    
    Args:
        product_id: ID товара
        product_name: Название товара (если доступно)
        
    Returns:
        str: Категория товара
        float: Рекомендуемый порог сходства для данной категории
    """
    # Здесь можно использовать внешнюю API или локальную базу данных
    # для определения категории по ID товара
    
    # Простая логика определения категории по имени товара (если доступно)
    if product_name:
        product_name = product_name.lower()
        for category in PRODUCT_CATEGORIES:
            if category in product_name:
                return category, PRODUCT_CATEGORIES[category]
    
    # По умолчанию возвращаем общую категорию
    return "default", PRODUCT_CATEGORIES["default"]

def extract_and_process_frames(video_path, frame_rate=FRAME_RATE, multi_scale=MULTI_SCALE):
    """
    Улучшенная функция извлечения и обработки кадров из видео
    
    Args:
        video_path: Путь к видео-файлу
        frame_rate: Частота извлечения кадров (кадров в секунду)
        multi_scale: Использовать ли обработку в нескольких масштабах
        
    Returns:
        list: Список предварительно обработанных кадров
    """
    try:
        if not os.path.exists(video_path):
            print(f"Ошибка: Файл видео не существует - {video_path}")
            return []
        
        # Загружаем модель YOLO
        yolo_model = get_yolo_model()
        
        # Открываем видео
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Ошибка: Не удалось открыть видео - {video_path}")
            return []
        
        processed_frames = []
        frame_count = 0
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30
            print(f"Не удалось определить FPS видео, используем значение по умолчанию: {fps}")
        
        # Общая длительность видео в секундах
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if total_frames > 0 else 0
        
        print(f"Извлечение кадров из видео: {video_path}")
        print(f"Частота кадров: {fps}, длительность: {duration:.2f}с, частота извлечения: {frame_rate}")
        
        # Расчет интервала между кадрами
        frame_interval = max(1, int(fps / frame_rate))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Извлекаем кадры с заданной частотой
            if frame_count % frame_interval == 0:
                time_code = frame_count / fps
                print(f"Обработка кадра {frame_count} (таймкод: {time_code:.2f}с)")
                
                # Преобразуем кадр в PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Предобработка кадра
                frame_pil = preprocess_image(frame_pil)
                
                # Массив для хранения результатов с разных масштабов
                all_detections = []
                all_scores = []
                all_classes = []
                all_cropped = []
                all_features = []
                
                # Основная обработка кадра
                try:
                    # Выполняем предсказание с YOLO
                    if device.type == 'cuda':
                        with torch.amp.autocast(device_type='cuda'):
                            results = yolo_model(frame_rgb, verbose=False)
                    else:
                        results = yolo_model(frame_rgb, verbose=False)
                    
                    # Извлекаем обнаруженные объекты
                    detections, confidence_scores, class_labels = get_detections(results)
                    
                    if detections:
                        # Вырезаем обнаруженные объекты
                        cropped_objects = crop_objects(frame_pil, detections)
                        
                        if cropped_objects:
                            # Добавляем результаты основного масштаба
                            all_detections.extend(detections)
                            all_scores.extend(confidence_scores)
                            all_classes.extend(class_labels)
                            all_cropped.extend(cropped_objects)
                            
                            # Извлекаем векторы признаков
                            features = [get_feature_vector(img) for img in cropped_objects]
                            # Фильтруем None значения
                            features = [f for f in features if f is not None]
                            all_features.extend(features)
                    
                    # Если включен режим множественных масштабов, обрабатываем дополнительно
                    if multi_scale and frame_pil.width > 400 and frame_pil.height > 400:
                        # Создаем уменьшенную копию изображения (0.5 от оригинала)
                        small_frame = frame_pil.resize((frame_pil.width // 2, frame_pil.height // 2), 
                                                  Image.LANCZOS)
                        
                        # Обрабатываем уменьшенное изображение
                        small_results = yolo_model(np.array(small_frame), verbose=False)
                        small_detections, small_scores, small_classes = get_detections(small_results)
                        
                        if small_detections:
                            # Масштабируем координаты обратно к оригинальному размеру
                            scaled_detections = []
                            for det in small_detections:
                                scaled_det = [coord * 2 for coord in det]
                                scaled_detections.append(scaled_det)
                            
                            small_cropped = crop_objects(frame_pil, scaled_detections)
                            
                            if small_cropped:
                                all_detections.extend(scaled_detections)
                                all_scores.extend(small_scores)
                                all_classes.extend(small_classes)
                                all_cropped.extend(small_cropped)
                                
                                small_features = [get_feature_vector(img) for img in small_cropped]
                                small_features = [f for f in small_features if f is not None]
                                all_features.extend(small_features)
                
                    # Создаем и сохраняем обработанный кадр, если есть признаки
                    if all_features:
                        processed_frame = ProcessedFrame(
                            frame_index=frame_count,
                            time_code=time_code,
                            detected_objects=all_cropped,
                            feature_vectors=all_features,
                            confidence_scores=all_scores,
                            detected_classes=all_classes
                        )
                        processed_frames.append(processed_frame)
                
                except Exception as e:
                    print(f"Ошибка при обработке кадра {frame_count}: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
            
            frame_count += 1
            
            # Показываем прогресс каждые 50 кадров
            if frame_count % 50 == 0 and total_frames > 0:
                progress = (frame_count / total_frames) * 100
                print(f"Прогресс обработки: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Освобождаем ресурсы
        cap.release()
        
        print(f"Обработано {len(processed_frames)} кадров из видео")
        return processed_frames
    
    except Exception as e:
        import traceback
        print(f"Ошибка при извлечении и обработке кадров: {str(e)}")
        print(traceback.format_exc())
        return []

def improved_compare_video_image(product_images, video_path, product_info=None, debug_save=False, 
                                frame_rate=FRAME_RATE):
    """
    Улучшенная функция проверки наличия товара в видео
    
    Args:
        product_images: Список путей к изображениям товара
        video_path: Путь к видео-файлу
        product_info: Словарь с информацией о товаре (id, название и т.д.)
        debug_save: Сохранять ли отладочные изображения
        frame_rate: Частота извлечения кадров
        
    Returns:
        tuple: (matched, timecodes, max_similarity, debug_info)
    """
    try:
        if not product_images or not video_path:
            print("Не указаны пути к изображениям товара или видео")
            return (False, [], 0.0, {"error": "Не указаны пути к изображениям товара или видео"})
        
        for img_path in product_images:
            if not os.path.exists(img_path):
                print(f"Ошибка: Файл изображения не существует - {img_path}")
                return (False, [], 0.0, {"error": f"Файл изображения не существует - {img_path}"})
        
        if not os.path.exists(video_path):
            print(f"Ошибка: Файл видео не существует - {video_path}")
            return (False, [], 0.0, {"error": f"Файл видео не существует - {video_path}"})
        
        # Определяем категорию товара и порог сходства
        product_category = "default"
        similarity_threshold = SIMILARITY_THRESHOLD_DEFAULT
        
        if product_info and "id" in product_info:
            product_name = product_info.get("name", "")
            product_category, similarity_threshold = determine_product_category(
                product_info["id"], product_name
            )
            print(f"Определена категория товара: {product_category}, порог: {similarity_threshold}")
        
        # Извлекаем и обрабатываем все кадры видео один раз
        processed_frames = extract_and_process_frames(video_path, frame_rate=frame_rate)
        
        if not processed_frames:
            print("Не удалось извлечь кадры из видео или не найдены объекты")
            return (False, [], 0.0, {"error": "Не удалось извлечь кадры из видео или не найдены объекты"})
        
        matched = False
        timecodes = []
        max_similarity_value = 0.0
        debug_info = {
            "processed_frames": len(processed_frames),
            "product_category": product_category,
            "threshold": similarity_threshold,
            "matches": []
        }
        
        # Проверяем каждое изображение товара
        for img_index, img_path in enumerate(product_images):
            print(f"Проверка изображения товара {img_index+1}/{len(product_images)}: {img_path}")
            
            try:
                # Загружаем изображение товара
                product_img = Image.open(img_path)
                
                # Применяем предобработку
                product_img = preprocess_image(product_img)
                
                # Обрабатываем изображение товара с YOLO
                if device.type == 'cuda':
                    with torch.amp.autocast(device_type='cuda'):
                        results = get_yolo_model()(np.array(product_img), verbose=False)
                else:
                    results = get_yolo_model()(np.array(product_img), verbose=False)
                
                # Извлекаем обнаруженные объекты
                detections, confidence_scores, _ = get_detections(results)
                
                if not detections:
                    print(f"На изображении {img_path} не обнаружены объекты, используем всё изображение целиком")
                    # Если объекты не найдены, используем всё изображение
                    cropped_objects = [product_img]
                    confidence_scores = [0.9]  # Высокая уверенность для целого изображения
                else:
                    # Вырезаем обнаруженные объекты
                    cropped_objects = crop_objects(product_img, detections)
                
                if not cropped_objects:
                    print(f"Не удалось обработать изображение {img_path}, пропускаем")
                    continue
                
                # Сохраняем отладочное изображение товара, если требуется
                if debug_save:
                    os.makedirs('debug_images', exist_ok=True)
                    for i, obj in enumerate(cropped_objects):
                        obj.save(f'debug_images/product_{os.path.basename(img_path)}_{i}.jpg')
                
                # Извлекаем векторы признаков
                product_features = [get_feature_vector(img) for img in cropped_objects]
                
                # Фильтруем None значения
                product_features = [f for f in product_features if f is not None]
                
                if not product_features:
                    print(f"Не удалось извлечь признаки из изображения {img_path}")
                    continue
                
                # Сравниваем с каждым обработанным кадром
                for frame_index, frame in enumerate(processed_frames):
                    # Проводим сравнение с динамическим порогом
                    is_match, frame_similarity = compare_features(
                        product_features, 
                        frame.feature_vectors,
                        threshold=similarity_threshold,
                        confidence_scores=frame.confidence_scores
                    )
                    
                    # Обновляем максимальное значение сходства
                    max_similarity_value = max(max_similarity_value, frame_similarity)
                    
                    if is_match:
                        matched = True
                        if frame.time_code not in timecodes:
                            timecodes.append(frame.time_code)
                        
                        # Добавляем информацию о совпадении для отладки
                        match_info = {
                            "time_code": frame.time_code,
                            "frame_index": frame.frame_index,
                            "similarity": float(frame_similarity),
                            "product_image": os.path.basename(img_path)
                        }
                        debug_info["matches"].append(match_info)
                        
                        # Сохраняем отладочные изображения, если требуется
                        if debug_save:
                            os.makedirs('debug_images', exist_ok=True)
                            # Сохраняем кадр с объектом
                            for i, obj in enumerate(frame.detected_objects[:3]):  # Сохраняем до 3 объектов
                                obj.save(f'debug_images/match_frame_{frame.time_code:.2f}_obj_{i}.jpg')
                
                # Если нашли совпадение для этого изображения товара, продолжаем проверку остальных
                if matched:
                    print(f"Найдено совпадение для изображения {img_path}")
                
            except Exception as e:
                print(f"Ошибка при обработке изображения {img_path}: {str(e)}")
                import traceback
                print(traceback.format_exc())
        
        # Сортируем таймкоды
        if matched and timecodes:
            timecodes = sorted(list(set(timecodes)))
        
        # Добавляем итоговую информацию в отладочные данные
        debug_info["max_similarity"] = float(max_similarity_value)
        debug_info["total_matches"] = len(debug_info["matches"])
        
        return (matched, timecodes, max_similarity_value, debug_info)
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Ошибка при сравнении видео и изображения: {str(e)}")
        print(error_trace)
        return (False, [], 0.0, {"error": str(e), "trace": error_trace})

def download_video(video_url, output_path, force_download=False):
    """
    Скачивание видео с помощью ffmpeg с расширенными опциями
    
    Args:
        video_url: URL видео для скачивания
        output_path: Путь сохранения видео
        force_download: Принудительно скачать, даже если файл существует
        
    Returns:
        bool: Успешно ли скачано видео
    """
    try:
        # Проверяем, существует ли уже скачанное видео
        if os.path.exists(output_path) and not force_download:
            # Проверяем размер файла, чтобы убедиться, что он не пустой
            file_size = os.path.getsize(output_path)
            if file_size > 1024:  # Файл больше 1KB, можем считать, что он корректен
                print(f"Используем существующее видео {output_path}")
                return True
            else:
                # Файл слишком мал, удаляем его и скачиваем заново
                os.remove(output_path)
        
        # Находим исполняемый файл ffmpeg
        ffmpeg_path = 'ffmpeg'
        for possible_path in [
            './Ffmpeg/bin/ffmpeg.exe', 
            './ffmpeg.exe', 
            'ffmpeg.exe', 
            '/usr/bin/ffmpeg', 
            '/usr/local/bin/ffmpeg'
        ]:
            if os.path.exists(possible_path):
                ffmpeg_path = possible_path
                break
        
        print(f"Скачиваю видео {video_url}...")
        
        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Скачиваем видео с улучшенными параметрами
        command = [
            ffmpeg_path,
            '-y',  # Перезаписать файл, если существует
            '-i', video_url,
            '-c', 'copy',  # Копировать потоки без перекодирования
            '-v', 'warning',  # Уровень вывода: только предупреждения
            '-stats',  # Показывать статистику
            '-reconnect', '1',  # Повторное подключение при ошибках сети
            '-reconnect_streamed', '1',
            '-reconnect_delay_max', '10',  # Максимальная задержка повторного подключения
            output_path
        ]
        
        # Запускаем процесс скачивания
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Проверяем, был ли создан файл видео
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
            print(f"Видео успешно скачано: {output_path}")
            return True
        else:
            print(f"Ошибка при скачивании видео: {result.stderr.decode('utf-8', errors='ignore')}")
            return False
    
    except Exception as e:
        print(f"Ошибка при скачивании видео: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def optimized_check(i, data_folder=None, debug_mode=False):
    """
    Улучшенная функция проверки наличия товара в видео
    
    Args:
        i: Индекс записи в JSON-файле
        data_folder: Путь к папке с данными
        debug_mode: Включить режим отладки (сохранение дополнительной информации)
    
    Returns:
        dict: Результат проверки с подробной информацией
    """
    startTime = time.time()
    images = {}
    result = ""
    results_dict = {
        "success": True, 
        "message": "", 
        "found_items": [], 
        "not_found_items": [], 
        "errors": [],
        "debug_info": {}
    }

    # Если путь к папке не указан, используем различные варианты путей
    if data_folder is None:
        # Пробуем определить путь по умолчанию
        default_paths = [
            r'C:\Users\FreeTok\Desktop\data',
            r'D:\Hackatons\WB_Hack\data',
            'data',  # Относительный путь
            './data'  # Еще один относительный путь
        ]
        
        # Проверяем каждый путь и используем первый существующий
        for path in default_paths:
            if os.path.exists(path):
                data_folder = path
                break
        
        if data_folder is None:
            error_msg = "ОШИБКА: Не удалось найти папку с данными. Укажите путь явно."
            results_dict["success"] = False
            results_dict["message"] = error_msg
            results_dict["errors"].append(error_msg)
            return results_dict
    
    # Проверяем существование папки
    if not os.path.exists(data_folder):
        error_msg = f"ОШИБКА: Папка с данными {data_folder} не найдена."
        results_dict["success"] = False
        results_dict["message"] = error_msg
        results_dict["errors"].append(error_msg)
        return results_dict
    
    # Путь к JSON-файлу
    json_path = os.path.join(data_folder, "videos.json")
    if not os.path.exists(json_path):
        error_msg = f"ОШИБКА: Файл videos.json не найден в папке {data_folder}."
        results_dict["success"] = False
        results_dict["message"] = error_msg
        results_dict["errors"].append(error_msg)
        return results_dict
    
    try:
        import json
        
        # Читаем JSON-файл с данными
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)
            
            # Проверяем, что индекс в пределах диапазона
            if i >= len(data):
                error_msg = f"ОШИБКА: Индекс {i} выходит за пределы данных (всего {len(data)} записей)"
                results_dict["success"] = False
                results_dict["message"] = error_msg
                results_dict["errors"].append(error_msg)
                return results_dict
            
            d = data[i]
            result += f"Обрабатываю запись: {d}\n"
            results_dict["record"] = d
            
            video_url = d['path_url']
            
            # Создаем временную папку, если её нет
            os.makedirs('temp', exist_ok=True)
            
            # Создаём папку debug для сохранения отладочных данных
            if debug_mode:
                os.makedirs('debug_images', exist_ok=True)
            
            # Используем уникальное имя для видео (с индексом)
            output_video = os.path.join('temp', f'output_video_{i}.mp4')
            
            # Скачиваем видео с помощью улучшенной функции
            download_success = download_video(video_url, output_video)
            
            if not download_success:
                error_msg = f"ОШИБКА: Не удалось скачать видео {video_url}."
                results_dict["success"] = False
                results_dict["message"] = error_msg
                results_dict["errors"].append(error_msg)
                return results_dict
            
            # Собираем изображения товаров
            result += "Анализ товаров:\n"
            
            # Создаём словарь для хранения информации о товарах
            product_info = {}
            
            for id in d['nm_ids']:
                item_result = {"id": id, "found": False, "timecodes": [], "checked_images": []}
                smallImages = []
                
                # Формируем пути к изображениям товара
                image_folders = {
                    "1": os.path.join(data_folder, f"images_1/{id}"),
                    "2-3": os.path.join(data_folder, f"images_2-3/{id}"),
                    "4-5": os.path.join(data_folder, f"images_4-5/{id}")
                }
                
                # Проверяем все возможные пути к изображениям
                for folder_name, folder_path in image_folders.items():
                    if os.path.exists(folder_path):
                        # Получаем список всех файлов изображений в папке
                        for img_file in os.listdir(folder_path):
                            if img_file.endswith(('.webp', '.jpg', '.jpeg', '.png')):
                                img_path = os.path.join(folder_path, img_file)
                                smallImages.append(img_path)
                                item_result["checked_images"].append(f"{folder_name}/{img_file}")
                
                # Если нет изображений, добавляем товар в список не найденных
                if not smallImages:
                    item_result["error"] = f"Не найдены изображения для товара {id}"
                    results_dict["not_found_items"].append(item_result)
                    result += f"  ВНИМАНИЕ: Не найдены изображения для товара {id}!\n"
                    continue
                
                # Добавляем информацию о товаре (используется для определения категории)
                product_info[id] = {"id": id, "name": d.get("name", "")}
                
                images.update({id: smallImages})
        
        # Выполняем сравнение для каждого товара с улучшенной функцией
        for imageid in images:
            result += f"Проверка товара {imageid}:\n"
            
            # Вызываем улучшенную функцию сравнения
            match_result = improved_compare_video_image(
                images[imageid], 
                output_video, 
                product_info=product_info.get(imageid, None),
                debug_save=debug_mode,
                frame_rate=FRAME_RATE
            )
            
            # Распаковываем результаты
            is_found, timecodes, max_similarity, debug_data = match_result
            
            # Сохраняем информацию о товаре
            item_result = {
                "id": imageid, 
                "found": is_found, 
                "timecodes": timecodes, 
                "max_similarity": float(max_similarity),
                "checked_images": [Path(p).name for p in images[imageid]]
            }
            
            # Добавляем отладочную информацию при необходимости
            if debug_mode:
                item_result["debug"] = debug_data
            
            if is_found:
                results_dict["found_items"].append(item_result)
                result += f"  + Найдено совпадение в видео! Таймкоды: {timecodes}\n"
                result += f"    Максимальное сходство: {max_similarity:.4f}\n"
            else:
                item_result["error"] = f"Товар {imageid} не найден в видео"
                results_dict["not_found_items"].append(item_result)
                result += f"  - ВНИМАНИЕ: Товар {imageid} не найден в видео!\n"
                result += f"    Максимальное сходство: {max_similarity:.4f} (ниже порога)\n"
        
        # Сохраняем путь к видео в результат для дальнейшей визуализации
        results_dict["video_path"] = output_video
        result += f"Видео сохранено: {output_video}\n"

        # Сохраняем пути к изображениям товаров для визуализации
        for imageid in images:
            for item in results_dict["found_items"] + results_dict["not_found_items"]:
                if item["id"] == imageid:
                    item["image_paths"] = images[imageid]
                    break
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        error_msg = f"Произошла ошибка при обработке:\n{str(e)}\n\n{error_trace}"
        
        results_dict["success"] = False
        results_dict["message"] = str(e)
        results_dict["errors"].append(error_msg)
        result += error_msg
    
    # Формируем итоговый результат
    elapsed_time = time.time() - startTime
    result += f"\nОбработка завершена за {elapsed_time:.2f} секунд"
    
    results_dict["processing_time"] = elapsed_time
    results_dict["raw_log"] = result
    
    return results_dict

# Функция для распознавания сложных товаров
def recognize_complex_product(product_image, video_path, product_category=None):
    """
    Специализированная функция для распознавания сложных товаров (одежда, мелкие предметы и т.д.)
    
    Args:
        product_image: Путь к изображению товара
        video_path: Путь к видео
        product_category: Категория товара для настройки параметров
        
    Returns:
        tuple: (found, timecodes, confidence)
    """
    try:
        # Загружаем изображение товара
        product_img = Image.open(product_image)
        product_img = preprocess_image(product_img)
        
        # Определяем порог сходства в зависимости от категории
        threshold = PRODUCT_CATEGORIES.get(product_category, SIMILARITY_THRESHOLD_DEFAULT)
        
        # Применяем дополнительные улучшения для определенных категорий
        if product_category == "одежда":
            # Для одежды важны текстуры и цвета, увеличиваем контраст
            enhancer = ImageEnhance.Contrast(product_img)
            product_img = enhancer.enhance(1.3)
            
            # Снижаем порог сходства для одежды
            threshold = 0.6
            
        elif product_category == "косметика":
            # Для косметики важны бренды и логотипы, усиливаем насыщенность
            enhancer = ImageEnhance.Color(product_img)
            product_img = enhancer.enhance(1.2)
            
            # Увеличиваем порог сходства для косметики (бренд должен четко распознаваться)
            threshold = 0.75
        
        # Извлекаем кадры из видео с более высокой частотой для поиска мелких товаров
        processed_frames = extract_and_process_frames(video_path, frame_rate=3)
        
        if not processed_frames:
            return (False, [], 0.0)
        
        # Обрабатываем изображение товара с YOLO
        yolo_model = get_yolo_model()
        results = yolo_model(np.array(product_img), verbose=False)
        
        # Извлекаем обнаруженные объекты
        detections, confidence_scores, _ = get_detections(results)
        
        # Если объекты не найдены, используем всё изображение
        if not detections:
            cropped_objects = [product_img]
            confidence_scores = [0.9]
        else:
            # Вырезаем обнаруженные объекты
            cropped_objects = crop_objects(product_img, detections)
            
            # Если не удалось вырезать объекты, используем исходное изображение
            if not cropped_objects:
                cropped_objects = [product_img]
                confidence_scores = [0.9]
        
        # Извлекаем векторы признаков
        product_features = [get_feature_vector(img) for img in cropped_objects]
        product_features = [f for f in product_features if f is not None]
        
        # Массивы для хранения результатов
        timecodes = []
        max_similarity = 0.0
        
        # Сравниваем с каждым кадром
        for frame in processed_frames:
            is_match, similarity = compare_features(
                product_features, 
                frame.feature_vectors,
                threshold=threshold,
                confidence_scores=frame.confidence_scores
            )
            
            max_similarity = max(max_similarity, similarity)
            
            if is_match:
                timecodes.append(frame.time_code)
        
        # Удаляем дубликаты и сортируем таймкоды
        if timecodes:
            timecodes = sorted(list(set(timecodes)))
            return (True, timecodes, max_similarity)
        else:
            return (False, [], max_similarity)
            
    except Exception as e:
        print(f"Ошибка при распознавании сложного товара: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return (False, [], 0.0)

# Для тестирования
if __name__ == "__main__":
    print("Улучшенная система для проверки товаров в видео")
    
    # Проверяем аргументы командной строки
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Проверка наличия товаров в видео')
    parser.add_argument('--index', type=int, default=0, help='Индекс записи в JSON-файле')
    parser.add_argument('--data', type=str, help='Путь к папке с данными')
    parser.add_argument('--debug', action='store_true', help='Включить режим отладки')
    parser.add_argument('--category', type=str, help='Категория товара (одежда, электроника, косметика)')
    parser.add_argument('--threshold', type=float, help='Пользовательский порог сходства (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Устанавливаем пользовательский порог сходства, если указан
    if args.threshold is not None and 0.0 <= args.threshold <= 1.0:
        SIMILARITY_THRESHOLD_DEFAULT = args.threshold
        print(f"Установлен пользовательский порог сходства: {SIMILARITY_THRESHOLD_DEFAULT}")
    
    # Запускаем проверку с переданными аргументами
    result = optimized_check(args.index, args.data, debug_mode=args.debug)
    print(result["raw_log"])
    
    # Сохраняем результаты в JSON-файл
    os.makedirs('results', exist_ok=True)
    with open(f'results/result_{args.index}.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)