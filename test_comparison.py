import os
from ultralytics import YOLO

import torch
from torchvision import transforms
import torchvision.models as models
from sklearn.metrics.pairwise import cosine_similarity

import cv2
from PIL import Image

# Глобальные переменные для моделей
global_model = None
global_model_1 = None

# Функция для получения экземпляра модели
def get_yolo_model():
    global global_model
    if global_model is None:
        # Загружаем модель, если она еще не загружена
        global_model = YOLO("yolo11n.pt", verbose=False)
    return global_model

# Функция для получения экземпляра модели извлечения признаков
def get_feature_model():
    global global_model_1
    if global_model_1 is None:
        # Загружаем модель, если она еще не загружена
        global_model_1 = models.efficientnet_b0(pretrained=True)
        global_model_1 = torch.nn.Sequential(*list(global_model_1.children())[:-1])
        global_model_1.eval()
        
        # Преобразование изображений для EfficientNet
        global transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return global_model_1

# Преобразование изображений для EfficientNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_detections(results):
    if not results or len(results) == 0: return None

    res = []
    for result in results[0]:
        res.append(result.boxes.xyxy.tolist()[0])
    
    return res

# Функция для извлечения объектов
def crop_objects(image, detections):
    global index
    if len(detections) == 0: return None

    cropped_images = []
    for detection in detections:
        xmin, ymin, xmax, ymax = [int(coord) for coord in detection]  # Ensure coordinates are integers
        try:
            cropped_image = image.crop((xmin, ymin, xmax, ymax))
            if cropped_image.size[0] > 0 and cropped_image.size[1] > 0:  # Check if not empty
                cropped_images.append(cropped_image)
                index += 1
        except:
            continue

    return cropped_images if cropped_images else None

# Функция для извлечения feature vectors
def get_feature_vector(image):
    # Получаем модель
    model_1 = get_feature_model()
    
    # Convert RGBA to RGB if needed
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model_1(image)
    return features.squeeze().numpy()

# Функция для сравнения feature vectors
def compare_features(features1, features2, threshold=0.7):
    for i, f1 in enumerate(features1):
        for j, f2 in enumerate(features2):
            similarity = cosine_similarity(f1.reshape(1, -1), f2.reshape(1, -1))
            if similarity > threshold:
                print(f"Объект {i} на изображении 1 и объект {j} на изображении 2 схожи! Сходство: {similarity}")
                return True
    return False

def compare_images(img1, img2, debug_save=False):
    if not img1 or not img2: return False

    # Получаем модель YOLO
    model = get_yolo_model()
    
    # Convert to RGB if needed
    if img1.mode != 'RGB':
        img1 = img1.convert('RGB')
    if img2.mode != 'RGB':
        img2 = img2.convert('RGB')
    
    # Predict with the model
    results1 = model(img1, verbose=False)  # predict on an image
    results2 = model(img2, verbose=False)  # predict on an image

    if len(results1) == 0 or len(results2) == 0: return False

    detections1 = get_detections(results1)
    detections2 = get_detections(results2)

    if detections1 is None or detections2 is None: return False
    if len(detections1) == 0 or len(detections2) == 0: return False

    # Извлечение объектов
    cropped_images1 = crop_objects(img1, detections1)
    cropped_images2 = crop_objects(img2, detections2)

    if cropped_images1 is None or cropped_images2 is None: return False
    if len(cropped_images1) == 0 or len(cropped_images2) == 0: return False

    # Извлечение feature vectors
    features1 = [get_feature_vector(img) for img in cropped_images1]
    features2 = [get_feature_vector(img) for img in cropped_images2]

    if len(features1) == 0 or len(features2) == 0: return False

    # Сохранение изображений для отладки, если требуется
    if debug_save:
        global index
        # Создаем папку для отладочных изображений, если её нет
        os.makedirs('test', exist_ok=True)
        
        # Проверяем, есть ли совпадение
        if compare_features(features1, features2):
            cropped_images1[0].save(f'test/frame_{index}.jpg')
            index += 1
            cropped_images2[0].save(f'test/frame_{index}.jpg')
            index += 1

    # Сравнение объектов
    return compare_features(features1, features2)

# Глобальная переменная для нумерации сохраняемых кадров
index = 0

def compare_video_image(imgPath, capPath, debug_save=False):
    """
    Сравнивает изображение с кадрами видео.
    
    Args:
        imgPath (str): Путь к изображению товара
        capPath (str): Путь к видео-файлу
        debug_save (bool): Флаг для сохранения отладочных изображений
        
    Returns:
        tuple: (matched, timecodes) - нашлось ли совпадение и на каких таймкодах
    """
    global index
    
    if not os.path.exists(imgPath) or not os.path.exists(capPath):
        print(f"Ошибка: файл не существует - {imgPath if not os.path.exists(imgPath) else capPath}")
        return None
    
    try:
        img = Image.open(imgPath)
        cap = cv2.VideoCapture(capPath)

        matched = False
        timecodes = []
        # Параметры для извлечения кадров
        frame_rate = 1  # 1 кадр в секунду
        frame_count = 0
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # Получаем FPS видео

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Извлечение кадров с заданной частотой
            if frame_count % (fps * frame_rate) == 0:
                # Преобразование кадра в PIL Image
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if compare_images(img, frame_pil, debug_save):
                    matched = True
                    timecodes.append(frame_count / fps)

            frame_count += 1

        cap.release()

        return (matched, timecodes)
    except Exception as e:
        import traceback
        print(f"Ошибка при сравнении видео и изображения: {str(e)}")
        print(traceback.format_exc())
        return None