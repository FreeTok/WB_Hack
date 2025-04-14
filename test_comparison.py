import os
import torch
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
from torchvision import transforms
import torchvision.models as models
from sklearn.metrics.pairwise import cosine_similarity
import time

global_yolo_model = None
global_feature_model = None
global_transform = None

THRESHOLD = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")

def get_yolo_model():
    global global_yolo_model
    if global_yolo_model is None:
        try:
            print(f"Загрузка модели YOLO на {device}...")
            
            global_yolo_model = YOLO("yolo11n.pt", verbose=False)
            
            if device.type == 'cuda':
                if hasattr(global_yolo_model, 'to'):
                    global_yolo_model.to(device)
                print("Модель YOLO использует GPU")
            
            print("Модель YOLO загружена успешно!")
        except Exception as e:
            print(f"Ошибка при загрузке модели YOLO: {str(e)}")
            import traceback
            print(traceback.format_exc())
            global_yolo_model = YOLO("yolo11n.pt", verbose=False)
    
    return global_yolo_model

def get_feature_model():
    global global_feature_model, global_transform
    if global_feature_model is None:
        try:
            print(f"Загрузка модели EfficientNet на {device}...")
            global_feature_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            global_feature_model = torch.nn.Sequential(*list(global_feature_model.children())[:-1])
            global_feature_model = global_feature_model.to(device)
            global_feature_model.eval()
            
            global_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            print("Модель EfficientNet загружена успешно!")
        except Exception as e:
            print(f"Ошибка при загрузке EfficientNet: {str(e)}")
            try:
                print("Попытка загрузить MobileNet...")
                global_feature_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
                global_feature_model = torch.nn.Sequential(*list(global_feature_model.children())[:-1])
                global_feature_model = global_feature_model.to(device)
                global_feature_model.eval()
                print("Модель MobileNet загружена успешно!")
            except Exception as e2:
                print(f"Ошибка при загрузке MobileNet: {str(e2)}")
                global_feature_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
                global_feature_model = torch.nn.Sequential(*list(global_feature_model.children())[:-1])
                global_feature_model = global_feature_model.to('cpu')
                global_feature_model.eval()
                print("Модель загружена на CPU вместо GPU")
    
    return global_feature_model

def get_transform():
    global global_transform
    if global_transform is None:
        global_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return global_transform

class ProcessedFrame:
    def __init__(self, frame_index, time_code, detected_objects=None, feature_vectors=None):
        self.frame_index = frame_index
        self.time_code = time_code
        self.detected_objects = detected_objects or []
        self.feature_vectors = feature_vectors or []

def get_detections(results):
    try:
        if not results or len(results) == 0: 
            return None

        detections = []
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    if box.xyxy is not None and len(box.xyxy) > 0:
                        if torch.is_tensor(box.xyxy):
                            box_coords = box.xyxy.cpu().tolist()[0]
                        else:
                            box_coords = box.xyxy[0]
                        detections.append(box_coords)
        
        return detections if detections else None
    except Exception as e:
        print(f"Ошибка в get_detections: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def crop_objects(image, detections):
    if not detections: 
        return None

    cropped_images = []
    for detection in detections:
        try:
            xmin, ymin, xmax, ymax = [int(coord) for coord in detection]
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(image.width, xmax), min(image.height, ymax)
            
            if xmax > xmin and ymax > ymin:  
                cropped_image = image.crop((xmin, ymin, xmax, ymax))
                if cropped_image.size[0] > 0 and cropped_image.size[1] > 0:
                    cropped_images.append(cropped_image)
        except Exception as e:
            print(f"Ошибка при кадрировании: {str(e)}")
            continue

    return cropped_images if cropped_images else None

def get_feature_vector(image):
    try:
        model = get_feature_model()
        transform = get_transform()
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = model(image_tensor)
            features = features.cpu()
        
        return features.squeeze().numpy()
    except Exception as e:
        print(f"Ошибка в get_feature_vector: {str(e)}")
        return None

def compare_features(features1, features2, threshold=THRESHOLD):
    try:
        for i, f1 in enumerate(features1):
            for j, f2 in enumerate(features2):
                similarity = cosine_similarity(f1.reshape(1, -1), f2.reshape(1, -1))
                if similarity > threshold:
                    print(f"Найдено сходство: {similarity[0][0]}")
                    return True
        return False
    except Exception as e:
        print(f"Ошибка в compare_features: {str(e)}")
        return False

def extract_and_process_frames(video_path, frame_rate=1):
    """
    Извлекает кадры из видео с заданной частотой и предварительно обрабатывает их.
    
    Args:
        video_path (str): Путь к видео-файлу
        frame_rate (int): Частота извлечения кадров (1 = 1 кадр в секунду)
        
    Returns:
        list: Список предварительно обработанных кадров
    """
    try:
        if not os.path.exists(video_path):
            print(f"Ошибка: Файл видео не существует - {video_path}")
            return []
        
        yolo_model = get_yolo_model()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Ошибка: Не удалось открыть видео - {video_path}")
            return []
        
        processed_frames = []
        frame_count = 0
        fps = int(cap.get(cv2.CAP_PROP_FPS)) if int(cap.get(cv2.CAP_PROP_FPS)) > 0 else 30
        
        
        print(f"Извлечение и обработка кадров из видео: {video_path}")
        print(f"Частота кадров: {fps}, частота извлечения: {frame_rate}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % (fps * frame_rate) == 0:
                time_code = frame_count / fps
                print(f"Обработка кадра {frame_count} (таймкод: {time_code:.2f})")
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                try:
                    if device.type == 'cuda':
                        with torch.amp.autocast('cuda'):
                            results = yolo_model(frame_rgb, verbose=False)
                    else:
                        results = yolo_model(frame_rgb, verbose=False)
                    
                    detections = get_detections(results)
                    
                    if detections:
                        cropped_objects = crop_objects(frame_pil, detections)
                        
                        if cropped_objects:
                            features = [get_feature_vector(img) for img in cropped_objects]
                            features = [f for f in features if f is not None]
                            
                            if features:
                                processed_frame = ProcessedFrame(
                                    frame_index=frame_count,
                                    time_code=time_code,
                                    detected_objects=cropped_objects,
                                    feature_vectors=features
                                )
                                processed_frames.append(processed_frame)
                except Exception as e:
                    print(f"Ошибка при обработке кадра {frame_count}: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
            
            frame_count += 1
            
        cap.release()
        
        print(f"Обработано {len(processed_frames)} кадров из видео")
        return processed_frames
    
    except Exception as e:
        import traceback
        print(f"Ошибка при извлечении и обработке кадров: {str(e)}")
        print(traceback.format_exc())
        return []

def optimized_compare_video_image(product_images, video_path, debug_save=False):
    """
    Оптимизированная функция проверки наличия товара в видео.
    
    Args:
        product_images (list): Список путей к изображениям товара
        video_path (str): Путь к видео-файлу
        debug_save (bool): Сохранять ли отладочные изображения
        
    Returns:
        tuple: (matched, timecodes) - нашлось ли совпадение и на каких таймкодах
    """
    try:
        if not product_images or not video_path:
            print("Не указаны пути к изображениям товара или видео")
            return (False, [])
        
        for img_path in product_images:
            if not os.path.exists(img_path):
                print(f"Ошибка: Файл изображения не существует - {img_path}")
                return (False, [])
        
        if not os.path.exists(video_path):
            print(f"Ошибка: Файл видео не существует - {video_path}")
            return (False, [])
        
        processed_frames = extract_and_process_frames(video_path)
        
        if not processed_frames:
            print("Не удалось извлечь кадры из видео или не найдены объекты")
            return (False, [])
        
        matched = False
        timecodes = []
        
       
        for img_path in product_images:
            print(f"Проверка изображения товара: {img_path}")
            
            try:
                product_img = Image.open(img_path)
                if product_img.mode != 'RGB':
                    product_img = product_img.convert('RGB')
                
                if device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        results = get_yolo_model()(np.array(product_img), verbose=False)
                else:
                    results = get_yolo_model()(np.array(product_img), verbose=False)
                
                
                detections = get_detections(results)
                
                if not detections:
                    print(f"На изображении {img_path} не обнаружены объекты")
                    continue
                
                cropped_objects = crop_objects(product_img, detections)
                
                if not cropped_objects:
                    print(f"Не удалось вырезать объекты из изображения {img_path}")
                    continue
                
                product_features = [get_feature_vector(img) for img in cropped_objects]
                
                product_features = [f for f in product_features if f is not None]
                
                if not product_features:
                    print(f"Не удалось извлечь признаки из изображения {img_path}")
                    continue
                
                for frame in processed_frames:
                    if compare_features(product_features, frame.feature_vectors):
                        matched = True
                        timecodes.append(frame.time_code)
                        
                        if debug_save:
                            os.makedirs('test', exist_ok=True)
                            cropped_objects[0].save(f'test/product_{os.path.basename(img_path)}')
                            frame.detected_objects[0].save(f'test/frame_{frame.time_code:.2f}.jpg')
                
                if matched:
                    print(f"Найдено совпадение для изображения {img_path}")
                    break
                
            except Exception as e:
                print(f"Ошибка при обработке изображения {img_path}: {str(e)}")
                import traceback
                print(traceback.format_exc())
        
        if matched and timecodes:
            timecodes = sorted(list(set(timecodes)))
        
        return (matched, timecodes)
        
    except Exception as e:
        import traceback
        print(f"Ошибка при сравнении видео и изображения: {str(e)}")
        print(traceback.format_exc())
        return (False, [])

def optimized_check(i, data_folder=None):
    """
    Оптимизированная функция проверки наличия товара в видео.
    
    Args:
        i (int): Индекс записи в JSON-файле
        data_folder (str, optional): Путь к папке с данными
    
    Returns:
        dict: Результат проверки с подробной информацией
    """
    startTime = time.time()
    images = {}
    result = ""
    results_dict = {"success": True, "message": "", "found_items": [], "not_found_items": [], "errors": []}

    if data_folder is None:
        default_paths = [
            r'C:\Users\FreeTok\Desktop\data',
            r'D:\Hackatons\WB_Hack\data',
            'data'  
        ]
        
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
    
    if not os.path.exists(data_folder):
        error_msg = f"ОШИБКА: Папка с данными {data_folder} не найдена."
        results_dict["success"] = False
        results_dict["message"] = error_msg
        results_dict["errors"].append(error_msg)
        return results_dict
    
    json_path = os.path.join(data_folder, "videos.json")
    if not os.path.exists(json_path):
        error_msg = f"ОШИБКА: Файл videos.json не найден в папке {data_folder}."
        results_dict["success"] = False
        results_dict["message"] = error_msg
        results_dict["errors"].append(error_msg)
        return results_dict
    
    try:
        import json
        import subprocess
        
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)
            
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
            
            os.makedirs('temp', exist_ok=True)
            
            os.makedirs('test', exist_ok=True)
            
            output_video = os.path.join('temp', f'output_video_{i}.mp4')
            
            ffmpeg_path = 'ffmpeg'
            for possible_path in ['./Ffmpeg/bin/ffmpeg.exe', './ffmpeg.exe', 'ffmpeg.exe']:
                if os.path.exists(possible_path):
                    ffmpeg_path = possible_path
                    break
            
            if os.path.exists(output_video):
                file_size = os.path.getsize(output_video)
                if file_size > 1024:  
                    result += f"Используем существующее видео {output_video}...\n"
                else:
                    os.remove(output_video)
                    result += f"Скачиваю видео {video_url}...\n"
                    command = [
                        ffmpeg_path,
                        '-i', video_url,
                        '-c', 'copy',
                        '-v', 'error',
                        output_video
                    ]
                    subprocess.run(command)
            else:
                result += f"Скачиваю видео {video_url}...\n"
                command = [
                    ffmpeg_path,
                    '-i', video_url,
                    '-c', 'copy',
                    '-v', 'error',
                    output_video
                ]
                subprocess.run(command)
            
            if not os.path.exists(output_video) or os.path.getsize(output_video) < 1024:
                error_msg = f"ОШИБКА: Не удалось скачать видео {video_url}. Проверьте, установлен ли ffmpeg."
                results_dict["success"] = False
                results_dict["message"] = error_msg
                results_dict["errors"].append(error_msg)
                return results_dict
            
            result += "Анализ товаров:\n"
            for id in d['nm_ids']:
                item_result = {"id": id, "found": False, "timecodes": [], "checked_images": []}
                smallImages = []
                
                image_paths = [
                    os.path.join(data_folder, f"images_1/{id}/1.webp"),
                    os.path.join(data_folder, f"images_2-3/{id}/2.webp"),
                    os.path.join(data_folder, f"images_2-3/{id}/3.webp"),
                    os.path.join(data_folder, f"images_4-5/{id}/4.webp"),
                    os.path.join(data_folder, f"images_4-5/{id}/5.webp")
                ]
                
                for path in image_paths:
                    if os.path.exists(path):
                        smallImages.append(path)
                        item_result["checked_images"].append(os.path.basename(path))
                
                if not smallImages:
                    item_result["error"] = f"Не найдены изображения для товара {id}"
                    results_dict["not_found_items"].append(item_result)
                    result += f"  ВНИМАНИЕ: Не найдены изображения для товара {id}!\n"
                    continue
                
                images.update({id: smallImages})
        
        for imageid in images:
            result += f"Проверка товара {imageid}:\n"
            
            match_result = optimized_compare_video_image(images[imageid], output_video, debug_save=False)
            
            item_result = {"id": imageid, "found": False, "timecodes": [], "checked_images": [p.split('/')[-1] for p in images[imageid]]}
            
            if match_result and match_result[0]:
                item_result["found"] = True
                item_result["timecodes"] = match_result[1]
                results_dict["found_items"].append(item_result)
                result += f"  - Найдено совпадение в видео! Таймкоды: {match_result[1]}\n"
            else:
                item_result["error"] = f"Товар {imageid} не найден в видео"
                results_dict["not_found_items"].append(item_result)
                result += f"  ВНИМАНИЕ: Товар {imageid} не найден в видео!\n"
        
        results_dict["video_path"] = output_video
        result += f"Видео сохранено: {output_video}\n"

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
    
    elapsed_time = time.time() - startTime
    result += f"\nОбработка завершена за {elapsed_time:.2f} секунд"
    
    results_dict["processing_time"] = elapsed_time
    results_dict["raw_log"] = result
    
    return results_dict

if __name__ == "__main__":
    print("Оптимизированная система для проверки товаров в видео")
    
    import sys
    if len(sys.argv) > 1:
        index = int(sys.argv[1])
        data_folder = None
        if len(sys.argv) > 2:
            data_folder = sys.argv[2]
        
        result = optimized_check(index, data_folder)
        print(result["raw_log"])
    else:
        result = optimized_check(0)
        print(result["raw_log"])


def expert_custom_check(data_folder):
    """
    Функция для кастомной экспертной проверки.
    Args:
        data_folder (str): Путь к папке с данными
    """
    # Вы можете заполнить эту функцию своей логикой
    # Нужно вернуть то, что будет отображено в интерфейсе
    
    return "Функция expert_custom_check не реализована. Реализуйте её в файле test_comparison.py."