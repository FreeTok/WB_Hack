import threading
import time
import sys
from test_comparison import get_yolo_model, get_feature_model, get_transform

# Глобальные переменные для отслеживания статуса загрузки моделей
models_loaded = False
loading_status = "Загрузка моделей..."
loading_error = None

# Переменные для отслеживания прогресса загрузки
total_models = 2  # Общее количество моделей для загрузки
loaded_models = 0  # Количество загруженных на данный момент моделей

def preload_models():
    """
    Функция для предварительной загрузки всех необходимых моделей.
    Запускается в отдельном потоке.
    """
    global models_loaded, loading_status, loading_error, loaded_models
    
    try:
        # Обновляем статус
        loading_status = "Загрузка модели YOLO..."
        print("Загрузка модели YOLO...")
        
        # Загружаем YOLO модель - функция get_yolo_model() загружает ее один раз
        # и сохраняет в глобальную переменную
        yolo_model = get_yolo_model()
        
        # Увеличиваем счетчик загруженных моделей
        loaded_models += 1
        
        # Обновляем статус
        loading_status = "Загрузка модели EfficientNet..."
        print("Загрузка модели EfficientNet...")
        
        # Загружаем модель EfficientNet
        feature_model = get_feature_model()
        
        # Убедимся, что трансформация также инициализирована
        transform = get_transform()
        
        # Увеличиваем счетчик загруженных моделей
        loaded_models += 1
        
        # Здесь можно добавить загрузку других моделей в будущем
        # Например:
        # 
        # loading_status = "Загрузка модели ResNet..."
        # resnet_model = get_resnet_model()  # Функция, которую нужно будет создать
        # loaded_models += 1
        # 
        # loading_status = "Загрузка модели MobileNet..."
        # mobilenet_model = get_mobilenet_model()  # Функция, которую нужно будет создать
        # loaded_models += 1
        
        # Обновляем статус
        loading_status = "Все модели загружены успешно!"
        print("Все модели загружены успешно!")
        
        # Устанавливаем флаг, что модели загружены
        models_loaded = True
        
    except Exception as e:
        import traceback
        error_msg = f"Ошибка при загрузке моделей: {str(e)}"
        loading_error = f"{error_msg}\n{traceback.format_exc()}"
        loading_status = error_msg
        print(error_msg)
        print(traceback.format_exc())

def start_preloading():
    """
    Запускает предварительную загрузку моделей в отдельном потоке.
    """
    preload_thread = threading.Thread(target=preload_models)
    preload_thread.daemon = True
    preload_thread.start()
    return preload_thread

def get_loading_status():
    """
    Возвращает текущий статус загрузки моделей.
    """
    # Вычисляем процент загрузки
    load_percentage = 0
    if total_models > 0:
        load_percentage = int((loaded_models / total_models) * 100)
    
    return {
        "loaded": models_loaded,
        "status": loading_status,
        "error": loading_error,
        "progress": load_percentage,
        "current": loaded_models,
        "total": total_models
    }

def add_model_to_preload():
    """
    Добавляет новую модель в список загрузки.
    Эта функция должна вызываться перед началом загрузки моделей.
    """
    global total_models
    total_models += 1