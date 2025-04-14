import threading
import time
import torch
from test_comparison import get_yolo_model, get_feature_model, get_transform

models_loaded = False
loading_status = "Загрузка моделей..."
loading_error = None

total_models = 2 
loaded_models = 0  

def preload_models():
    """
    Функция для предварительной загрузки всех необходимых моделей.
    Запускается в отдельном потоке.
    """
    global models_loaded, loading_status, loading_error, loaded_models
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            device_count = torch.cuda.device_count()
            cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "неизвестно"
            
            props = torch.cuda.get_device_properties(0)
            total_memory = props.total_memory / 1024 / 1024 / 1024  # В ГБ
            
            loading_status = f"CUDA доступна. Устройств: {device_count}, версия: {cuda_version}, память: {total_memory:.2f} ГБ"
            print(loading_status)
        else:
            loading_status = "CUDA недоступна. Используется CPU."
            print(loading_status)
        
        loading_status = "Загрузка модели YOLO..."
        print(loading_status)
        
        yolo_model = get_yolo_model()
        loaded_models += 1
        
        if yolo_model is not None:
            dummy_input = torch.zeros(1, 3, 640, 640)
            if torch.cuda.is_available():
                dummy_input = dummy_input.to('cuda')
                with torch.amp.autocast('cuda'):
                    with torch.no_grad():
                        _ = yolo_model(dummy_input, verbose=False)
            else:
                with torch.no_grad():
                    _ = yolo_model(dummy_input, verbose=False)
            print("Модель YOLO прогрета")
        
        loading_status = "Загрузка модели EfficientNet..."
        print(loading_status)
        
        feature_model = get_feature_model()
        
        if feature_model is not None:
            dummy_input = torch.zeros(1, 3, 224, 224)
            if torch.cuda.is_available():
                dummy_input = dummy_input.to('cuda')
                with torch.amp.autocast('cuda'):
                    with torch.no_grad():
                        _ = feature_model(dummy_input)
            else:
                with torch.no_grad():
                    _ = feature_model(dummy_input)
            print("Модель EfficientNet прогрета")
        
        transform = get_transform()
        loaded_models += 1
        
        loading_status = "Все модели загружены успешно и готовы к использованию!"
        print(loading_status)
        
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
    load_percentage = 0
    if total_models > 0:
        load_percentage = int((loaded_models / total_models) * 100)
    
    gpu_info = ""
    if torch.cuda.is_available() and not models_loaded and not loading_error:
        try:
            allocated = torch.cuda.memory_allocated(0) / 1024 / 1024  # МБ
            reserved = torch.cuda.memory_reserved(0) / 1024 / 1024    # МБ
            gpu_info = f" (GPU: {allocated:.0f}МБ/{reserved:.0f}МБ)"
        except:
            pass
    
    return {
        "loaded": models_loaded,
        "status": loading_status + gpu_info,
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

if __name__ == "__main__":
    print("Запуск предварительной загрузки моделей...")
    thread = start_preloading()
    
    while thread.is_alive():
        status = get_loading_status()
        print(f"Прогресс: {status['progress']}% - {status['status']}")
        time.sleep(1)
    
    print("Загрузка моделей завершена!")