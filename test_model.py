import json
import os
import subprocess
import time
import sys

from test_comparison import compare_video_image

def check(i, data_folder=None):
    """
    Проверяет наличие товара в видео.
    
    Args:
        i (int): Индекс записи в JSON-файле
        data_folder (str, optional): Путь к папке с данными. По умолчанию используется путь из исходного кода.
    
    Returns:
        str: Результат проверки
    """
    startTime = time.time()
    images = {}
    result = ""

    # Если путь к папке не указан, используем значение по умолчанию
    if data_folder is None:
        data_folder = r'C:\Users\FreeTok\Desktop\data'
    
    # Проверяем существование папки
    if not os.path.exists(data_folder):
        return f"ОШИБКА: Папка с данными {data_folder} не найдена."
    
    # Путь к JSON-файлу
    json_path = os.path.join(data_folder, "videos.json")
    if not os.path.exists(json_path):
        return f"ОШИБКА: Файл videos.json не найден в папке {data_folder}."
    
    try:
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)
            
            # Проверяем, что индекс в пределах диапазона
            if i >= len(data):
                return f"ОШИБКА: Индекс {i} выходит за пределы данных (всего {len(data)} записей)"
            
            d = data[i]
            result += f"Обрабатываю запись: {d}\n"
            
            video_url = d['path_url']
            
            # Создаем временную папку, если её нет
            os.makedirs('temp', exist_ok=True)
            output_video = 'temp/output_video.mp4'
            
            # Скачиваем видео с помощью ffmpeg
            command = [
                'ffmpeg',
                '-i', video_url,
                '-c', 'copy',
                '-v', 'error',
                output_video
            ]
            result += f"Скачиваю видео {video_url}...\n"
            subprocess.run(command)
            
            # Обработка изображений товаров
            result += "Анализ товаров:\n"
            for id in d['nm_ids']:
                smallImages = []
                
                # Путь к изображениям относительно папки с данными
                smallImages.append(os.path.join(data_folder, f"images_1/{id}/1.webp"))
                smallImages.append(os.path.join(data_folder, f"images_2-3/{id}/2.webp"))
                smallImages.append(os.path.join(data_folder, f"images_2-3/{id}/3.webp"))
                smallImages.append(os.path.join(data_folder, f"images_4-5/{id}/4.webp"))
                smallImages.append(os.path.join(data_folder, f"images_4-5/{id}/5.webp"))
                
                images.update({id: smallImages})
        
        # Выполняем сравнение для каждого товара
        for imageid in images:
            result += f"Проверка товара {imageid}:\n"
            found = False
            for image in images[imageid]:
                if os.path.exists(image):
                    match_result = compare_video_image(image, output_video)
                    if match_result and match_result[0]:
                        found = True
                        result += f"  - Найдено совпадение в видео! Таймкоды: {match_result[1]}\n"
                    else:
                        result += f"  - Изображение {os.path.basename(image)} не найдено в видео\n"
                else:
                    result += f"  - Файл {image} не существует\n"
            
            if not found:
                result += f"  ВНИМАНИЕ: Товар {imageid} не найден в видео!\n"
        
        # Удаляем временное видео
        try:
            os.remove(output_video)
        except:
            pass
    
    except Exception as e:
        import traceback
        result += f"Произошла ошибка при обработке:\n{str(e)}\n\n{traceback.format_exc()}"
    
    result += f"\nОбработка завершена за {time.time() - startTime:.2f} секунд"
    return result

# Если скрипт запущен напрямую
if __name__ == "__main__":
    # Проверяем наличие аргументов командной строки
    if len(sys.argv) > 1:
        # Если переданы аргументы, используем их
        index = int(sys.argv[1])
        
        # Если передан путь к данным, используем его
        data_folder = None
        if len(sys.argv) > 2:
            data_folder = sys.argv[2]
        
        # Запускаем проверку с переданными аргументами
        result = check(index, data_folder)
        print(result)
    else:
        # Если аргументы не переданы, запускаем с индексом 2
        result = check(2)
        print(result)