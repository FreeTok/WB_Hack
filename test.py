import json
import sys

def print_json_element(file_path, index):
    try:
        # Открываем JSON файл
        with open(file_path, 'r', encoding='utf-8') as file:
            # Загружаем JSON в Python объект
            data = json.load(file)
            
        # Проверяем, является ли JSON списком
        if isinstance(data, list):
            # Проверяем, что индекс находится в допустимом диапазоне
            if 0 <= index < len(data):
                print(f"Элемент с индексом {index}:")
                print(json.dumps(data[index], ensure_ascii=False, indent=4))
            else:
                print(f"Ошибка: индекс {index} вне диапазона. Доступные индексы: 0-{len(data)-1}")
        else:
            print("Ошибка: JSON файл не содержит список, невозможно получить элемент по индексу.")
            
    except FileNotFoundError:
        print(f"Ошибка: файл '{file_path}' не найден.")
    except json.JSONDecodeError:
        print(f"Ошибка: файл '{file_path}' содержит некорректный JSON.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    # Проверяем аргументы командной строки
    if len(sys.argv) != 3:
        print("Использование: python script.py <путь_к_json_файлу> <индекс>")
    else:
        file_path = sys.argv[1]
        try:
            index = int(sys.argv[2])
            print_json_element(file_path, index)
        except ValueError:
            print("Ошибка: индекс должен быть целым числом.")