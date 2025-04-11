import json
import os
import subprocess
import time
import sys
import torch

# Импортируем оптимизированную функцию из optimized_comparison
from test_comparison import optimized_check

def check(i, data_folder=None):
    """
    Обертка для вызова оптимизированной функции проверки.
    Сохраняет совместимость с существующим интерфейсом.
    
    Args:
        i (int): Индекс записи в JSON-файле
        data_folder (str, optional): Путь к папке с данными. По умолчанию определяется автоматически.
    
    Returns:
        dict: Результат проверки с подробной информацией
    """
    # Вызываем оптимизированную функцию проверки
    return optimized_check(i, data_folder)

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
        print(result["raw_log"])
    else:
        # Если аргументы не переданы, запускаем с индексом 0
        result = check(0)
        print(result["raw_log"])