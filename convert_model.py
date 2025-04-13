import torch
import torchvision.models as models
import os
import argparse

def convert_model(input_path, output_path, mode="full"):
    """
    Преобразует модель из формата checkpoint в стандартный формат PyTorch.
    
    Args:
        input_path (str): Путь к исходному файлу модели
        output_path (str): Путь для сохранения преобразованной модели
        mode (str): Режим сохранения: "full" для полной модели, "extractor" для feature extractor
    """
    print(f"Загрузка модели из {input_path}...")
    
    # Определяем устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    # Загружаем исходную модель
    checkpoint = torch.load(input_path, map_location=device)
    
    # Извлекаем метаданные для сохранения
    metadata = {}
    for key in ['val_loss', 'val_accuracy', 'precision', 'recall', 'f1', 'embedding_dim']:
        if key in checkpoint:
            metadata[key] = checkpoint[key]
    
    # Проверяем структуру загруженного объекта
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("Найден ключ 'model_state_dict'")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("Найден ключ 'state_dict'")
    else:
        state_dict = checkpoint
        print("Используем загруженный объект как state_dict")
    
    # Создаем новый словарь с правильными ключами для EfficientNet
    efficientnet_state_dict = {}
    
    # Выделяем только веса backbone
    has_backbone_prefix = any('backbone.' in k for k in state_dict.keys())
    
    if has_backbone_prefix:
        print("Обнаружен префикс 'backbone.' в ключах, адаптируем state_dict...")
        for key, value in state_dict.items():
            if key.startswith('backbone.'):
                new_key = key[len('backbone.'):]
                efficientnet_state_dict[new_key] = value
    else:
        # Если нет префикса backbone, предполагаем, что это уже правильный словарь
        print("Префикс 'backbone.' не обнаружен, используем state_dict как есть")
        efficientnet_state_dict = state_dict
    
    print(f"Извлечено {len(efficientnet_state_dict)} весов для модели.")
    
    # Создаем новую модель EfficientNet
    full_model = models.efficientnet_b0(weights=None)
    
    # Пытаемся загрузить веса (с допуском неполного совпадения)
    try:
        full_model.load_state_dict(efficientnet_state_dict, strict=False)
        print("Веса успешно загружены в модель EfficientNet")
    except Exception as e:
        print(f"Предупреждение при загрузке весов: {e}")
    
    # В зависимости от режима сохраняем полную модель или feature extractor
    if mode == "extractor":
        # Создаем feature extractor (без классификационного слоя)
        model_to_save = torch.nn.Sequential(*list(full_model.children())[:-1])
        print("Создан feature extractor без классификационного слоя")
    else:
        model_to_save = full_model
        print("Используется полная модель EfficientNet")
    
    # Переводим модель в режим оценки
    model_to_save.eval()
    
    # Добавляем метаданные при сохранении
    save_dict = {
        "model": model_to_save.state_dict(),
        **metadata
    }
    
    # Сохраняем модель
    torch.save(save_dict, output_path)
    print(f"Модель успешно сохранена в {output_path}")
    
    # Для проверки сохраняем также чистый state_dict
    plain_output_path = output_path.replace('.pth', '_plain.pth')
    torch.save(model_to_save.state_dict(), plain_output_path)
    print(f"Также сохранен чистый state_dict в {plain_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Конвертация модели EfficientNet")
    parser.add_argument("--input", type=str, default="custom_efficientnet_b0.pth", help="Путь к исходной модели")
    parser.add_argument("--output", type=str, default="converted_model.pth", help="Путь для сохранения модели")
    parser.add_argument("--mode", type=str, choices=["full", "extractor"], default="extractor", 
                        help="Режим сохранения: 'full' для полной модели, 'extractor' для feature extractor")
    
    args = parser.parse_args()
    
    convert_model(args.input, args.output, args.mode)
