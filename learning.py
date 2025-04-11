import os
import json
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import subprocess
import argparse
import logging
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import random

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ProductTraining")

# Определение сети для сравнения товаров и видео
class ProductMatcher(nn.Module):
    def __init__(self, backbone="efficientnet_b0", embedding_dim=512, pretrained=True):
        super(ProductMatcher, self).__init__()
        
        # Загружаем предобученный backbone
        if backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        else:
            raise ValueError(f"Неподдерживаемый backbone: {backbone}")
        
        # Проекционная голова для получения эмбеддингов
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim)
        )
        
    def forward_one(self, x):
        """Прямой проход для одного изображения"""
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten
        embedding = self.projection_head(features)
        # L2 нормализация для косинусного сходства
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding
    
    def forward(self, x1, x2=None):
        """Прямой проход для пары изображений или одного изображения"""
        embedding1 = self.forward_one(x1)
        
        if x2 is not None:
            embedding2 = self.forward_one(x2)
            return embedding1, embedding2
        else:
            return embedding1


# Контрастивная функция потерь
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, reduction='mean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        
    def forward(self, embedding1, embedding2, label):
        """
        embedding1, embedding2: нормализованные эмбеддинги
        label: 1 для положительных пар (совпадение), 0 для отрицательных (разные товары)
        """
        # Косинусное расстояние = 1 - косинусное сходство
        cosine_similarity = torch.sum(embedding1 * embedding2, dim=1)
        distance = 1.0 - cosine_similarity
        
        # Контрастивная потеря
        loss = (1-label) * torch.pow(cosine_similarity, 2) + \
               (label) * torch.pow(torch.clamp(self.margin - cosine_similarity, min=0.0), 2)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# Датасет для обучения
class ProductVideoDataset(Dataset):
    def __init__(self, data_root, videos_json, product_json, transform=None):
        """
        data_root: корневая директория с данными
        videos_json: путь к JSON-файлу с видео
        product_json: путь к JSON-файлу с товарами
        transform: трансформации для изображений
        """
        self.data_root = data_root
        self.transform = transform if transform else transforms.ToTensor()
        self.temp_dir = os.path.join(data_root, "temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Загружаем данные
        self.videos_data = self._load_json(videos_json)
        self.products_data = self._load_json(product_json)
        
        # Индексы для быстрого доступа
        self.nm_id_to_product = {item['nm_id']: item for item in self.products_data}
        self.video_id_to_video = {item['video_id']: item for item in self.videos_data}
        
        # Создаем список пар продукт-видео
        self.pairs = []
        self._prepare_pairs()
        
        logger.info(f"Подготовлено {len(self.pairs)} пар продукт-видео для обучения")
    
    def _load_json(self, path):
        """Загрузка JSON-файла"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки JSON из {path}: {e}")
            return []
    
    def _prepare_pairs(self):
        """Подготовка пар продукт-видео для обучения"""

        max_pairs = 100  # Ограничение на 100 пар

        for video in self.videos_data:
            video_id = video['video_id']
            nm_ids = video.get('nm_ids', [])
            
            # Пропускаем видео без товаров
            if not nm_ids:
                continue
            
            # Для каждого товара на видео создаем положительную пару
            for nm_id in nm_ids:
                if nm_id not in self.nm_id_to_product:
                    continue
                
                # Проверяем наличие изображений товара
                product_images = self._get_product_image_paths(nm_id)
                if not product_images:
                    continue
                
                # Добавляем положительную пару (товар присутствует на видео)
                self.pairs.append({
                    'product_id': nm_id,
                    'product_images': product_images,
                    'video_id': video_id,
                    'video_url': video['path_url'],
                    'label': 1  # Положительная пара
                })

                if len(self.pairs) >= max_pairs:
                    logger.info(f"Достигнут лимит в {max_pairs} пар")
                    return
                
                # Добавляем отрицательную пару (случайный товар, не присутствующий на видео)
                # Выбираем случайный товар, не из списка nm_ids
                negative_nm_ids = set(self.nm_id_to_product.keys()) - set(nm_ids)
                if negative_nm_ids:
                    neg_nm_id = random.choice(list(negative_nm_ids))
                    neg_images = self._get_product_image_paths(neg_nm_id)
                    
                    if neg_images:
                        self.pairs.append({
                            'product_id': neg_nm_id,
                            'product_images': neg_images,
                            'video_id': video_id,
                            'video_url': video['path_url'],
                            'label': 0  # Отрицательная пара
                        })
    
    def _get_product_image_paths(self, nm_id):
        """Получение путей к изображениям товара"""
        image_paths = []
        
        # Проверяем наличие основного изображения
        img1_path = os.path.join(self.data_root, f"images_1/{nm_id}/1.webp")
        if os.path.exists(img1_path):
            image_paths.append(img1_path)
        
        # Проверяем наличие дополнительных изображений
        for img_idx in [2, 3]:
            img_path = os.path.join(self.data_root, f"images_2-3/{nm_id}/{img_idx}.webp")
            if os.path.exists(img_path):
                image_paths.append(img_path)
        
        # Проверяем наличие дополнительных изображений 4-5
        for img_idx in [4, 5]:
            img_path = os.path.join(self.data_root, f"images_4-5/{nm_id}/{img_idx}.webp")
            if os.path.exists(img_path):
                image_paths.append(img_path)
        
        return image_paths
    
    def _extract_random_frame(self, video_url):
        """Извлечение случайного кадра из видео"""
        # Создаем временный файл для кадра
        output_frame = os.path.join(self.temp_dir, f"frame_{hash(video_url)}_{random.randint(0, 10000)}.jpg")
        
        try:
            # Получаем информацию о длительности видео
            probe_command = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_url
            ]
            
            result = subprocess.run(probe_command, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Ошибка при получении длительности видео: {result.stderr}")
                return None
            
            duration = float(result.stdout.strip())
            if duration <= 0:
                return None
            
            # Выбираем случайное время в пределах длительности видео
            random_time = random.uniform(0, duration)
            
            # Извлекаем кадр
            extract_command = [
                'ffmpeg',
                '-ss', str(random_time),
                '-i', video_url,
                '-frames:v', '1',
                '-q:v', '2',
                '-v', 'error',
                output_frame
            ]
            
            result = subprocess.run(extract_command, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Ошибка при извлечении кадра: {result.stderr}")
                return None
            
            if os.path.exists(output_frame):
                return output_frame
            
            return None
        except Exception as e:
            logger.error(f"Исключение при извлечении кадра: {e}")
            return None
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """Получение пары (изображение товара, кадр из видео) и метки"""
        item = self.pairs[idx]
        
        # Выбираем случайное изображение товара
        product_image_path = random.choice(item['product_images'])
        product_image = Image.open(product_image_path).convert('RGB')
        
        # Извлекаем случайный кадр из видео
        frame_path = self._extract_random_frame(item['video_url'])
        
        if frame_path and os.path.exists(frame_path):
            frame = Image.open(frame_path).convert('RGB')
            # Удаляем временный файл кадра
            try:
                os.remove(frame_path)
            except:
                pass
        else:
            # Если не удалось извлечь кадр, создаем пустое изображение
            frame = Image.new('RGB', (224, 224), color='black')
            # Для пустого кадра устанавливаем отрицательную метку
            label = 0
        
        # Применяем трансформации
        if self.transform:
            product_image = self.transform(product_image)
            frame = self.transform(frame)
        
        return product_image, frame, torch.tensor(item['label'], dtype=torch.float32)


# Функция для обучения модели
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, save_path):
    """Обучение модели"""
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': []
    }
    
    # Лучшая потеря на валидации для сохранения модели
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        logger.info(f"Эпоха {epoch+1}/{epochs}")
        
        # Режим обучения
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        # Обучение на тренировочном наборе
        for product_imgs, frame_imgs, labels in tqdm(train_loader, desc="Обучение"):
            product_imgs = product_imgs.to(device)
            frame_imgs = frame_imgs.to(device)
            labels = labels.to(device)
            
            # Обнуляем градиенты
            optimizer.zero_grad()
            
            # Прямой проход
            embedding1, embedding2 = model(product_imgs, frame_imgs)
            
            # Вычисляем потери
            loss = criterion(embedding1, embedding2, labels)
            
            # Обратное распространение и оптимизация
            loss.backward()
            optimizer.step()
            
            # Собираем статистику
            train_loss += loss.item()
            
            # Вычисляем точность
            similarity = torch.sum(embedding1 * embedding2, dim=1)
            predictions = torch.where(similarity >= 0.75, 1.0, 0.0)
            predictions = torch.where(similarity <= 0.25, 0.0, predictions)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = correct / total if total > 0 else 0
        
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(train_accuracy)
        
        logger.info(f"Обучение - Потери: {avg_train_loss:.4f}, Точность: {train_accuracy:.4f}")
        
        # Валидация
        if val_loader:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for product_imgs, frame_imgs, labels in tqdm(val_loader, desc="Валидация"):
                    product_imgs = product_imgs.to(device)
                    frame_imgs = frame_imgs.to(device)
                    labels = labels.to(device)
                    
                    # Прямой проход
                    embedding1, embedding2 = model(product_imgs, frame_imgs)
                    
                    # Вычисляем потери
                    loss = criterion(embedding1, embedding2, labels)
                    
                    # Собираем статистику
                    val_loss += loss.item()
                    
                    # Вычисляем точность
                    similarity = torch.sum(embedding1 * embedding2, dim=1)
                    predictions = torch.where(similarity >= 0.75, 1.0, 0.0)
                    predictions = torch.where(similarity <= 0.25, 0.0, predictions)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total if total > 0 else 0
            
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            logger.info(f"Валидация - Потери: {avg_val_loss:.4f}, Точность: {val_accuracy:.4f}")
            
            # Обновляем планировщик скорости обучения
            scheduler.step(avg_val_loss)
            
            # Сохраняем лучшую модель
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                
                # Создаем директорию для сохранения, если она не существует
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'embedding_dim': model.projection_head[-1].out_features,
                    'backbone': 'efficientnet_b0'  # Или другое значение в зависимости от модели
                }, f"{save_path}_best.pth")
                
                logger.info(f"Сохранена лучшая модель с потерями: {best_val_loss:.4f}")
    
    # Сохраняем последнюю модель
    torch.save({
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'embedding_dim': model.projection_head[-1].out_features,
        'backbone': 'efficientnet_b0'  # Или другое значение в зависимости от модели
    }, f"{save_path}_last.pth")
    
    # Строим графики обучения
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Потери на обучении')
    if history['val_loss']:
        plt.plot(history['val_loss'], label='Потери на валидации')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracy'], label='Точность на обучении')
    if history['val_accuracy']:
        plt.plot(history['val_accuracy'], label='Точность на валидации')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    
    # Сохраняем график
    plt.savefig(f"{save_path}_training_history.png")
    plt.close()
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Обучение модели для поиска товаров на видео")
    parser.add_argument("--data_dir", type=str, required=True, help="Корневая директория с данными")
    parser.add_argument("--videos_json", type=str, required=True, help="Путь к JSON-файлу с видео")
    parser.add_argument("--products_json", type=str, required=True, help="Путь к JSON-файлу с товарами")
    parser.add_argument("--output_dir", type=str, default="models", help="Директория для сохранения модели")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0", choices=["resnet50", "efficientnet_b0"], 
                        help="Базовая архитектура")
    parser.add_argument("--embedding_dim", type=int, default=512, help="Размерность эмбеддинга")
    parser.add_argument("--batch_size", type=int, default=32, help="Размер батча")
    parser.add_argument("--epochs", type=int, default=10, help="Количество эпох")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Скорость обучения")
    parser.add_argument("--val_split", type=float, default=0.1, help="Доля данных для валидации")
    parser.add_argument("--max_samples", type=int, default=None, help="Максимальное количество образцов (для тестирования)")
    parser.add_argument("--margin", type=float, default=0.5, help="Граница для контрастивной потери")
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимости")
    
    args = parser.parse_args()
    
    # Устанавливаем seed для воспроизводимости
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Создаем директорию для сохранения модели
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Сохраняем параметры запуска
    params_path = os.path.join(args.output_dir, "training_params.json")
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    
    # Определяем устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Используется устройство: {device}")
    
    # Трансформации для изображений
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Создаем датасет
    full_dataset = ProductVideoDataset(
        data_root=args.data_dir,
        videos_json=args.videos_json,
        product_json=args.products_json,
        transform=train_transform
    )
    
    # Ограничиваем количество образцов (для тестирования)
    if args.max_samples and args.max_samples < len(full_dataset):
        logger.info(f"Ограничение до {args.max_samples} образцов для тестирования")
        indices = list(range(len(full_dataset)))
        random.shuffle(indices)
        subset_indices = indices[:args.max_samples]
        full_dataset = Subset(full_dataset, subset_indices)
    
    # Разделяем на обучающую и валидационную выборки
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * args.val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Создаем DataLoader'ы
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Создано {len(train_dataset)} образцов для обучения и {len(val_dataset)} для валидации")
    
    # Создаем модель
    model = ProductMatcher(
        backbone=args.backbone,
        embedding_dim=args.embedding_dim,
        pretrained=True
    ).to(device)
    
    # Функция потерь и оптимизатор
    criterion = ContrastiveLoss(margin=args.margin)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Путь для сохранения модели
    save_path = os.path.join(args.output_dir, f"product_matcher_{args.backbone}")
    
    # Обучаем модель
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        save_path=save_path
    )
    
    logger.info("Обучение завершено!")


if __name__ == "__main__":
    main()