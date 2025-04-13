import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import random
import logging
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score

# Импортируем наш датасет
from improved_dataset import ProductVideoDataset, HardTripletDataset

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

# Устанавливаем seed для воспроизводимости
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Определение сети для сопоставления товаров и видео
class ProductMatcher(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0", embedding_dim=512):
        super(ProductMatcher, self).__init__()
        
        # Загружаем предобученный backbone
        if backbone_name == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            in_features = self.backbone.fc.in_features
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif backbone_name == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone_name == "efficientnet_b3":
            self.backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Неподдерживаемый backbone: {backbone_name}")
        
        # Проекционная голова для эмбеддингов
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
    def forward_one(self, x):
        """Прямой проход для одного изображения"""
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten
        embedding = self.projection_head(features)
        # L2 нормализация для косинусного сходства
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding
    
    def forward(self, x1, x2=None, x3=None):
        """Прямой проход для одного или нескольких изображений"""
        embedding1 = self.forward_one(x1)
        
        if x2 is not None:
            embedding2 = self.forward_one(x2)
            
            if x3 is not None:
                embedding3 = self.forward_one(x3)
                return embedding1, embedding2, embedding3
            
            return embedding1, embedding2
        
        return embedding1

# Контрастивная функция потерь
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, embedding1, embedding2, label):
        """
        embedding1, embedding2: нормализованные эмбеддинги
        label: 1 для положительных пар (совпадение), 0 для отрицательных (разные товары)
        """
        # Косинусное сходство
        cosine_similarity = torch.sum(embedding1 * embedding2, dim=1)
        
        # Контрастивная потеря
        loss = (1-label) * torch.pow(cosine_similarity, 2) + \
               (label) * torch.pow(torch.clamp(self.margin - cosine_similarity, min=0.0), 2)
        
        return loss.mean()

# Triplet-функция потерь
class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        """
        anchor, positive, negative: нормализованные эмбеддинги
        """
        # Вычисляем косинусное сходство
        pos_sim = torch.sum(anchor * positive, dim=1)
        neg_sim = torch.sum(anchor * negative, dim=1)
        
        # Трансформируем в расстояние (1 - сходство)
        pos_dist = 1.0 - pos_sim
        neg_dist = 1.0 - neg_sim
        
        # Вычисляем потери (хотим, чтобы pos_dist < neg_dist - margin)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        
        return loss.mean()

# Функция для обучения модели с контрастивной потерей
def train_model_contrastive(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, save_path):
    """Обучение модели с использованием контрастивной потери"""
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
        for batch in tqdm(train_loader, desc="Обучение"):
            product_imgs = batch['product'].to(device)
            frame_imgs = batch['frame'].to(device)
            labels = batch['label'].to(device)
            
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
            predictions = (similarity >= 0.5).float()
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
                for batch in tqdm(val_loader, desc="Валидация"):
                    product_imgs = batch['product'].to(device)
                    frame_imgs = batch['frame'].to(device)
                    labels = batch['label'].to(device)
                    
                    # Прямой проход
                    embedding1, embedding2 = model(product_imgs, frame_imgs)
                    
                    # Вычисляем потери
                    loss = criterion(embedding1, embedding2, labels)
                    
                    # Собираем статистику
                    val_loss += loss.item()
                    
                    # Вычисляем точность
                    similarity = torch.sum(embedding1 * embedding2, dim=1)
                    predictions = (similarity >= 0.5).float()
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total if total > 0 else 0
            
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            logger.info(f"Валидация - Потери: {avg_val_loss:.4f}, Точность: {val_accuracy:.4f}")
            
            # Обновляем планировщик скорости обучения
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()
            
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
                    'backbone': 'efficientnet_b0',
                    'embedding_dim': model.projection_head[-2].out_features
                }, f"{save_path}_best.pth")
                
                logger.info(f"Сохранена лучшая модель с потерями: {best_val_loss:.4f}")
    
    # Сохраняем последнюю модель
    torch.save({
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'backbone': 'efficientnet_b0',
        'embedding_dim': model.projection_head[-2].out_features
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

# Функция для обучения модели с триплетной потерей
def train_model_triplet(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, save_path):
    """Обучение модели с использованием триплетной потери"""
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # Лучшая потеря на валидации для сохранения модели
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        logger.info(f"Эпоха {epoch+1}/{epochs}")
        
        # Режим обучения
        model.train()
        train_loss = 0.0
        
        # Обучение на тренировочном наборе
        for batch in tqdm(train_loader, desc="Обучение"):
            anchor_imgs = batch['anchor'].to(device)
            positive_imgs = batch['positive'].to(device)
            negative_imgs = batch['negative'].to(device)
            
            # Обнуляем градиенты
            optimizer.zero_grad()
            
            # Прямой проход
            anchor_emb, positive_emb, negative_emb = model(anchor_imgs, positive_imgs, negative_imgs)
            
            # Вычисляем потери
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            
            # Обратное распространение и оптимизация
            loss.backward()
            optimizer.step()
            
            # Собираем статистику
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        history['train_loss'].append(avg_train_loss)
        
        logger.info(f"Обучение - Потери: {avg_train_loss:.4f}")
        
        # Валидация
        if val_loader:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Валидация"):
                    anchor_imgs = batch['anchor'].to(device)
                    positive_imgs = batch['positive'].to(device)
                    negative_imgs = batch['negative'].to(device)
                    
                    # Прямой проход
                    anchor_emb, positive_emb, negative_emb = model(anchor_imgs, positive_imgs, negative_imgs)
                    
                    # Вычисляем потери
                    loss = criterion(anchor_emb, positive_emb, negative_emb)
                    
                    # Собираем статистику
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            history['val_loss'].append(avg_val_loss)
            
            logger.info(f"Валидация - Потери: {avg_val_loss:.4f}")
            
            # Обновляем планировщик скорости обучения
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()
            
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
                    'backbone': 'efficientnet_b0',
                    'embedding_dim': model.projection_head[-2].out_features
                }, f"{save_path}_best.pth")
                
                logger.info(f"Сохранена лучшая модель с потерями: {best_val_loss:.4f}")
    
    # Сохраняем последнюю модель
    torch.save({
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'backbone': 'efficientnet_b0',
        'embedding_dim': model.projection_head[-2].out_features
    }, f"{save_path}_last.pth")
    
    # Строим графики обучения
    plt.figure(figsize=(10, 5))
    
    plt.plot(history['train_loss'], label='Потери на обучении')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(history['val_loss'], label='Потери на валидации')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    
    # Сохраняем график
    plt.savefig(f"{save_path}_training_history.png")
    plt.close()
    
    return model, history

def main():
    parser = argparse.ArgumentParser(description="Обучение модели для поиска товаров на видео")
    parser.add_argument("--data_dir", type=str, required=True, help="Корневая директория с данными")
    parser.add_argument("--videos_json", type=str, required=True, help="Путь к JSON-файлу с видео")
    parser.add_argument("--output_dir", type=str, default="models", help="Директория для сохранения модели")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0", choices=["resnet50", "efficientnet_b0", "efficientnet_b3"], 
                        help="Базовая архитектура")
    parser.add_argument("--loss_type", type=str, default="contrastive", choices=["contrastive", "triplet"], 
                        help="Тип функции потерь")
    parser.add_argument("--embedding_dim", type=int, default=512, help="Размерность эмбеддинга")
    parser.add_argument("--batch_size", type=int, default=8, help="Размер батча")
    parser.add_argument("--epochs", type=int, default=10, help="Количество эпох")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Скорость обучения")
    parser.add_argument("--val_split", type=float, default=0.2, help="Доля данных для валидации")
    parser.add_argument("--max_samples", type=int, default=1000, help="Максимальное количество образцов")
    parser.add_argument("--margin", type=float, default=0.5, help="Граница для функции потерь")
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимости")
    
    args = parser.parse_args()
    
    # Устанавливаем seed для воспроизводимости
    set_seed(args.seed)
    
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
    
    # Создаем датасет в зависимости от типа функции потерь
    if args.loss_type == "contrastive":
        logger.info(f"Создаем контрастивный датасет, макс. размер: {args.max_samples}")
        dataset = ProductVideoDataset(
            data_root=args.data_dir,
            videos_json=args.videos_json,
            transform=train_transform,
            max_pairs=args.max_samples
        )
    else:  # triplet
        logger.info(f"Создаем триплетный датасет, макс. размер: {args.max_samples}")
        dataset = HardTripletDataset(
            data_root=args.data_dir,
            videos_json=args.videos_json,
            transform=train_transform,
            max_triplets=args.max_samples
        )
    
    # Разделяем на обучающую и валидационную выборки
    dataset_size = len(dataset)
    val_size = int(dataset_size * args.val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Создаем DataLoader'ы
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4 if torch.cuda.is_available() else 2,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 2,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"Создано {len(train_dataset)} образцов для обучения и {len(val_dataset)} для валидации")
    
    # Создаем модель
    model = ProductMatcher(
        backbone_name=args.backbone,
        embedding_dim=args.embedding_dim
    ).to(device)
    
    # Выбираем функцию потерь
    if args.loss_type == "contrastive":
        criterion = ContrastiveLoss(margin=args.margin)
    else:  # triplet
        criterion = TripletLoss(margin=args.margin)
    
    # Оптимизатор и планировщик скорости обучения
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    
    # Планировщик скорости обучения с уменьшением при плато
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Путь для сохранения модели
    save_path = os.path.join(args.output_dir, f"product_matcher_{args.backbone}_{args.loss_type}")
    
    # Обучаем модель
    if args.loss_type == "contrastive":
        model, history = train_model_contrastive(
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
    else:  # triplet
        model, history = train_model_triplet(
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