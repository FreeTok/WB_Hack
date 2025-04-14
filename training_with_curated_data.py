import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"curated_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CuratedTraining")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CuratedDataset(Dataset):
    def __init__(self, data_file, transform=None):
        self.transform = transform
        self.data_file = os.path.abspath(data_file)
        self.data_dir = os.path.dirname(os.path.abspath(data_file))
        
        self._path_cache = {}
        self.error_count = 0
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            logger.info(f"Успешно загружены данные из {data_file}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных из {data_file}: {e}")
            self.data = []
        
        if isinstance(self.data, list):
            logger.info(f"Загружено {len(self.data)} пар изображений в новом формате из {data_file}")
        else:
            self.data = self._process_old_format(self.data)
            logger.info(f"Загружено {len(self.data)} пар изображений в старом формате из {data_file}")
        
        self._validate_files()
        
        positives = sum(1 for item in self.data if item["label"] == 1)
        negatives = sum(1 for item in self.data if item["label"] == 0)
        
        logger.info(f"Положительных пар: {positives}, отрицательных пар: {negatives}")
        logger.info(f"Проверено существование файлов. {self.error_count} пар имеют недействительные пути.")
    
    def _process_old_format(self, data_dict):
        processed_data = []
        
        for pair in data_dict.get("positive_pairs", []):
            frame_path = pair.get("frame", "")
            product_path = pair.get("product", "")
            
            frame_path = os.path.normpath(frame_path)
            product_path = os.path.normpath(product_path)
            
            processed_data.append({
                "frame_path": frame_path,
                "product_path": product_path,
                "label": 1
            })
        
        for pair in data_dict.get("negative_pairs", []):
            frame_path = pair.get("frame", "")
            product_path = pair.get("product", "")
            
            frame_path = os.path.normpath(frame_path)
            product_path = os.path.normpath(product_path)
            
            processed_data.append({
                "frame_path": frame_path,
                "product_path": product_path,
                "label": 0
            })
        
        return processed_data
    
    def _validate_files(self):
        valid_data = []
        
        for item in self.data:
            frame_path = self._fix_path(item.get("frame_path", ""))
            product_path = self._fix_path(item.get("product_path", ""))
            
            if frame_path and product_path:
                valid_data.append({
                    'frame_path': frame_path,
                    'product_path': product_path,
                    'label': item.get('label', 0)
                })
            else:
                self.error_count += 1
                logger.warning(f"Не удалось исправить пути для пары: {item}")
        
        self.data = valid_data
        logger.info(f"Проверка файлов завершена. Осталось {len(valid_data)} валидных пар")
    
    def _fix_path(self, path):
        if not path:
            return None
            
        if path in self._path_cache:
            return self._path_cache[path]
        
        path = os.path.normpath(path)
        
        if os.path.isabs(path) and os.path.exists(path):
            self._path_cache[path] = path
            return path
        
        potential_paths = []
        
        data_path = os.path.abspath(os.path.join(self.data_dir, path))
        potential_paths.append(data_path)
        
        normalized_path = os.path.normpath(os.path.join(self.data_dir, path))
        potential_paths.append(normalized_path)
        
        abs_path = os.path.abspath(path)
        potential_paths.append(abs_path)
        
        for prefix in ['curated', 'data', 'images', 'temp', 'videos', 'frames', 'curated_data']:
            if os.path.basename(path).lower().startswith(prefix.lower()) or os.path.dirname(path).lower().startswith(prefix.lower()):
                root_path = os.path.abspath(os.path.join(os.path.dirname(self.data_dir), path))
                potential_paths.append(root_path)
                
                parent_root_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(self.data_dir)), path))
                potential_paths.append(parent_root_path)
        
        filename = os.path.basename(path)
        for root, dirs, files in os.walk(self.data_dir):
            if filename in files:
                file_path = os.path.join(root, filename)
                potential_paths.append(file_path)
        
        for test_path in set(potential_paths):  
            if os.path.exists(test_path):
                self._path_cache[path] = test_path
                return test_path
        
        logger.warning(f"Не удалось найти файл по пути {path}. Проверены варианты: {potential_paths}")
        return None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            
            frame_img = Image.open(item['frame_path']).convert('RGB')
            product_img = Image.open(item['product_path']).convert('RGB')
            
            if self.transform:
                frame_img = self.transform(frame_img)
                product_img = self.transform(product_img)
            
            return {
                'frame': frame_img,
                'product': product_img,
                'label': torch.tensor(item['label'], dtype=torch.float32)
            }
        
        except Exception as e:
            logger.error(f"Ошибка при загрузке пары {idx}: {e}")
            dummy = torch.zeros(3, 224, 224)
            return {
                'frame': dummy,
                'product': dummy,
                'label': torch.tensor(0, dtype=torch.float32)
            }
        
        
class ProductMatcher(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0", embedding_dim=512):
        super(ProductMatcher, self).__init__()
        
        from torchvision import models
        
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
        
        self.backbone_name = backbone_name
        self.feature_dim = in_features
        
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
    def forward_one(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  
        embedding = self.projection_head(features)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding
    
    def forward(self, x1, x2):
        embedding1 = self.forward_one(x1)
        embedding2 = self.forward_one(x2)
        return embedding1, embedding2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, embedding1, embedding2, label):
        cosine_similarity = torch.sum(embedding1 * embedding2, dim=1)
        
        loss = (1-label) * torch.pow(cosine_similarity, 2) + \
               (label) * torch.pow(torch.clamp(self.margin - cosine_similarity, min=0.0), 2)
        
        return loss.mean()

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, save_path):
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        logger.info(f"Эпоха {epoch+1}/{epochs}")
        
        model.train()
        train_loss = 0.0
        all_predictions = []
        all_labels = []
        
        for batch in tqdm(train_loader, desc="Обучение"):
            frame_imgs = batch['frame'].to(device)
            product_imgs = batch['product'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            frame_emb, product_emb = model(frame_imgs, product_imgs)
            
            loss = criterion(frame_emb, product_emb, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            with torch.no_grad():
                similarities = torch.sum(frame_emb * product_emb, dim=1)
                predictions = (similarities > 0.5).float()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_train_loss = train_loss / len(train_loader)
        
        train_accuracy = accuracy_score(all_labels, all_predictions)
        
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(train_accuracy)
        
        logger.info(f"Обучение - Потери: {avg_train_loss:.4f}, Точность: {train_accuracy:.4f}")
        
        model.eval()
        val_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Валидация"):
                frame_imgs = batch['frame'].to(device)
                product_imgs = batch['product'].to(device)
                labels = batch['label'].to(device)
                
                frame_emb, product_emb = model(frame_imgs, product_imgs)
                
                loss = criterion(frame_emb, product_emb, labels)
                
                val_loss += loss.item()
                
                similarities = torch.sum(frame_emb * product_emb, dim=1)
                predictions = (similarities > 0.5).float()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        val_accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, zero_division=0)
        
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['precision'].append(precision)
        history['recall'].append(recall)
        history['f1'].append(f1)
        
        logger.info(f"Валидация - Потери: {avg_val_loss:.4f}, Точность: {val_accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_accuracy': val_accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'backbone': model.backbone_name,
                'embedding_dim': model.projection_head[-2].out_features
            }, f"{save_path}_best.pth")
            
            logger.info(f"Сохранена лучшая модель с потерями: {best_val_loss:.4f}")
    
    torch.save({
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'backbone': model.backbone_name,
        'embedding_dim': model.projection_head[-2].out_features
    }, f"{save_path}_last.pth")
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Потери на обучении')
    plt.plot(history['val_loss'], label='Потери на валидации')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    plt.title('Динамика потерь')
    
    plt.subplot(2, 2, 2)
    plt.plot(history['train_accuracy'], label='Точность на обучении')
    plt.plot(history['val_accuracy'], label='Точность на валидации')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    plt.title('Динамика точности')
    
    plt.subplot(2, 2, 3)
    plt.plot(history['precision'], label='Precision')
    plt.plot(history['recall'], label='Recall')
    plt.xlabel('Эпоха')
    plt.ylabel('Значение')
    plt.legend()
    plt.title('Precision и Recall')
    
    plt.subplot(2, 2, 4)
    plt.plot(history['f1'], label='F1')
    plt.xlabel('Эпоха')
    plt.ylabel('F1')
    plt.legend()
    plt.title('F1 Score')
    
    plt.tight_layout()
    
    plt.savefig(f"{save_path}_training_history.png")
    plt.close()
    
    return model, history

def train_model_mixed_precision(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, device, epochs, save_path):
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        logger.info(f"Эпоха {epoch+1}/{epochs}")
        
        model.train()
        train_loss = 0.0
        all_predictions = []
        all_labels = []
        
        for batch in tqdm(train_loader, desc="Обучение"):
            frame_imgs = batch['frame'].to(device)
            product_imgs = batch['product'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                frame_emb, product_emb = model(frame_imgs, product_imgs)
                loss = criterion(frame_emb, product_emb, labels)
            
            scaler.scale(loss).backward()
            
            scaler.step(optimizer)
            
            scaler.update()
            
            train_loss += loss.item()
            
            with torch.no_grad():
                similarities = torch.sum(frame_emb * product_emb, dim=1)
                predictions = (similarities > 0.5).float()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_train_loss = train_loss / len(train_loader)
        
        train_accuracy = accuracy_score(all_labels, all_predictions)
        
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(train_accuracy)
        
        logger.info(f"Обучение - Потери: {avg_train_loss:.4f}, Точность: {train_accuracy:.4f}")
        
        model.eval()
        val_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Валидация"):
                frame_imgs = batch['frame'].to(device)
                product_imgs = batch['product'].to(device)
                labels = batch['label'].to(device)
                
                with torch.cuda.amp.autocast():
                    frame_emb, product_emb = model(frame_imgs, product_imgs)
                    loss = criterion(frame_emb, product_emb, labels)
                
                val_loss += loss.item()
                
                similarities = torch.sum(frame_emb * product_emb, dim=1)
                predictions = (similarities > 0.5).float()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        val_accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, zero_division=0)
        
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['precision'].append(precision)
        history['recall'].append(recall)
        history['f1'].append(f1)
        
        logger.info(f"Валидация - Потери: {avg_val_loss:.4f}, Точность: {val_accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_accuracy': val_accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'backbone': model.backbone_name,
                'embedding_dim': model.projection_head[-2].out_features
            }, f"{save_path}_best.pth")
            
            logger.info(f"Сохранена лучшая модель с потерями: {best_val_loss:.4f}")
    
    torch.save({
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'backbone': model.backbone_name,
        'embedding_dim': model.projection_head[-2].out_features
    }, f"{save_path}_last.pth")
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Потери на обучении')
    plt.plot(history['val_loss'], label='Потери на валидации')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    plt.title('Динамика потерь')
    
    plt.subplot(2, 2, 2)
    plt.plot(history['train_accuracy'], label='Точность на обучении')
    plt.plot(history['val_accuracy'], label='Точность на валидации')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    plt.title('Динамика точности')
    
    plt.subplot(2, 2, 3)
    plt.plot(history['precision'], label='Precision')
    plt.plot(history['recall'], label='Recall')
    plt.xlabel('Эпоха')
    plt.ylabel('Значение')
    plt.legend()
    plt.title('Precision и Recall')
    
    plt.subplot(2, 2, 4)
    plt.plot(history['f1'], label='F1')
    plt.xlabel('Эпоха')
    plt.ylabel('F1')
    plt.legend()
    plt.title('F1 Score')
    
    plt.tight_layout()
    
    plt.savefig(f"{save_path}_training_history.png")
    plt.close()
    
    return model, history

def main():
    parser = argparse.ArgumentParser(description="Обучение модели на кураторских данных")
    parser.add_argument("--data_file", type=str, required=True, help="Путь к JSON-файлу с данными")
    parser.add_argument("--output_dir", type=str, default="models", help="Директория для сохранения модели")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0", choices=["resnet50", "efficientnet_b0", "efficientnet_b3"], 
                        help="Базовая архитектура")
    parser.add_argument("--embedding_dim", type=int, default=512, help="Размерность эмбеддинга")
    parser.add_argument("--batch_size", type=int, default=8, help="Размер батча")
    parser.add_argument("--epochs", type=int, default=20, help="Количество эпох")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Скорость обучения")
    parser.add_argument("--val_split", type=float, default=0.2, help="Доля данных для валидации")
    parser.add_argument("--margin", type=float, default=0.5, help="Граница для функции потерь")
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимости")
    parser.add_argument("--mixed_precision", action="store_true", help="Использовать смешанную точность")
    parser.add_argument("--debug", action="store_true", help="Режим отладки с подробным выводом")
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Включен режим отладки")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    params_path = os.path.join(args.output_dir, "curated_training_params.json")
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Используется устройство: {device}")
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    logger.info(f"Загрузка данных из файла: {args.data_file}")
    
    if not os.path.exists(args.data_file):
        logger.error(f"Файл с данными не найден: {args.data_file}")
        logger.info("Поиск альтернативных файлов данных...")
        
        data_file_alternatives = [
            os.path.join(os.path.dirname(args.data_file), "training_data.json"),
            os.path.join(os.path.dirname(args.data_file), "training_pairs.json"),
            os.path.join(os.path.dirname(args.output_dir), "curated_data", "training_data.json")
        ]
        
        for alt_file in data_file_alternatives:
            if os.path.exists(alt_file):
                logger.info(f"Найден альтернативный файл данных: {alt_file}")
                args.data_file = alt_file
                break
        else:
            logger.error("Не найдено альтернативных файлов данных. Выход.")
            return
    
    dataset = CuratedDataset(
        data_file=args.data_file,
        transform=train_transform
    )
    
    if len(dataset) == 0:
        logger.error("Датасет пуст! Проверьте файл с данными и пути к изображениям.")
        return
    
    dataset_size = len(dataset)
    val_size = int(dataset_size * args.val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"Создано {train_size} образцов для обучения и {val_size} для валидации")
    
    model = ProductMatcher(
        backbone_name=args.backbone,
        embedding_dim=args.embedding_dim
    ).to(device)
    
    criterion = ContrastiveLoss(margin=args.margin)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    save_path = os.path.join(args.output_dir, f"product_matcher_{args.backbone}")
    
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision and torch.cuda.is_available() else None
    
    if args.mixed_precision and torch.cuda.is_available():
        logger.info("Обучение со смешанной точностью")
        model, history = train_model_mixed_precision(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epochs=args.epochs,
            save_path=save_path
        )
    else:
        logger.info("Обучение с стандартной точностью")
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