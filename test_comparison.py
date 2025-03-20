from ultralytics import YOLO

import torch
from torchvision import transforms
import torchvision.models as models
from sklearn.metrics.pairwise import cosine_similarity

import cv2

from PIL import Image

# Load a model
model = YOLO("yolo11n.pt", verbose=False)  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

# Загрузка ResNet для извлечения признаков
model_resnet = models.resnet50(pretrained=True)
model_resnet = torch.nn.Sequential(*list(model_resnet.children())[:-1])
model_resnet.eval()

# Преобразование изображений для ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_detections(results):
    if not results or len(results) == 0: return None

    res = []
    for result in results[0]:
        res.append(result.boxes.xyxy.tolist()[0])
    
    return res

index = 0
# Функция для извлечения объектов
def crop_objects(image, detections):
    global index
    if len(detections) == 0: return None

    cropped_images = []
    for detection in detections:
        xmin, ymin, xmax, ymax = detection
        cropped_image = image.crop((xmin, ymin, xmax, ymax))
        cropped_images.append(cropped_image)
        
        index += 1

    return cropped_images

# Функция для извлечения feature vectors
def get_feature_vector(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model_resnet(image)
    return features.squeeze().numpy()

# Функция для сравнения feature vectors
def compare_features(features1, features2, threshold=0.75):
    for i, f1 in enumerate(features1):
        for j, f2 in enumerate(features2):
            similarity = cosine_similarity(f1.reshape(1, -1), f2.reshape(1, -1))
            if similarity > threshold:
                print(f"Объект {i} на изображении 1 и объект {j} на изображении 2 схожи! Сходство: {similarity}")
                return True
    return False

def compare_images(img1, img2):
    if not img1 or not img2: return False

    # Predict with the model
    results1 = model(img1, verbose=False)  # predict on an image
    results2 = model(img2, verbose=False)  # predict on an image

    if len(results1) == 0 or len(results2) == 0: return False

    detections1 = get_detections(results1)
    detections2 = get_detections(results2)

    if len(detections1) == 0 or len(detections2) == 0: return False

    # Извлечение объектов
    cropped_images1 = crop_objects(img1, detections1)
    cropped_images2 = crop_objects(img2, detections2)

    if len(cropped_images1) == 0 or len(cropped_images2) == 0: return False

    # Извлечение feature vectors
    features1 = [get_feature_vector(img) for img in cropped_images1]
    features2 = [get_feature_vector(img) for img in cropped_images2]

    if len(features1) == 0 or len(features2) == 0: return False

    # Сравнение объектов
    return compare_features(features1, features2)


img1 = Image.open('Input/A60/image1.jpg')
# img2 = Image.open('Input/A60/video.jpg')

# compare_images(img1, img2)


# Загрузка видео
video_path = 'Input/A60/video.mp4'
cap = cv2.VideoCapture(video_path)

# Параметры для извлечения кадров
frame_rate = 1  # 1 кадр в секунду
frame_count = 0
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Получаем FPS видео

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Извлечение кадров с заданной частотой
    if frame_count % (fps * frame_rate) == 0:
        # Преобразование кадра в PIL Image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if compare_images(img1, frame_pil):
            print(frame_count / fps)
            frame_pil.save(f'frame_{index}.jpg')
            index += 1

    frame_count += 1

cap.release()