import os
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import pandas as pd
import threading

# Загружаем данные из CSV
df = pd.read_csv('D:/Hackatons/WB_Hack/results.csv')

# Базовые директории для видео и изображений
video_base_dir = "D:/Hackatons/WB_Hack/temp/"
image_base_dir = "D:/Hackatons/data/"

# Создаём окно Tkinter
root = tk.Tk()
root.title("Просмотр видео и фотографий")
root.geometry("1500x1200")  # Увеличиваем размер окна, чтобы разместить все

# Место для вывода видео и изображений
frame = tk.Frame(root)
frame.pack()

video_labels = []
image_labels = []

# Функция для загрузки и отображения видео
def play_video(row_index, video_label):
    video_path = os.path.join(video_base_dir, f"output_video_{row_index}.mp4")
    
    if not os.path.exists(video_path):
        messagebox.showerror("Ошибка", f"Видео {video_path} не найдено!")
        return
    
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        video_label.config(image=img_tk)
        video_label.image = img_tk
        root.update_idletasks()
        root.update()

    cap.release()

# Функция для отображения изображений
def display_images(image_id, image_label):
    image_folders = ["images_1", "images_2-3", "images_4-5"]
    image_paths = []

    # Проверяем все папки с изображениями для текущего image_id
    for folder in image_folders:
        folder_path = os.path.join(image_base_dir, folder, str(image_id))
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".webp"):
                    image_paths.append(os.path.join(folder_path, file))

    if image_paths:
        image = Image.open(image_paths[0])
        img_tk = ImageTk.PhotoImage(image=image)
        image_label.config(image=img_tk)
        image_label.image = img_tk
    else:
        messagebox.showerror("Ошибка", "Изображения не найдены!")

# Функция для загрузки и отображения 5 видео и изображений за раз
def show_videos_and_images(start_idx):
    # Очистка предыдущих видео и изображений
    for label in video_labels:
        label.destroy()
    for label in image_labels:
        label.destroy()

    video_labels.clear()
    image_labels.clear()

    # Отображаем 5 видео и их изображения
    for i in range(start_idx, min(start_idx + 5, len(df))):
        video_link = df.iloc[i, 0]
        image_id = df.iloc[i, 1]

        video_label = tk.Label(frame)
        video_label.grid(row=i - start_idx, column=0)
        video_labels.append(video_label)

        image_label = tk.Label(frame)
        image_label.grid(row=i - start_idx, column=1)
        image_labels.append(image_label)

        # Загружаем видео в отдельном потоке, чтобы не блокировать интерфейс
        threading.Thread(target=play_video, args=(i, video_label), daemon=True).start()

        # Отображаем фото
        display_images(image_id, image_label)

# Функция для переключения на следующую группу видео
def next_group():
    global current_row
    if current_row + 5 < len(df):
        current_row += 5
        show_videos_and_images(current_row)
    else:
        messagebox.showinfo("Конец", "Больше нет данных!")

# Кнопка для переключения на следующую группу
next_button = tk.Button(root, text="Следующие 5", command=next_group)
next_button.pack()

# Начинаем с первой группы
current_row = 0
show_videos_and_images(current_row)

# Запускаем интерфейс
root.mainloop()
