import tkinter as tk
from tkinter import ttk
import cv2
import os
import threading
import time
import csv
import pandas as pd
import re
from PIL import Image, ImageTk

class VideoPlayer:
    def __init__(self, root, csv_file_path):
        self.root = root
        self.root.title("Просмотр видео и изображений")
        self.root.state('zoomed')  # Максимизировать окно
        
        # Пути к папкам
        self.video_dir = r"D:\Hackatons\WB_Hack\temp"
        self.images_dirs = {
            "images_1": r"D:\Hackatons\data\images_1",
            "images_2-3": r"D:\Hackatons\data\images_2-3",
            "images_4-5": r"D:\Hackatons\data\images_4-5"
        }
        
        # Загрузка данных из CSV
        self.data = pd.read_csv(csv_file_path) if csv_file_path.endswith('.csv') else self.load_from_custom_format(csv_file_path)
        
        # Текущий индекс начала отображаемых видео
        self.current_index = 0
        
        # Создание фреймов для видео и изображений
        self.create_ui()
        
        # Кнопки навигации
        self.create_navigation_buttons()
        
        # Словарь для хранения потоков видео
        self.video_threads = {}
        self.running = True
        
        # Запуск первых 5 видео
        self.show_videos()

    def load_from_custom_format(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Используем регулярное выражение для поиска URL и числа
                match = re.match(r'(\d+)\s*(https://[^\s]+)\s*(\d+)', line)
                if match:
                    index = int(match.group(1))
                    url = match.group(2)
                    value = int(match.group(3))
                    data.append({'index': index, 'url': url, 'value': value})
        return pd.DataFrame(data)

    def create_ui(self):
        # Основной контейнер с прокруткой
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Создаем фреймы для каждой строки видео+изображения
        self.video_frames = []
        for i in range(5):
            row_frame = ttk.Frame(self.main_frame)
            row_frame.pack(fill=tk.BOTH, expand=True, pady=5)
            
            # Фрейм для видео
            video_frame = ttk.LabelFrame(row_frame, text=f"Видео {i}")
            video_frame.pack(side=tk.LEFT, padx=5)
            
            # Label для отображения видео
            video_label = ttk.Label(video_frame)
            video_label.pack(padx=5, pady=5)
            
            # Фрейм для изображений
            images_frame = ttk.Frame(row_frame)
            images_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            # Сохраняем ссылки на элементы интерфейса
            self.video_frames.append({
                'row_frame': row_frame,
                'video_frame': video_frame,
                'video_label': video_label,
                'images_frame': images_frame,
                'image_labels': []  # Будем добавлять по мере необходимости
            })

    def create_navigation_buttons(self):
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        
        # Кнопка "Предыдущие 5"
        self.prev_button = ttk.Button(button_frame, text="Предыдущие 5", 
                                     command=self.show_previous, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        # Кнопка "Следующие 5"
        self.next_button = ttk.Button(button_frame, text="Следующие 5", 
                                     command=self.show_next)
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        # Метка с информацией о текущем диапазоне
        self.info_label = ttk.Label(button_frame, 
                                   text=f"Отображаются видео {self.current_index+1}-{min(self.current_index+5, len(self.data))}")
        self.info_label.pack(side=tk.LEFT, padx=20)

    def show_videos(self):
        # Остановить текущие потоки видео
        self.stop_video_threads()
        
        # Очистить все изображения
        for frame_data in self.video_frames:
            for label in frame_data['image_labels']:
                label.destroy()
            frame_data['image_labels'] = []
        
        # Для каждого фрейма из 5, показать соответствующее видео и изображения
        for i in range(5):
            idx = self.current_index + i
            if idx < len(self.data):
                # Получаем данные для текущей строки
                row = self.data.iloc[idx]
                
                # Название видео файла
                video_file = os.path.join(self.video_dir, f"output_video_{idx}.mp4")
                
                # ID из URL (второй столбец)
                url = row.iloc[1] if isinstance(row, pd.Series) else row['url']
                
                # Извлекаем ID из URL (предполагаем, что это часть URL между последним / и .m3u8)
                long_id = re.search(r'/([^/]+)/index\.m3u8$', url)
                if long_id:
                    long_id = long_id.group(1)
                else:
                    # Если не удалось извлечь ID, используем пустую строку
                    long_id = ""
                
                # Обновляем заголовок видео фрейма
                self.video_frames[i]['video_frame'].config(text=f"Видео {idx} (ID: {long_id})")
                
                # Запускаем видео в отдельном потоке
                if os.path.exists(video_file):
                    thread = threading.Thread(target=self.play_video, 
                                             args=(video_file, self.video_frames[i]['video_label'], idx))
                    thread.daemon = True
                    thread.start()
                    self.video_threads[idx] = thread
                else:
                    # Если видео не существует, показываем сообщение об ошибке
                    self.video_frames[i]['video_label'].config(text=f"Видео {idx} не найдено")
                
                # Загружаем и отображаем изображения для данного ID
                self.load_images(long_id, self.video_frames[i]['images_frame'], i)
            else:
                # Очищаем фрейм, если нет соответствующего видео
                self.video_frames[i]['video_label'].config(image='')
                self.video_frames[i]['video_frame'].config(text=f"Видео {i} (нет данных)")
        
        # Обновляем состояние кнопок и информационную метку
        self.update_navigation_state()

    def load_images(self, long_id, images_frame, frame_index):
        # Найти изображения в каталогах
        image_files = []
        
        # Проверяем в каждом из каталогов
        for dir_name, dir_path in self.images_dirs.items():
            dir_path_with_id = os.path.join(dir_path, long_id)
            if os.path.exists(dir_path_with_id):
                # Если такой каталог существует, ищем в нем изображения
                files = [f for f in os.listdir(dir_path_with_id) if f.endswith('.webp')]
                image_files.extend([os.path.join(dir_path_with_id, f) for f in files])
        
        # Сортируем файлы по имени
        image_files.sort()
        
        # Создаем горизонтальный контейнер для изображений, если есть изображения
        if image_files:
            # Создаем холст с горизонтальной прокруткой для изображений
            canvas = tk.Canvas(images_frame)
            scrollbar = ttk.Scrollbar(images_frame, orient="horizontal", command=canvas.xview)
            scrollable_frame = ttk.Frame(canvas)
            
            canvas.configure(xscrollcommand=scrollbar.set)
            
            canvas.pack(side=tk.TOP, fill=tk.X, expand=True)
            scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            
            # Загружаем и отображаем каждое изображение
            for img_file in image_files:
                try:
                    # Загружаем изображение и изменяем его размер
                    img = Image.open(img_file)
                    img = img.resize((150, 150), Image.LANCZOS)
                    img_tk = ImageTk.PhotoImage(img)
                    
                    # Создаем метку для изображения
                    img_label = ttk.Label(scrollable_frame, image=img_tk)
                    img_label.image = img_tk  # Сохраняем ссылку
                    img_label.pack(side=tk.LEFT, padx=5, pady=5)
                    
                    # Добавляем метку в список для последующей очистки
                    self.video_frames[frame_index]['image_labels'].append(img_label)
                    
                except Exception as e:
                    print(f"Ошибка при загрузке изображения {img_file}: {e}")
        else:
            # Если изображений нет, показываем сообщение
            no_img_label = ttk.Label(images_frame, text=f"Нет изображений для ID: {long_id}")
            no_img_label.pack(padx=5, pady=5)
            self.video_frames[frame_index]['image_labels'].append(no_img_label)

    def play_video(self, video_file, label, video_idx):
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            label.config(text=f"Не удалось открыть видео {video_idx}")
            return
        
        # Получаем размеры видео
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Устанавливаем размер отображения
        display_width = 320
        display_height = int(height * (display_width / width))
        
        while self.running and video_idx in self.video_threads:
            ret, frame = cap.read()
            if ret:
                # Изменяем размер кадра
                frame = cv2.resize(frame, (display_width, display_height))
                
                # Конвертируем цвет из BGR в RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Создаем изображение для Tkinter
                img = Image.fromarray(frame_rgb)
                img_tk = ImageTk.PhotoImage(image=img)
                
                # Обновляем изображение в метке
                label.config(image=img_tk)
                label.image = img_tk  # Сохраняем ссылку
                
                # Задержка для контроля скорости воспроизведения
                time.sleep(0.03)  # Примерно 30 FPS
            else:
                # Если достигнут конец видео, начинаем заново
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Освобождаем ресурсы
        cap.release()

    def stop_video_threads(self):
        # Очищаем словарь потоков, что приведет к остановке циклов воспроизведения
        self.video_threads.clear()
        # Даем немного времени для остановки потоков
        time.sleep(0.1)

    def show_next(self):
        if self.current_index + 5 < len(self.data):
            self.current_index += 5
            self.show_videos()

    def show_previous(self):
        if self.current_index - 5 >= 0:
            self.current_index -= 5
            self.show_videos()

    def update_navigation_state(self):
        # Обновляем состояние кнопок навигации
        self.prev_button.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if self.current_index + 5 < len(self.data) else tk.DISABLED)
        
        # Обновляем информационную метку
        self.info_label.config(text=f"Отображаются видео {self.current_index+1}-{min(self.current_index+5, len(self.data))} из {len(self.data)}")

    def on_closing(self):
        # Останавливаем все потоки воспроизведения
        self.running = False
        self.stop_video_threads()
        self.root.destroy()

if __name__ == "__main__":
    # Путь к файлу CSV
    csv_path = r"D:\Hackatons\WB_Hack\results.csv"  # Замените на реальный путь к вашему файлу
    
    root = tk.Tk()
    app = VideoPlayer(root, csv_path)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()