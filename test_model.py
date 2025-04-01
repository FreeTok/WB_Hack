# from PIL import Image

# # Crop object form image
# def crop_objects(image, detections):
#     cropped_images = []
#     for detection in detections:
#         xmin, ymin, xmax, ymax = detection
#         cropped_image = image.crop((xmin, ymin, xmax, ymax))
#         cropped_images.append(cropped_image)
#     return cropped_images


# img = Image.open('Input/A60/image.jpg') # Load the image
# crop_objects(img, [(574.1428,  172.4365,  876.4652, 1095.7454)])[0].save('new_image.jpg') # Save the image


import json
import os
import subprocess
import time

from test_comparison import compare_video_image

def check(i):
    startTime = time.time()
    images = {}

    with open(r'C:\Users\FreeTok\Desktop\data\videos.json') as f:
        d = json.load(f)[i]
        print(d)
        video_url = d['path_url']
        output_video = 'temp/output_video.mp4'
        command = [
            'ffmpeg',
            '-i', video_url,
            '-c', 'copy',  # Копируем потоки без перекодирования (быстрее)
            '-v', 'error', 
            output_video
        ]
        subprocess.run(command)
        
        for id in d['nm_ids']:
            smallImages = []

            smallImages.append(rf'C:\Users\FreeTok\Desktop\data\images_1\{id}\1.webp')
            smallImages.append(rf'C:\Users\FreeTok\Desktop\data\images_2-3\{id}\2.webp')
            smallImages.append(rf'C:\Users\FreeTok\Desktop\data\images_2-3\{id}\3.webp')
            smallImages.append(rf'C:\Users\FreeTok\Desktop\data\images_4-5\{id}\4.webp')
            smallImages.append(rf'C:\Users\FreeTok\Desktop\data\images_4-5\{id}\5.webp')

            images.update({id : smallImages})

        # compare_video_image( , capPath=output_video)

    for imageid in images:
        print(f'############### {imageid}')
        for image in images[imageid]:
            # print(image)
            compare_video_image(image, output_video)

    print(f'lasted: {time.time() - startTime}')
    os.remove(output_video)

check(8)






# video_url = 'https://wibes-07.wbbasket.ru/331c51ce-0366-11f0-9a37-36c86be66d90/353134383736-wibe-video-1080/index.m3u8'
# output_file = 'output_video.mp4'

# # Запускаем ffmpeg для скачивания и конвертации в MP4
# command = [
#     'ffmpeg',
#     '-i', video_url,
#     '-c', 'copy',  # Копируем потоки без перекодирования (быстрее)
#     output_file
# ]

# subprocess.run(command)