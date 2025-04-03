# from PIL import Image

# img = Image.open(r'C:\\Users\\FreeTok\\Desktop\\data\\images_1\\236795893\\1.webp')
# print(img)

import json

with open(r'C:\Users\FreeTok\Desktop\data\videos.json') as f:
    d = len(json.load(f))
    print(d)
