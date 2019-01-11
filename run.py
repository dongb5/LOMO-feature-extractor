import os

import json
import cv2

import lomo

data_path = 'data'
img_list = os.listdir(data_path)
if len(img_list) == 0:
    print('Data directory is empty.')
    exit()

with open('config.json', 'r') as f:
    config = json.load(f)

for img_name in img_list:
    if img_name == '.gitkeep':
        continue

    img = cv2.imread(os.path.join(data_path, img_name))

    lomo_desc = lomo.LOMO(img, config)

    print('Lomo feature size:', lomo_desc.shape[0])
