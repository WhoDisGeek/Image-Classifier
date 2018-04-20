import os
from PIL import Image
from jsonlines import open, jsonlines
import os
import sys
import argparse
import operator
import random

def create_image(image, filename):
    img = Image.new('RGB', (256, 256), "white")
    pixels = img.load()

    x = -1
    y = -1

    for stroke in image:
        for i in range(len(stroke[0])):
            if x != -1:
                for point in get_line(stroke[0][i], stroke[1][i], x, y):
                    pixels[point[0], point[1]] = (0, 0, 0)
            pixels[stroke[0][i], stroke[1][i]] = (0, 0, 0)
            x = stroke[0][i]
            y = stroke[1][i]
        x = -1
        y = -1
    img.save(filename)


def get_line(x1, y1, x2, y2):
    points = []
    issteep = abs(y2 - y1) > abs(x2 - x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1
    deltay = abs(y2 - y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2 + 1):
        if issteep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()
    return points


rootDir = sys.argv[1]  # root dir
current_dataset = os.path.join(rootDir, 'ndjson')  # path where ndjson files exists relative to root dir
final_dir = os.path.join(rootDir, 'CNN-Total')  # final path where images are store relative to root dir

os.chdir(rootDir)
labels = []
for ndj_file in os.listdir(current_dataset):
    folder = os.path.join(final_dir, os.path.splitext(os.path.basename(ndj_file))[0])
    os.makedirs(folder, exist_ok=True)
    count = 0
    print(folder)
    with jsonlines.open(os.path.join(current_dataset, ndj_file), mode='r') as reader:
        for obj in reader:
            if random.randint(1, 3) == 2:
                create_image(obj['drawing'], os.path.join(folder, str(count) + '.jpg'))
                count = count + 1
                if count > 50000:
                    break
    print(ndj_file, 'done')