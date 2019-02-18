import cv2
import json
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import sys

sys.path.append('C:\\Users\\flavi\\PycharmProjects\\ctw-traffic-signs')
import settings

from pythonapi import anno_tools
with open(settings.TRAIN_AUGMENTED) as f:
    lines = f.readlines()
    for line in lines:
        anno = json.loads(line)
        if anno['image_id'] == "0600223":
            to_show = anno
            break
path = os.path.join(settings.TRAINVAL_AUGMENTED_DIR, to_show['image_id']+'.png')
assert os.path.exists(path), 'file not exists: {}'.format(path)
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(16, 16))
ax = plt.gca()
plt.imshow(img)

for instance in anno_tools.each_char(anno):
    color = (0, 1, 0) if instance['is_chinese'] else (1, 0, 0)
    ax.add_patch(patches.Polygon(instance['polygon'], fill=False, color=color))
'''
for ignore in anno['ignore']:
    color = (1, 1, 0)
    ax.add_patch(patches.Polygon(ignore['polygon'], fill=False, color=color))
'''
plt.show()