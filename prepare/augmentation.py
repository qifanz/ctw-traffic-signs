import cv2
import json
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import pprint
import math

sys.path.append('C:\\Users\\flavi\\PycharmProjects\\ctw-traffic-signs')
import settings

from pythonapi import anno_tools




def add_snow(image):
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    image_HLS = np.array(image_HLS, dtype = np.float64)
    brightness_coefficient = 2
    snow_point=500 ## increase this for more snow
    image_HLS[:,:,1][image_HLS[:,:,1]<snow_point] = image_HLS[:,:,1][image_HLS[:,:,1]<snow_point]*brightness_coefficient ## scale pixel values up for channel 1(Lightness)
    image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 ##Sets all values above 255 to 255
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
    return image_RGB

def generate_random_lines(imshape, slant, drop_length):
    drops = []
    for i in range(1500):  ## If You want heavy rain, try increasing this
        if slant < 0:
            x = np.random.randint(slant, imshape[1])
        else:
            x = np.random.randint(0, imshape[1] - slant)
        y = np.random.randint(0, imshape[0] - drop_length)
        drops.append((x, y))
    return drops

def add_luminosity(image):
    image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)  ## Conversion to HLS
    image_HLS = np.array(image_HLS, dtype=np.float64)
    brightness_coefficient = 1.3
    image_HLS[:, :, 1] = image_HLS[:, :,
                         1] * brightness_coefficient  ## scale pixel values down for channel 1(Lightness)
    image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255  ##Sets all values above 255 to 255
    image_HLS = np.array(image_HLS, dtype=np.uint8)
    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)  ## Conversion to RGB
    return image_RGB

def decrease_luminosity(image):
    image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)  ## Conversion to HLS
    image_HLS = np.array(image_HLS, dtype=np.float64)
    brightness_coefficient = 0.85
    image_HLS[:, :, 1] = image_HLS[:, :,
                         1] * brightness_coefficient  ## scale pixel values down for channel 1(Lightness)
    image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255  ##Sets all values above 255 to 255
    image_HLS = np.array(image_HLS, dtype=np.uint8)
    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)  ## Conversion to RGB
    return image_RGB


def add_rain(image):
    image2=image
    imshape = image2.shape
    slant_extreme = 10
    slant = np.random.randint(-slant_extreme, slant_extreme)
    drop_length = 20
    drop_width = 2
    drop_color = (200, 200, 200)  ## a shade of gray
    rain_drops = generate_random_lines(imshape, slant, drop_length)

    for rain_drop in rain_drops:
        cv2.line(image2, (rain_drop[0], rain_drop[1]), (rain_drop[0] + slant, rain_drop[1] + drop_length), drop_color,
                 drop_width)
    image2 = cv2.blur(image2, (7, 7))  ## rainy view are blurry

    brightness_coefficient = 0.9  ## rainy days are usually shady
    image_HLS = cv2.cvtColor(image2, cv2.COLOR_RGB2HLS)  ## Conversion to HLS
    image_HLS[:, :, 1] = image_HLS[:, :,
                         1] * brightness_coefficient  ## scale pixel values down for channel 1(Lightness)
    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)  ## Conversion to RGB
    return image_RGB

def augment_rain (path,old_id):
    img_original = cv2.imread(path)
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    img_rain = add_rain(img_original)
    rain_id = old_id[0] + '1' + old_id[2:]
    annot['image_id'] = rain_id
    f2.write(json.dumps(annot) + '\n')
    plt.figure(figsize=(16, 16))
    ax = plt.gca()
    plt.imsave('../data/images/augmented/' + rain_id + '.png', img_rain)
    plt.close()

def augment_snow (path,old_id):
    img_original = cv2.imread(path)
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    img_snow = add_snow(img_original)
    snow_id = old_id[0] + '2' + old_id[2:]
    annot['image_id'] = snow_id
    f2.write(json.dumps(annot) + '\n')
    plt.figure(figsize=(16, 16))
    ax = plt.gca()
    plt.imsave('../data/images/augmented/' + snow_id + '.png',
               img_snow)
    plt.close()

def augment_lum_plus (path,old_id):
    img_original = cv2.imread(path)
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    img_lumplus = add_luminosity(img_original)
    lumplus_id = old_id[0] + '3' + old_id[2:]
    annot['image_id'] = lumplus_id
    f2.write(json.dumps(annot) + '\n')
    plt.figure(figsize=(16, 16))
    ax = plt.gca()
    plt.imsave('../data/images/augmented/' + lumplus_id + '.png', img_lumplus)
    plt.close()

def augment_lum_min (path,old_id):
    img_original = cv2.imread(path)
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    img_lummin = decrease_luminosity(img_original)
    lummin_id = old_id[0] + '4' + old_id[2:]
    annot['image_id'] = lummin_id
    f2.write(json.dumps(annot) + '\n')
    plt.figure(figsize=(16, 16))
    ax = plt.gca()
    plt.imsave('../data/images/augmented/' + lummin_id + '.png', img_lummin)
    plt.close()

def augment_flip_hor(path,old_id,annot):
    img_original = cv2.imread(path)
    img_flip = cv2.flip(img_original, 1)
    new_id = old_id[0] + '5' + old_id[2:]
    cv2.imwrite('../data/images/augmented/' + new_id + '.png', img_flip)
    annotations = annot['annotations']
    for sentence in annotations :
        for instance in sentence:
            xmin=instance['adjusted_bbox'][0]
            w=instance['adjusted_bbox'][2]
            instance['adjusted_bbox'][0] = 2048-xmin #-w NO NEED TO MINUS W !

            w1 = abs(instance['polygon'][0][0] - instance['polygon'][1][0])
            w2 = abs(instance['polygon'][2][0] - instance['polygon'][3][0])

            instance['polygon'][0][0] = 2048 - instance['polygon'][0][0] #- w1
            instance['polygon'][1][0] = 2048 - instance['polygon'][1][0] #- w1

            instance['polygon'][2][0] = 2048 - instance['polygon'][2][0] #- w2
            instance['polygon'][3][0] = 2048 - instance['polygon'][3][0] #- w2
    annot['image_id'] = new_id
    f2.write(json.dumps(annot) + '\n')

def augment_rotate_30 (path,old_id,annot) :
    img_original = cv2.imread(path)
    (h, w) = img_original.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, -30, 1)
    rotated = cv2.warpAffine(img_original, M, (w, h))
   # img_rot = cv2.flip(img_original, 1)
    new_id = old_id[0] + '6' + old_id[2:]
    cv2.imwrite('../data/images/augmented/' + new_id + '.png', rotated)
    annotations = annot['annotations']

    costheta = math.cos(math.pi/6)
    sintheta = math.sin(math.pi/6)

    annotations = annot['annotations']
    for sentence in annotations:
        for instance in sentence:
            x1 = instance['polygon'][0][0] -1024
            y1 = instance['polygon'][0][1] -1024
            x2 = instance['polygon'][1][0] -1024
            y2 = instance['polygon'][1][1] -1024
            x3 = instance['polygon'][2][0] -1024
            y3 = instance['polygon'][2][1] -1024
            x4 = instance['polygon'][3][0] -1024
            y4 = instance['polygon'][3][1] -1024

            instance['polygon'][0][0] = 1024-x1*costheta-y1*sintheta
            instance['polygon'][1][0] = -x2*costheta-y2*sintheta +1024
            instance['polygon'][2][0] = -x3*costheta-y3*sintheta+1024
            instance['polygon'][3][0] = -x4*costheta-y4*sintheta+1024

            instance['polygon'][0][1] = 1024 -x1 * sintheta + y1 * costheta
            instance['polygon'][1][1] = 1024 -x2 * sintheta + y2 * costheta
            instance['polygon'][2][1] = 1024 - x3 * sintheta + y3 * costheta
            instance['polygon'][3][1] = 1024 - x4 * sintheta + y4 * costheta

    annot['image_id'] = new_id
    f2.write(json.dumps(annot) + '\n')

def augment_image (annot,f2) :
    old_id = annot['image_id']
    path = os.path.join(settings.TRAINVAL_IMAGE_DIR, annot['file_name'])
    assert os.path.exists(path), 'file not exists: {}'.format(path)
    #augment_rain(path,old_id)
    #augment_snow(path,old_id)
    #augment_lum_plus(path,old_id)
    #augment_lum_min(path,old_id)
    augment_flip_hor (path,old_id,annot)
    augment_rotate_30(path,old_id,annot)

with open(settings.TRAIN) as f:
    f1 = open(settings.TRAIN_PROCESSED, "w+")
    f2 = open(settings.TRAIN_AUGMENTED,"w+")
    list_filtered=[]
    json_lines = f.readlines()
    for i in range(len(json_lines)):
        annot=json.loads(json_lines[i])
        annotations=annot['annotations']
        id = annot['image_id']
        sentence = annotations[0]
        has_traffic_signs = False
        for word in sentence:
            if (len(word['attributes'])==0) :
                #print ('has traffic signs')
                has_traffic_signs=True
        if (has_traffic_signs):
            #print(i)
            list_filtered.append(annot)
            augment_image(annot,f2)
    print(len(list_filtered))
    for line in list_filtered:
        f1.write(json.dumps(line)+'\n')
        #print(json.dumps(line))
    f.close()

