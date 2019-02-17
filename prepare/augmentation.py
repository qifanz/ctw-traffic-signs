import cv2
import json
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import pprint

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

def augment_image (annot,f2) :
    old_id = annot['image_id']
    path = os.path.join(settings.TRAINVAL_IMAGE_DIR, annot['file_name'])
    assert os.path.exists(path), 'file not exists: {}'.format(path)

    img_original = cv2.imread(path)
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    img_rain = add_rain(img_original)
    rain_id = old_id[0]+'1'+old_id[2:]
    annot['image_id']=rain_id
    f2.write(json.dumps(annot)+'\n')
    plt.figure(figsize=(16, 16))
    ax = plt.gca()
    plt.imsave('../data/images/augmented/'+rain_id+'.png',img_rain)
    plt.close()

    img_original = cv2.imread(path)
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    img_snow = add_snow(img_original)
    snow_id = old_id[0]+'2'+old_id[2:]
    annot['image_id']=snow_id
    f2.write(json.dumps(annot)+'\n')
    plt.figure(figsize=(16, 16))
    ax = plt.gca()
    plt.imsave('../data/images/augmented/' + snow_id + '.png',
               img_snow)
    plt.close()

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



with open(settings.TRAIN) as f:
    f1 = open("../data/annotations/train_processed.jsonl", "w+")
    f2 = open("../data/annotations/augmented/train_processed.jsonl","w+")
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
            #augment_image(annot,f2)
    print(len(list_filtered))
    for line in list_filtered:
        f1.write(json.dumps(line)+'\n')
        #print(json.dumps(line))
    f.close()

