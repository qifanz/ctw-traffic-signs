from os import listdir
from os.path import isfile, join
import numpy as np
import pprint
import math
import shutil

import settings_qifan as setting


def getDarknetCoord(coords):
    for i in range(0, 8):
        coords[i] = int(coords[i])
    x = (coords[0] + coords[2] + coords[4] + coords[6]) / (4 *setting.IMAGE_W)
    y = (coords[1] + coords[3] + coords[5] + coords[7]) / (4 *setting.IMAGE_H)
    w = (coords[2] - coords[0])/setting.IMAGE_W
    h = (coords[5] - coords[3])/setting.IMAGE_H
    return x, y, w, h


def createDarknetLine (coords) :
    line = "0 "
    for i in range(4):
        line = line +str(round(coords[i],4))+" "
    line = line +'\n'
    return line

f_train = open(join('..','data',"train.txt"),"w+")
trainAnnots = [f for f in listdir(setting.TRAIN_ANNOT_DIR) if isfile(join(setting.TRAIN_ANNOT_DIR, f))]
for annot in trainAnnots:
    toWrite = []
    with open(join(setting.TRAIN_ANNOT_DIR, annot)) as f:
        lines = f.readlines()
        filteredLines = []
        toWrite = []
        for line in lines:
            if not line == '\n':
                filteredLines.append(line)
        i = 0
        while i < len(filteredLines):
            try:
                print(filteredLines[i + 1])
                className = int(filteredLines[i + 1])
            except ValueError:
                coords = filteredLines[i].strip('\n').split(',')
                toWrite.append(getDarknetCoord(coords))
            i = i + 2
        print(toWrite)
        f1 = open(join(setting.PROCESSED_TRAIN_ANNOT_DIR,annot),"w+")
        for j in range(0,len(toWrite)):
            f1.write (createDarknetLine(toWrite[j]))
        f1.close()
        f_train.write("data/traffic/"+annot.replace("txt","jpg")+'\n')
f_train.close()

f_val = open(join('..','data',"val.txt"),"w+")

valAnnots = [f for f in listdir(setting.VAL_ANNOT_DIR) if isfile(join(setting.VAL_ANNOT_DIR, f))]
for annot in valAnnots:
    toWrite = []
    with open(join(setting.VAL_ANNOT_DIR, annot)) as f:
        lines = f.readlines()
        filteredLines = []
        toWrite = []
        for line in lines:
            if not line == '\n':
                filteredLines.append(line)
        i = 0
        while i < len(filteredLines):
            try:
                print(filteredLines[i + 1])
                className = int(filteredLines[i + 1])
            except ValueError:
                coords = filteredLines[i].strip('\n').split(',')
                toWrite.append(getDarknetCoord(coords))
            i = i + 2
        print(toWrite)
        f1 = open(join(setting.PROCESSED_VAL_ANNOT_DIR,annot),"w+")
        for j in range(0,len(toWrite)):
            f1.write (createDarknetLine(toWrite[j]))
        f1.close()
        f_val.write("data/traffic/"+annot.replace("txt","jpg")+'\n')
f_val.close()
