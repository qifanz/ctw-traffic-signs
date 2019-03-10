# ctw-traffic-signs
This is a project for the interview questions. The main goal is to detect chinese characters on traffic signs.
## Dataset
In this project the dataset Traffic Panel Database is used. This dataset is elaborated by National Nature Science Foundation of China(NSFC). 
The TPD includes 2329 traffic images containing various types of traffic panels, The images are collected under a wide variety of different situations, such as weather condition, illumination, different surroundings, partial occluson and so on.

## Preparation
You should first download dataset http://www.nlpr.ia.ac.cn/pal/trafficdata/panel.html (dataset B which is labelled).
Randomly split them according to your preferred proportion (I used 85% ~ 15%). Put them into /data/image/train and /data/image/val.
Randomly split annotation file and put them into /data/annotation/train and /data/annotation/val.
Verify the path in /prepare/settings_qifan.py correponds to the correct path.

## Preprocessing
This project is going to use Yolo-v3 based on Darknet. You should run in /prapare:
``` python preprocess.py ```
This script will process the original annotations and create the Darknet compatible annotations in /data/annotation/train_processed and /data/annotation/val_processed.
You can also adapt the augmentation.py (originally created to augment the ctw dataset by me) script to augment the dataset. In order to train faster, this step is not applied.





# Legacy version
## Preparation
You should first download trainval data from https://ctwdataset.github.io/, and put them into /data/images/trainval.  
Similarly you should download the annotation files and put them into /data/annotations.  
You can then visualize an image with the corresponding bounding boxes by running in /prepare:  
```  python draw_bb.py  ```  
If you wish to change an image, you could simply change the id in line 16. (This is just a simple tester)
  
## Preprocessing
This step mainly consists of filtering data that contains traffic signs and data augmentation.  
To do so, you could run in /prepare:  
```  python augmentation.py  ```  
Augmentated dataset will be generated in /data/images/augmented. And a new jsonl annotations file will be generated in /data/annotations/augmented. This step could take very long time. (up to several hours) 
  
The second digit of image_id represents the type of augmentation:  
- 1 for rainy
- 2 for snowy
- 3 for illumination plus
- 4 for illumination minus
- 5 for horizontal flipping
- 6 for rotation
  
 The bounding box are already recalculated for augmented images. You could use draw_bb.py to visualize them.
