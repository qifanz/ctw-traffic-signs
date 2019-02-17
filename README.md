# ctw-traffic-signs
This is a project for the interview questions. The main goal is to detect chinese characters on traffic signs.

## Preparation
You should first download trainval data from https://ctwdataset.github.io/, and put them into /data/images/trainval.
Similarly you should download the annotation files and put them into /data/annotations.
You can then visualize them by running in /prepare:
```  python draw_bb.py ``` 
  
## Preprocessing
This step mainly consists of filtering data that contains traffic signs and data augmentation.
To do so, you could run in /prepare:
``` python augmentation.py ```
Augmentated dataset will be generated in /data/images/augmented. And a new jsonl annotations file will be generated in /data/annotations/augmented. This step could take very long time. (up to several hours)
