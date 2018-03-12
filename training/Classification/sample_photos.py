import os
import glob
import shutil
import random
positives=glob.glob("/Users/ben/Dropbox/GoogleCloud/Training/Positives/*.jpg")
negatives=glob.glob("/Users/ben/Dropbox/GoogleCloud/Training/Negatives/*.jpg")

random.shuffle(positives)
random.shuffle(negatives)

positives_training=positives[:100]
positives_testing=positives[100:150]

negatives_training=negatives[:100]
negatives_testing=negatives[100:150]

for x in positives_training:
    bname=os.path.basename(x)
    dst=os.path.join("/Users/ben/Documents/DeepMeerkat/training/Classification/Samples/Training/Positives",bname)
    shutil.copy(x, dst)

for x in positives_testing:
    bname=os.path.basename(x)
    dst=os.path.join("/Users/ben/Documents/DeepMeerkat/training/Classification/Samples/Testing/Positives",bname)
    shutil.copy(x, dst)

for x in negatives_training:
    bname=os.path.basename(x)
    dst=os.path.join("/Users/ben/Documents/DeepMeerkat/training/Classification/Samples/Training/Negatives",bname)
    shutil.copy(x, dst)
    
for x in negatives_testing:
    bname=os.path.basename(x)
    dst=os.path.join("/Users/ben/Documents/DeepMeerkat/training/Classification/Samples/Testing/Negatives",bname)
    shutil.copy(x, dst)
    