import glob 
import numpy as np 

files = glob.glob("text_c10/*/*.txt")
blobs = []

for file in files:
    image_name = file.split("/")[-1].split(".")[0]
    image_id = image_name.split("_")[-1]
    f = open(file)
    lines = f.read().split("\n")
    for line in lines:
        if line:
            blobs.append("%s\t%s.jpg\t%s" % (image_id, image_name, line))

np.random.shuffle(blobs)

g = open("trainval.tsv", "wb")
h = open("test.tsv", "wb")

for i, blob in enumerate(blobs):
    if i < 80000: 
        g.write(blob)
        g.write("\n")
    else:
        h.write(blob)
        h.write("\n")
