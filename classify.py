# 2017 load pictures and analyze
# https://github.com/tspannhw/mxnet_rpi/blob/master/analyze.py
import time
import sys
import datetime
import subprocess
import sys
import urllib2
import os
import datetime
import traceback
import math
import random, string
import base64
import json
from time import gmtime, strftime
import mxnet as mx
import inception_predict
import numpy as np
import cv2
import math
import random, string
import time

from time import gmtime, strftime
start = time.time()
cap = cv2.VideoCapture(0)

packet_size=3000

def randomword(length):
 return ''.join(random.choice(string.lowercase) for i in range(length))

#while True:

# Create unique image name
uniqueid = 'mxnet_uuid_{0}_{1}'.format(randomword(3),strftime("%Y%m%d%H%M%S",gmtime()))

ret, frame = cap.read()

imgdir = 'images/'
filename = 'tx1_image_{0}_{1}.jpg'.format(randomword(3),strftime("%Y%m%d%H%M%S",gmtime()))
cv2.imwrite(imgdir + filename, frame)

# Run inception prediction on image
try:
     topn = inception_predict.predict_from_local_file(imgdir + filename, N=5)
except:
     errorcondition = "true"

# CPU Temp
f = open("/sys/devices/virtual/thermal/thermal_zone1/temp","r")
cputemp = str( f.readline() )
cputemp = cputemp.replace('\n','')
cputemp = cputemp.strip()
cputemp = str(round(float(cputemp)) / 1000)
cputempf = str(round(9.0/5.0 * float(cputemp) + 32))
f.close()

# GPU Temp
f = open("/sys/devices/virtual/thermal/thermal_zone2/temp","r")
gputemp = str( f.readline() )
gputemp = gputemp.replace('\n','')
gputemp = gputemp.strip()
gputemp = str(round(float(gputemp)) / 1000)
gputempf = str(round(9.0/5.0 * float(gputemp) + 32))
f.close()

# Face Detect
p = os.popen('/media/nvidia/96ed93f9-7c40-4999-85ba-3eb24262d0a5/jetson-inference-master/build/aarch64/bin/facedetect.sh ' + filename).read()
face = p.replace('\n','|')
face = face.strip()

# NVidia Image Net Classify
p2 = os.popen('/media/nvidia/96ed93f9-7c40-4999-85ba-3eb24262d0a5/jetson-inference-master/build/aarch64/bin/runclassify.sh ' + filename).read()
imagenet = p2.replace('\n','|')
imagenet = imagenet.strip()

# 5 MXNET Analysis
top1 = str(topn[0][1])
top1pct = str(round(topn[0][0],3) * 100)

top2 = str(topn[1][1])
top2pct = str(round(topn[1][0],3) * 100)

top3 = str(topn[2][1])
top3pct = str(round(topn[2][0],3) * 100)

top4 = str(topn[3][1])
top4pct = str(round(topn[3][0],3) * 100)

top5 = str(topn[4][1])
top5pct = str(round(topn[4][0],3) * 100)

end = time.time()
# face[-4096:]
row = { 'uuid': uniqueid,  'top1pct': top1pct, 'top1': top1, 'top2pct': top2pct, 'top2': top2,'top3pct': top3pct, 'top3': top3,'top4pct': top4pct,'top4': top4, 'top5pct': top5pct,'top5': top5, 'cputemp': cputemp, 'gputemp': gputemp, 'imagefilename': filename, 'gputempf': gputempf, 'cputempf': cputempf, 'runtime': str(round(end - start)), 'facedetect': face, 'imagenet': imagenet }
json_string = json.dumps(row)

print (json_string )
