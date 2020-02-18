# Write Python3 code here 
import os 
import cv2 
import numpy as np 
import tensorflow as tf 
import sys
import argparse
  
# This is needed since the notebook is stored in the object_detection folder. 
sys.path.append("..") 
  
# Import utilites 
from utils import label_map_util 
from utils import visualization_utils as vis_util 

parser = argparse.ArgumentParser(description='Classify')
parser.add_argument("--inference_graph", help = "path to the directory where frozen_inference_graph is stored")
parser.add_argument("--training_dir", help = "path to the directory where labelmap is stored")
parser.add_argument("--vid", help = "path to the video file to use")
parser.add_argument("--cam", help = "index of usb camera to use")
parser.add_argument("--split", action='store_true' ,help = "if defined will split the image into rgb and use r")
parser.add_argument("--flip", action='store_true' ,help = "if defined will flip the image vertically")

args, leftovers = parser.parse_known_args()

if (args.cam is not None) and (args.vid is not None):
    print("Cannot define both vid and cam. Only one must be used")
    exit

vid_index = None
if args.cam is not None:
    # convert string to int
    vid_index = int(args.cam)
elif args.vid is not None:
    vid_index = args.vid
else:
    print("Must define vid or cam")
    exit

split_image = args.split
flip_image = args.flip
  
# Grab path to current working directory 
CWD_PATH = os.getcwd() 

# Path to frozen detection graph .pb file, which contains the model that is used 
# for object detection. 
PATH_TO_CKPT = os.path.join(CWD_PATH, args.inference_graph, 'frozen_inference_graph.pb') 
  
# Path to label map file 
PATH_TO_LABELS = os.path.join(CWD_PATH, args.training_dir, 'labelmap.pbtxt')

# Number of classes the object detector can identify 
NUM_CLASSES = 2
  
# Load the label map. 
# Label maps map indices to category names, so that when our convolution 
# network predicts `5`, we know that this corresponds to `king`. 
# Here we use internal utility functions, but anything that returns a 
# dictionary mapping integers to appropriate string labels would be fine 
label_map = label_map_util.load_labelmap(PATH_TO_LABELS) 
categories = label_map_util.convert_label_map_to_categories( 
        label_map, max_num_classes = NUM_CLASSES, use_display_name = True) 
category_index = label_map_util.create_category_index(categories) 

# Load the Tensorflow model into memory. 
detection_graph = tf.Graph() 
with detection_graph.as_default(): 
    od_graph_def = tf.GraphDef() 
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid: 
        serialized_graph = fid.read() 
        od_graph_def.ParseFromString(serialized_graph) 
        tf.import_graph_def(od_graph_def, name ='') 
  
    sess = tf.Session(graph = detection_graph) 
  
# Define input and output tensors (i.e. data) for the object detection classifier 
  
# Input tensor is the image 
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0') 
  
# Output tensors are the detection boxes, scores, and classes 
# Each box represents a part of the image where a particular object was detected 
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0') 
  
# Each score represents level of confidence for each of the objects. 
# The score is shown on the result image, together with the class label. 
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0') 
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0') 
  
# Number of objects detected 
num_detections = detection_graph.get_tensor_by_name('num_detections:0') 

cap = cv2.VideoCapture(vid_index)
frame_counter = 0
while(True):
    # Load image from usb camera using OpenCV and 
    # expand image dimensions to have shape: [1, None, None, 3] 
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = cap.read()
    frame_counter += 1

    if args.vid is not None:
        if frame_counter >= cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
            frame_counter = 0
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)
    else:
        if (ret):
            if (split_image):
                image_r = frame[:, :, 1]
                frame = cv2.cvtColor(image_r,cv2.COLOR_GRAY2RGB)
            if (flip_image):
                frame = cv2.flip(frame,0)

            image_expanded = np.expand_dims(frame, axis = 0) 
            
            # Perform the actual detection by running the model with the image as input 
            (boxes, scores, classes, num) = sess.run( 
                [detection_boxes, detection_scores, detection_classes, num_detections], 
                feed_dict ={image_tensor: image_expanded}) 
            
            # Draw the results of the detection (aka 'visualize the results') 
            
            vis_util.visualize_boxes_and_labels_on_image_array( 
                frame, 
                np.squeeze(boxes), 
                np.squeeze(classes).astype(np.int32), 
                np.squeeze(scores), 
                category_index, 
                use_normalized_coordinates = True, 
                line_thickness = 8, 
                min_score_thresh = 0.60) 
        
            # All the results have been drawn on the image. Now display the image. 
            cv2.imshow('Object detector', frame) 

    k = cv2.waitKey(1) 
    if (k == ord('q')):
        break
  
# Clean up 
cap.release()
cv2.destroyAllWindows()