from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import os, json, cv2, random
from matplotlib import pyplot as plt
import numpy as np

import tensorflow as tf
from datetime import datetime
from datetime import timedelta
from random import randint
import argparse

# Define and parse input arguments
parser = argparse.ArgumentParser()

parser.add_argument('--image', help='Name of the single image to perform detection on. To run detection on multiple images, use --imagedir',
                    required=True)
args = parser.parse_args()

classname = {
  "0": "overripe",
  "1": "ripe",
  "2": "underripe",
  "3": "unripe",
  
}

def PolygonCroper(img,contour):
    pts=np.array(contour)
    geometry=[]
    for p in pts[0]:
        geometry.append(p[0])
    geometry =np.array(geometry).astype(int)
    ## (1) Crop the bounding rect
    rect = cv2.boundingRect(geometry)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()
    img_crop = cv2.cvtColor(croped, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("C:/Users/User/Documents/EBTECH/EB_Detectron2/cropped.png", croped)
    return img_crop

def Midpoint(contour):
    pts=np.array(contour)
    geometry=[]
    for p in pts[0]:
        geometry.append(p[0])
    geometry =np.array(geometry).astype(int)
    ## (1) Crop the bounding rect
    rect = cv2.boundingRect(geometry)
    x,y,w,h = rect
    point = [(x+(x+w))/2, (y+(y+h))/2]
    return point

def Harvest_Date_Estimate(image):
    # load weights into new model
    model_filepath = 'C:/Users/User/Documents/EBTECH/EB_Detectron2'
    json_file = open(os.path.join(model_filepath,"model5.json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights('C:/Users/User/Documents/EBTECH/EB_Detectron2/weight-epoch-28-valmse-0.27.hdf5')
    print("Loaded model from disk")
    
    # pre-processing
    image =image[...,::-1] # reverse the channels
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = tf.image.resize(input_arr,(224,224))
    input_arr = np.array([input_arr])
    input_arr*1./255
    input_arr=input_arr*1./255
    input_arr
    
    # predict age after anthesis
    predictions = loaded_model.predict(input_arr)
    print(np.squeeze(predictions))
    print(round(float(np.squeeze(predictions))))

    # predict harvest date using predicted anthesis age
    first_ripe_day = 113
    last_ripe_day = 140
    predicted_day = round(float(np.squeeze(predictions)))
    day_b4_ripe = first_ripe_day - predicted_day
    day_to_overripe = last_ripe_day - predicted_day
    first_date = (datetime.now() + timedelta(days=day_b4_ripe) ).strftime('%Y-%m-%d')
    last_date = (datetime.now() + timedelta(days=day_to_overripe) ).strftime('%Y-%m-%d')
    print("Start_Harvest :", first_date)
    print("End_Harvest :", last_date)
    harvest = [first_date, last_date]
    return harvest



cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6 # Set threshold for this model
cfg.MODEL.WEIGHTS = 'C:/Users/User/Documents/EBTECH/EB_Detectron2/mask_rcnn_R_50_FPN_3x.pth' # Set path model .pth
cfg.MODEL.DEVICE = "cpu" # cpu or cuda
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
predictor = DefaultPredictor(cfg)

# im = cv2.imread("C:/Users/User/Desktop/testing/overripe_day_150_pic_0.png")
# im = cv2.imread("C:/Users/User/Desktop/testing/ripe_day_120_pic_0.png")
# im = cv2.imread("C:/Users/User/Desktop/testing/underripe_day_80_pic_0.png")
# im = cv2.imread("C:/Users/User/Desktop/testing/unripe_day_40_pic_0.png")
# im = cv2.imread(r"C:\Users\User\Desktop\testing\more1.PNG")
im = cv2.imread(args.image)
im2 = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
overlay = im.copy()
outputs = predictor(im)

saving_path = os.path.join("C:/Users/User/Desktop/","output_"+str(args.image).split("\\")[-1])

mask_array = outputs['instances'].pred_masks.to("cpu").numpy()
num_instances = mask_array.shape[0]
scores = outputs['instances'].scores.to("cpu").numpy()
labels = outputs['instances'].pred_classes .to("cpu").numpy()
bbox   = outputs['instances'].pred_boxes.to("cpu").tensor.numpy()
mask_array = np.moveaxis(mask_array, 0, -1)

mask_array_instance = []
point_polygon_instance =[]
show = [str(j)+" : "+classname[str(j)]for j in labels]
print(show)

# img_mask = np.zeros([h, w, 3], np.uint8)
for i in range(num_instances):
    name = '{}'.format(classname[str(labels[i])])
    color = list(np.random.random(size=3) * 256)
    img = np.zeros_like(im)
    mask_array_instance.append(mask_array[:, :, i:(i+1)])
    img = np.where(mask_array_instance[i] == True, 255, img)
    array_img = np.asarray(img)
    img_mask2 = cv2.cvtColor(array_img, cv2.COLOR_RGB2GRAY)
    (thresh, im_bw) = cv2.threshold(img_mask2, 127, 255, 0)
    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cropped = PolygonCroper(im2,contours)
    midpoint =Midpoint(contours)
    date = Harvest_Date_Estimate(cropped)
    cv2.drawContours(im, contours, -1, color, 30)
    # cv2.polylines(im, contours, True, color, 3)
    # cv2.fillPoly(overlay, contours, color,)
    harvest_range = 'HDE: {}/{}-{}/{}/{}'.format(str(date[0]).split('-')[-1],str(date[0]).split('-')[1],str(date[1]).split('-')[-1],str(date[1]).split('-')[1],str(date[1]).split('-')[0])
    cv2.rectangle(im, (int(midpoint[0])-500, int(midpoint[1])-100), (int(midpoint[0])+700, int(midpoint[1])+200), color, cv2.FILLED)
    cv2.putText(im, name, (int(midpoint[0])-460, int(midpoint[1])+38), cv2.FONT_HERSHEY_PLAIN, 8, (0,0,0), 5)
    cv2.putText(im, harvest_range, (int(midpoint[0])-480, int(midpoint[1])+155), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,0), 5)
    # image_new = cv2.addWeighted(overlay, 0.4, im, 1 - 0.4, 0)
    

# image_new = cv2.cvtColor(image_new, cv2.COLOR_BGR2RGB)
cv2.imwrite(saving_path, im)
cv2.imshow("image", im)
cv2.waitKey(0)
cv2.destroyAllWindows()