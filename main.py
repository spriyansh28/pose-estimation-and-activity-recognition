import numpy as np
import cv2
import time
import logging
import argparse

import display
import data_preprocessing 

from tf_pose.networks import get_graph_path, model_wh
from tf_pose.estimator import TfPoseEstimator
from tf_pose import common

from keras.models import load_model

def live_image_source():
    key_word = "--source"
    choices = ["cam", "storage"]

    parser = argparse.ArgumentParser()
    parser.add_argument(key_word, required=False, default='cam')
    inp = parser.parse_args().source
    if inp == "cam":
        return True
    elif inp == "storage":
        return False
    else:
        print("\nWrong command line input !\n")    


logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class image_loader(object):
    def __init__(self):
        if live_image_source():
            self.cam = cv2.VideoCapture(0)
            self.num_images = 10000

        else:
            self.cam = cv2.VideoCapture('test_video_4.mp4')
            self.num_images = 10000

    def load_next_image(self):
        ret_val, img = self.cam.read()
        img =cv2.flip(img, 1)
        action_type = "unknown"
        return img, action_type



class SkeletonDetector(object):

    def __init__(self, model="cmu"):
        models = set({"mobilenet_thin", "cmu"})
        self.model = model if model in models else "mobilenet_thin"
        self.resize_out_ratio = 4.0

        w, h = model_wh("432x368")
        if w == 0 or h == 0:
            e = TfPoseEstimator(get_graph_path(self.model),
                                target_size=(432, 368))
        else:
            e = TfPoseEstimator(get_graph_path(self.model), target_size=(w, h))

        self.w = w 
        self.h = h
        self.e = e
        self.fps_time = time.time()

    def detect(self, image):
        t = time.time()

        humans = self.e.inference(image, resize_to_default=(self.w > 0 and self.h > 0),
                                  upsample_size=self.resize_out_ratio)

        print("humans:", humans)
        elapsed = time.time() - t
        logger.info('inference image in %.4f seconds.' % (elapsed))

        return humans
    
    def draw(self, img_disp, humans):
        img_disp = TfPoseEstimator.draw_humans(img_disp, humans, imgcopy=False)

        logger.debug('show+')
        cv2.putText(img_disp,
                    "FPS: %f" % (1.0 / (time.time() - self.fps_time)),
                    (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        self.fps_time = time.time()

    @staticmethod
    def humans_to_skelsInfo(humans, action_type="None"):
        skelsInfo = []
        NaN = 0
        for human in humans:
            skeleton = [NaN]*(55)
            skeleton[0] = action_type
            for i, body_part in human.body_parts.items(): 
                idx = body_part.part_idx
                skeleton[1+2*idx]=body_part.x
                skeleton[1+2*idx+1]=body_part.y
            skelsInfo.append(skeleton)
        return skelsInfo
    
    @staticmethod
    def get_ith_skeleton(skelsInfo, ith_skeleton=0):
        return np.array(skelsInfo[ith_skeleton][1:35])



class ActionClassifier(object):
    
    def __init__(self, model_path):

        self.dnn_model = load_model(model_path)
        self.action_dict = ["kick", "punch", "squat", "stand", "wave"]

    def predict(self, skeleton):

        tmp = data_preprocessing.pose_normalization(skeleton)
        skeleton_input = np.array(tmp).reshape(-1, len(tmp))
            
        predicted_idx = np.argmax(self.dnn_model.predict(skeleton_input))
        prediced_label = self.action_dict[predicted_idx]

        return prediced_label

int2str = lambda num, blank: ("{:0"+str(blank)+"d}").format(num)


if __name__ == "__main__":
 
    my_detector = SkeletonDetector()

    images_loader = image_loader()

    classifier = ActionClassifier("action_poses.h5")

    ith_img = 1
    while ith_img <= images_loader.num_images:
        img, action_type = images_loader.load_next_image()
        image_disp = img.copy()

        print("\n\n========================================")
        print("\nProcessing {}/{}th image\n".format(ith_img, images_loader.num_images))

        humans = my_detector.detect(img)
        skelsInfo = SkeletonDetector.humans_to_skelsInfo(humans, action_type)
        for ith_skel in range(0, len(skelsInfo)):
            skeleton = SkeletonDetector.get_ith_skeleton(skelsInfo, ith_skel)

            prediced_label = classifier.predict(skeleton)
            print( prediced_label)

            if 1:
                if ith_skel == 0:
                    my_detector.draw(image_disp, humans)
                
                display.drawActionResult(image_disp, skeleton, prediced_label)

        if 1: 
            if live_image_source():
                cv2.imshow("Action", 
                cv2.resize(image_disp,(0,0),fx=2,fy=2))
                q = cv2.waitKey(1)
                if q!=-1 and chr(q) == 'q':
                    break

            else:
                cv2.imshow("Action", 
                cv2.resize(image_disp,(0,0),fx=1,fy=1))
                q = cv2.waitKey(1)
                if q!=-1 and chr(q) == 'q':
                    break
            

        print("\n")
        ith_img += 1

