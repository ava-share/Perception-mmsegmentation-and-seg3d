#!/usr/bin/env python
# coding=utf-8
'''
Author: Jianheng Liu
Date: 2021-10-23 23:05:43
LastEditors: Jianheng Liu
LastEditTime: 2021-11-02 12:31:27
Description: MMSegmentor
'''

# Check Pytorch installation
import threading
import sys
sys.path.append("/home/avalocal/catkin_ws/src/mmsegmentation_ros/mmsegmentation")
import mmseg

from mmseg.models import build_segmentor
from sensor_msgs.msg import Image 
from std_msgs.msg import String, Header,Float32, MultiArrayDimension, UInt8
import rospy
import numpy as np
import cv2
import ros_numpy

import os
from PIL import Image as IM
from geometry_msgs.msg import Point32

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

#######
from mmseg.models.segmentors import EncoderDecoder #inference as INF


#######

from mmsegmentation_ros.msg import DetectedRoadArea

from mmcv.ops import get_compiling_cuda_version, get_compiler_version
import mmseg
import mmcv
from logging import debug
from mmcv import image
import torch
import torchvision

class Segmentor:

    def __init__(self):
        #torch.set_num_threads(8)
        # # Choose to use a config and initialize the detecto
        # Config file
        self.config_path = rospy.get_param('~config_path')
        # Checkpoint file
        self.checkpoint_path = rospy.get_param('~checkpoint_path')
        # Device used for inference
        self.device = rospy.get_param('~device', 'cuda:0')
        # Color palette used for segmentation map
        self.palette = rospy.get_param('~palette', 'cityscapes')
        # Opacity of painted segmentation map. In (0, 1] range. 
        self.opacity = rospy.get_param('~opacity', 0.5)

        self._publish_rate = rospy.get_param('~publish_rate', 50)
        self._is_service = rospy.get_param('~is_service', False)
        self._visualization = rospy.get_param('~visualization', True)

        # build the model from a config file and a checkpoint file
        self.model = init_segmentor(
            self.config_path, self.checkpoint_path, device=self.device)

        self._last_msg = None
        self._msg_lock = threading.Lock()

        self.image_pub = rospy.Publisher("~debug_image", Image, queue_size=1)
        self.roadarea_pub = rospy.Publisher("~roadarea", DetectedRoadArea, queue_size=1)

        image_sub = rospy.Subscriber(
                "~image_topic", Image, self._image_callback, queue_size=1, buff_size=2**24)

    def run(self):

        while not rospy.is_shutdown():
            self.time1 = rospy.Time.now()
            if self._msg_lock.acquire(False):
                msg = self._last_msg
                self._last_msg = None
                self._msg_lock.release()
            else:

                continue

            if msg is not None:
                
                image_np = ros_numpy.numpify(msg)

                image_np = cv2.resize(image_np, (256,193)) #
                area_msg = DetectedRoadArea()

                with torch.no_grad(): #not sure if this is needed
                	result, confidence = inference_segmentor(self.model, image_np)
                #print(result) #output format is an array of 0 and 1s (not road and road) the size of the image (pixels) 
                #the underlying code has been slightly modified to return seg_logit in encoder_decoder.py
                sum=0
                out_area = []; shape_mat = 0 
             
                confidence = (np.array(confidence[:,1].cpu()))[0] #convert tensor to array
                #From here to end of area_msg has to be improved, this takes 0.2 seconds
                for_conf_id = np.where(result[0] == 1) 
                #print(for_conf_id)
                #to_average_conf = confidence[for_conf_id] #find confidence only for detected road
                to_average_conf = confidence[result[0] == 1]
                average_conf = np.average(to_average_conf) #average confidence for positive detection
                #print('average confidence: ', average_conf)
                
                #out_area = np.resize(result[0],(2048,1544))
                shape_mat=result[0].shape
                out_area = np.reshape(result[0], (-1,), order='A')#np.reshape(np.asarray(result[0]), (-1,), order='A')#np.asarray(out_area.reshape(-1,order='A')) #np.asarray(result[0]).reshape(-1,order='A')#this is fairly slow

                area_msg.RoadArea.layout.dim = []
                area_msg.RoadArea.data = (out_area)
                area_msg.RoadArea.layout.dim.append(MultiArrayDimension())
                area_msg.RoadArea.layout.dim.append(MultiArrayDimension())
                area_msg.RoadArea.layout.dim[0].label = "width"   
                area_msg.RoadArea.layout.dim[1].label = "height"                 
                area_msg.RoadArea.layout.dim[0].size = shape_mat[0]
                area_msg.RoadArea.layout.dim[1].size = shape_mat[1]

                #area_msg.confidence = average_conf
                # # Visualize results
                if self._visualization:
			
                    debug_image = self.model.show_result(
                        image_np, result, palette=get_palette(self.palette), show=False, opacity=self.opacity)

                    area_msg.header = msg.header
                    image_out = ros_numpy.msgify(Image, debug_image, 'bgr8')
                    image_out.header.stamp = msg.header.stamp
                    #print(msg)
                    self.image_pub.publish(image_out)
                    self.roadarea_pub.publish(area_msg)
                    self.time2 = rospy.Time.now()
                    print(self.time2.to_sec() - self.time1.to_sec())
            #rate.sleep()

    def _image_callback(self, msg):
        #rospy.logdebug("Get an image")
        
        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._msg_lock.release()

    def service_handler(self, request):
        return self._image_callback(request.image)


def main():
    rospy.init_node('mmdetector')

    obj = Segmentor()
    obj.run()


if __name__ == '__main__':
    main()
