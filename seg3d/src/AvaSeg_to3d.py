#!/usr/bin/env python3

'''
Author: Pardis Taghavi, Jonas Lossner
Texas A&M University - Fall 2022
'''

from pyexpat.errors import XML_ERROR_ENTITY_DECLARED_IN_PE
from re import L
from smtplib import LMTP
import sys
from tkinter import X
from xml.etree.ElementTree import XML
sys.path.insert(1, "/home/avalocal/catkin_ws/src/laneatt_ros")
sys.path.insert(1, "/home/avalocal/Desktop/rosSeg/src/mmsegmentation_ros")
from PIL import Image as Imm
import std_msgs.msg
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
import message_filters
from laneatt_ros.msg import DetectedLane
from laneatt_ros.msg import DetectedRoadArea
import sensor_msgs.point_cloud2 as pc2
import ros_numpy
from scipy import interpolate
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy import spatial
from matplotlib import pyplot as plt


from shapely.geometry import Polygon
from shapely.geometry import MultiPoint
from shapely.geometry import Point
import matplotlib.pyplot as plt


#########DATA

#tf transfer matrix between lidar-camera

#The following should be replaced by subscribing to the /camera_info topic as well as the /tf topics

#ext_cam: estrinsic camera matrix
#Tmat : lidar/camera data

rect=np.array([[891.8531968155474/2,    0.000000     , 288.1985664394375/2 ,    0.000000],
                  [0.000000     , 890.714241372335/2,  193.818451578666/2,    0.000000],
                   [0.00000     , 0.0000000       , 1.00000000       , 0.000000]])
#rect=np.array([[3567.41278726219,    0.000000     , 1152.79426575775 ,    0.000000],
#                  [0.000000     , 3562.85696548934,  775.273806314664,    0.000000],
#                   [0.00000     , 0.0000000       , 1.00000000       , 0.000000]]) #original full size cal (2048x1544)

T1=np.array([[0.021925470380414,-0.045549999800206,0.998721418247751, 1.207224761165979],
             [-0.998142761906601,-0.057787694764603,0.019277167511240, 0.448298426247539],
            [0.056835733496287,-0.997289215750373,-0.046732424995337, -0.753386884363737],
                     [0.000000,           0.000000,               0.000000,   1.00000]])


##################################################################################################
def inverse_rigid_transformation(arr):
    irt=np.zeros_like(arr)
    Rt=np.transpose(arr[:3,:3])
    tt=-np.matmul(Rt,arr[:3,3])
    irt[:3,:3]=Rt
    irt[0,3]=tt[0]
    irt[1,3]=tt[1]
    irt[2,3]=tt[2]
    irt[3,3]=1
    return irt

##################################################################################################

T_vel_cam=inverse_rigid_transformation(T1)

##############################################################################################
lim_x=[2, 50]
lim_y=[-15,15]
lim_z=[-5,5]
height= 772#1024  
width= 1024#772

pixel_lim=5
##############################################################################################

class realCoor():
    
    def __init__(self):

        #self.p_pub = rospy.Publisher("/used_pcl", PointCloud2, queue_size=1) #uncomment if publishing cropped pointcloud
        self.seg_pub = rospy.Publisher("/segmented_pcl", PointCloud2, queue_size=1)

        self.fields = [pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),pc2.PointField(name='y', offset=4,datatype=pc2.PointField.FLOAT32, count=1),pc2.PointField(name='z', offset=8,datatype=pc2.PointField.FLOAT32, count=1),pc2.PointField(name='intensity', offset=12,datatype=pc2.PointField.FLOAT32, count=1)]
        self.vis=True

        self.pcdSub=message_filters.Subscriber("/lidar_tc/velodyne_points", PointCloud2)  #/kitti/velo/pointcloud
        self.used_pointcloud = PointCloud2() #rospy.subscriber
        self.segSub=message_filters.Subscriber("/mmsegmentor/roadarea", DetectedRoadArea)
        self.header = std_msgs.msg.Header()
        self.header.frame_id = 'lidar_tc'

        ds=message_filters.ApproximateTimeSynchronizer(([self.pcdSub, self.segSub]),2, 0.3)
        ds.registerCallback(self.segmentation_callback)
        print("realCoor initialized")
        rospy.spin()
        
    def create_cloud(self,line_3d, which):
    
        self.header.stamp = rospy.Time.now()
        
        if which == 0:
            self.lane_pointcloud = pc2.create_cloud(self.header, self.fields, line_3d)
            self.pcl_pub.publish(self.lane_pointcloud)
        elif which ==1:
            self.used_pointcloud = pc2.create_cloud(self.header, self.fields, line_3d)
            self.p_pub.publish(self.used_pointcloud)
        elif which ==2:
            self.used_pointcloud = pc2.create_cloud(self.header, self.fields, line_3d)
            self.seg_pub.publish(self.used_pointcloud)

    def segmentation_callback(self,msgLidar, msgPoint):
        
        #start_time_seg = rospy.Time.now().to_sec()
        #print('matching cloud/image')
        if msgPoint.RoadArea!=[]:
           
            arr=np.array(msgPoint.RoadArea.data)

            msgPoint = arr.reshape(msgPoint.RoadArea.layout.dim[0].size, msgPoint.RoadArea.layout.dim[1].size) #375,1242(this would output y, x?!)
            road_pixels = np.array(np.where(msgPoint == 1))

            pc = ros_numpy.numpify(msgLidar)
            points=np.zeros((pc.shape[0],4))
            points[:,0]=pc['x']
            points[:,1]=pc['y']
            points[:,2]=pc['z']
            points[:,3]=1

            pc_arr=self.crop_pointcloud(points) #to reduce computational expense
            pc_arr_pick=np.transpose(pc_arr)       

            m1=np.matmul(T_vel_cam,pc_arr_pick)#4*N

            uv1= np.matmul(rect,m1) #4*N         

            uv1[0,:]=  np.divide(uv1[0,:],uv1[2,:])
            uv1[1,:]=  np.divide(uv1[1,:],uv1[2,:])
         

            line_3d=[]
            v=uv1[0,:]
            u=uv1[1,:]


            points_list=list(zip(road_pixels[0],road_pixels[1]))

            p=MultiPoint(points_list).convex_hull

            a,b= p.exterior.coords.xy
            points=list(list(zip(a,b)))
            poly=Polygon(points)

            res3=[]
            for x in uv1.T:
                res3.append(bool(poly.contains(Point(x[1], x[0]))))
            res3=np.array(res3)
           
            intersections1=  self.isin_tolerance(v,road_pixels[1,:],3) #size of v
            intersections2=  self.isin_tolerance(u,road_pixels[0,:],3)
            #isin_tolerance pulled from here: https://stackoverflow.com/questions/51744613/numpy-setdiff1d-with-tolerance-comparing-a-numpy-array-to-another-and-saving-o/51747164#51747164



            idx = np.where((intersections1 == True) & (intersections2 == True) & (res3==True))
            idx = np.array(idx)

            
            
            if idx.shape[0] > 0 :
                
                line_3d=np.vstack((pc_arr_pick[0][idx[:]],pc_arr_pick[1][idx[:]],pc_arr_pick[2][idx[:]],np.ones((1,idx.shape[1]))))
            else:
                rospy.logerr('no matching points between image pixel and lidar points!')
            self.vis=True
            if self.vis == True and line_3d!=[]:
                
                line_3d=np.array((line_3d).T)    
                _, idx=np.unique(line_3d[:,0:2], axis=0, return_index=True)
                line_3d_unique=line_3d[idx]
                self.create_cloud(line_3d_unique,2)
                #self.create_cloud(pc_arr,1) #uncomment this line to publish the cropped pointcloud
            else:
                rospy.logwarn("Visualization is disabled or no points fused - no pointcloud published")

            #time_end_seg = rospy.Time.now().to_sec()
            #print("Segmentation time :: ", (start_time_seg - time_end_seg))

    def isin_tolerance(self,A, B, tol):
        A = np.asarray(A)
        B = np.asarray(B)

        Bs = np.sort(B) # skip if already sorted
        idx = np.searchsorted(Bs, A)

        linvalid_mask = idx==len(B)
        idx[linvalid_mask] = len(B)-1
        lval = Bs[idx] - A
        #lval[linvalid_mask] *=-1

        rinvalid_mask = idx==0
        idx1 = idx-1
        idx1[rinvalid_mask] = 0
        rval = A - Bs[idx1]
        #rval[rinvalid_mask] *=-1
        res= ((np.minimum(lval, rval) <= tol) & (np.minimum(lval, rval) >=0))
        return res
    #Bd is an array with 2*N elements
    # Bd is lex sorted 1) 0 ... 2) 1 

    def crop_pointcloud(self, pointcloud):
        # remove points outside of detection cube defined in 'configs.lim_*'
        mask = np.where((pointcloud[:, 0] >= lim_x[0]) & (pointcloud[:, 0] <= lim_x[1]) & (pointcloud[:, 1] >=lim_y[0]) & (pointcloud[:, 1] <= lim_y[1]) & (pointcloud[:, 2] >= lim_z[0]) & (pointcloud[:, 2] <= lim_z[1]))
        pointcloud = pointcloud[mask]
        return pointcloud
        
if __name__=='__main__':
    rospy.init_node("segmentationTO3d")
    realCoor()















  
