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


#print(data['camera_matrix']['data'])
#################################################################################################

#D=np.array([D: [-0.245253, 0.149647, 0.003117, 0.000761, 0.0]]

#The following should be replaced by subscribing to the /camera_info topic as well as the /tf topics



P3_rect=np.array([[7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, -3.395242000000e+02],
                  [0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 2.199936000000e+00],
                   [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 2.729905000000e-03]])


R0_rect=np.array([[9.999239000000e-01, 9.837760000000e-03, -7.445048000000e-03, 0.00000],
                 [-9.869795000000e-03, 9.999421000000e-01, -4.278459000000e-03, 0.00000],
                 [7.402527000000e-03, 4.351614000000e-03, 9.999631000000e-01, 0.00000],
                 [0.000000,           0.000000,               0.000000,   1.00000]])


R3_rect=np.array([[9.998321e-01, -7.193136e-03, 1.685599e-02, 0],
[7.232804e-03, 9.999712e-01, -2.293585e-03, 0],
[-1.683901e-02, 2.415116e-03, 9.998553e-01, 0],
[0, 0, 0, 1]])#'''#from file this is R_rect_03

#mtx=np.array([[3407.91772, 0.0000000000, 1066.72048],
 #               [0.00000000, 3451.94116, 825.36976],
  #              [0.0000000000, 0.0000000000, 1.0000000]])

#R0_rect=np.array([[9.999239000000e-01, 9.837760000000e-03, -7.445048000000e-03, 0.00000],
 #                [-9.869795000000e-03, 9.999421000000e-01, -4.278459000000e-03, 0.00000],
  #                [7.402527000000e-03, 4.351614000000e-03, 9.999631000000e-01, 0.00000],
  #                [0.000000,           0.000000,               0.000000,   1.00000]])



T1=np.array([[7.533745000000e-03, -9.999714000000e-01, -6.166020000000e-04, -4.069766000000e-03],
                    [1.480249000000e-02, 7.280733000000e-04, -9.998902000000e-01, -7.631618000000e-02],
                     [9.998621000000e-01, 7.523790000000e-03, 1.480755000000e-02, -2.717806000000e-01],
                     [0.000000,           0.000000,               0.000000,   1.00000]])




#D= np.array([-0.245253, 0.149647, 0.003117, 0.000761, 0.0])
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

#print(inverse_rigid_transformation(lidar_extrinsic))
##################################################################################################

#T_vel_cam=inverse_rigid_transformation(T1)
#print(T_vel_cam)
T_vel_cam=T1

##############################################################################################
lim_x=[0, 50]
lim_y=[-25,25]
lim_z=[-5,5]
height= 375
width= 1242

pixel_lim=3
##############################################################################################

class realCoor():
    
    def __init__(self):

        self.p_pub = rospy.Publisher("/used_pcl", PointCloud2, queue_size=1)
        self.seg_pub = rospy.Publisher("/segmented_pcl", PointCloud2, queue_size=1)

        self.fields = [pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),pc2.PointField(name='y', offset=4,datatype=pc2.PointField.FLOAT32, count=1),pc2.PointField(name='z', offset=8,datatype=pc2.PointField.FLOAT32, count=1),pc2.PointField(name='intensity', offset=12,datatype=pc2.PointField.FLOAT32, count=1)]
        self.vis=True

        self.pcdSub=message_filters.Subscriber("/kitti/velo/pointcloud", PointCloud2)#/kitti/velo/pointcloud
        self.used_pointcloud = PointCloud2() #rospy.subscriber
        self.segSub=message_filters.Subscriber("/mmsegmentor/roadarea", DetectedRoadArea)
        self.header = std_msgs.msg.Header()
        self.header.frame_id = 'velo_link'
        #self.img_pub = rospy.Publisher("~debug_image", Image, queue_size=1)
        ds=message_filters.ApproximateTimeSynchronizer(([self.pcdSub, self.segSub]),10, 0.1)
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

        if msgPoint.RoadArea!=[]:
           
            #print("update mmseg")
            arr=np.array(msgPoint.RoadArea.data)
            #print(np.shape(arr))
            msgPoint = np.array(msgPoint.RoadArea.data).reshape(height, width) #375,1242(this would output y, x?!)
            #print(type(msgPoint))

            '''data = np.array(msgPoint*255, dtype=np.uint8)
            img = Imm.fromarray(data, 'L')
            img.show()'''

            road_pixels = np.array(np.where(msgPoint == 1))
            '''a=road_pixels.T
            sort_idx=np.lexsort((a[:,1],a[:,0]))
            sorted_roadPixel_indecies=a[sort_idx].T'''
            #print(sorted_roadPixel_indecies)
            #print(road_pixels, "shape of road pixels")   #0-->u, 1-->v || 0-->y, 1--->x
            
            pc = ros_numpy.numpify(msgLidar)
            points=np.zeros((pc.shape[0],4))
            points[:,0]=pc['x']
            points[:,1]=pc['y']
            points[:,2]=pc['z']
            points[:,3]=1

            pc_arr=self.crop_pointcloud(points) #to reduce computational expense
            pc_arr_pick=np.transpose(pc_arr)       
            #print(pc_arr_pick.shape)
            m1=np.matmul(T_vel_cam,pc_arr_pick)#4*N
            m2=np.matmul(R0_rect,m1)
            uv1= np.matmul(P3_rect,m2) #4*N        
            #check = uv1
            uv1[0,:]=  np.divide(uv1[0,:],uv1[2,:])
            uv1[1,:]=  np.divide(uv1[1,:],uv1[2,:])
            #v=uv1[0,:]
            #u=uv1[1,:]

            #print(uv1, "uv1")  
            #uv_points=(uv1[0:2, :]).T  #pointcloud_pixels
            #print(uv_points.shape, "uv1")
            #tmp,=np.where((uv_points[:,0]>0 )& (uv_points[:,0]<width )& (uv_points[:,1]>0) &(uv_points[:,1]<height))
            #print(tmp,"tmp")
            #uv_points=uv_points[tmp]
            #sort_idx2=np.lexsort((uv_points[:,1],uv_points[:,0]))
            #sorted_pointcloud=np.transpose(uv_points[sort_idx2])
            #print(sorted_pointcloud, "this")

            line_3d=[]
            v=uv1[0,:]
            u=uv1[1,:]

            #print(road_pixels.shape, " road pixels - for check 1")
            #print(uv1.shape,"uv1 - for check 2")

            points_list=list(zip(road_pixels[0],road_pixels[1]))
            #print(points_list)
            p=MultiPoint(points_list).convex_hull

            a,b= p.exterior.coords.xy
            points=list(list(zip(a,b)))
            poly=Polygon(points)
            #poly.contains(Point(180,9))
            res3=[]
            for x in uv1.T:
                res3.append(bool(poly.contains(Point(x[1], x[0]))))
            res3=np.array(res3)
            #approach 1 (non-relevent points are generated as wll)
            #intersections1 = self.isin_tolerance(sorted_pointcloud[0,:],sorted_roadPixel_indecies[1,:], 2)
            intersections1=  self.isin_tolerance(v,road_pixels[1,:],3) #size of v
            intersections2=  self.isin_tolerance(u,road_pixels[0,:],3)
            #isin_tolerance pulled from here: https://stackoverflow.com/questions/51744613/numpy-setdiff1d-with-tolerance-comparing-a-numpy-array-to-another-and-saving-o/51747164#51747164

            #print(intersections1.shape,"intersection 1 shape")
            #print(intersections2.shape,"intersection 2 shape")
            #print(res3.shape, "res3 shape")

            idx = np.where((intersections1 == True) & (intersections2 == True) & (res3==True))
            idx = np.array(idx)
            #print(idx.shape,"shape of idx array")
            
            
            if idx.shape[0] > 0 :
            #if line_3d!=[]:
                #line_3d=np.vstack((u[idx[:]],v[idx[:]],uv1[2,:][idx[:]],np.ones((1,idx.shape[1]))))#, np.ones(idx.shape)]]
                
                line_3d=np.vstack((pc_arr_pick[0][idx[:]],pc_arr_pick[1][idx[:]],pc_arr_pick[2][idx[:]],np.ones((1,idx.shape[1]))))#, np.ones(idx.shape)]]
            else:
                rospy.logerr('no matching points between image pixel and lidar points!')
            self.vis=True
            if self.vis == True and line_3d!=[]:
                #line_3d=np.array(np.transpose(line_3d)) 
                #print(line_3d.shape, "line_3d shape")
                line_3d=np.array((line_3d).T)    
                #print(line_3d.shape)      
                _, idx=np.unique(line_3d[:,0:2], axis=0, return_index=True)
                line_3d_unique=line_3d[idx]
                self.create_cloud(line_3d_unique,2)
                #self.create_cloud(pc_arr,1) #uncomment this line to publish the cropped
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
    def isin_tolerance2D(self,A, Bs,Bs1, tol):
            A = np.asarray(A)
            Bs = np.asarray(Bs)
            Bs1 = np.asarray(Bs1)

            #Bs = np.sort(Bd) # skip if already sorted
            idx = np.searchsorted(Bs, A)
            
            linvalid_mask = idx==len(Bs)
            idx[linvalid_mask] = len(Bs)-1
            print(idx, "idx")
            lval = Bs[idx] - A
            Bs1[idx]*=-1
            #lval[linvalid_mask] *=-1

            rinvalid_mask = idx==0
            idx1 = idx-1
            idx1[rinvalid_mask] = 0
            print(idx1, "idx1")
            rval = A - Bs[idx1]
            Bs1[idx1]*=-1
            #rval[rinvalid_mask] *=-1
            print(lval, "lval")
            print(rval, "rval")
            res= ((np.minimum(lval, rval) <= tol) & (np.minimum(lval, rval) >=0))
            Bs1*=-1
            Bs1[np.where(Bs1<0)]=0

            print(res, "res")
            print(Bs1, "cp")
            return res, Bs1


    def crop_pointcloud(self, pointcloud):
        # remove points outside of detection cube defined in 'configs.lim_*'
        mask = np.where((pointcloud[:, 0] >= lim_x[0]) & (pointcloud[:, 0] <= lim_x[1]) & (pointcloud[:, 1] >=lim_y[0]) & (pointcloud[:, 1] <= lim_y[1]) & (pointcloud[:, 2] >= lim_z[0]) & (pointcloud[:, 2] <= lim_z[1]))
        pointcloud = pointcloud[mask]
        return pointcloud
        
if __name__=='__main__':
    rospy.init_node("segmentationTO3d")
    realCoor()















  
