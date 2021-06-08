#!/usr/bin/env python3

from numpy.compat.py3k import contextlib_nullcontext
import rospy
import os
import rospy,cv_bridge
from sensor_msgs.msg import Image
import cv2
import numpy as np
from scipy.spatial import distance
from roomba.msg import DetectedObject


class Detector:

    def __init__(self):
        rospy.init_node('detector')

        self.no_image = True

        # set up ROS / OpenCV bridge
        self.bridge = cv_bridge.CvBridge()    
        rospy.Subscriber('camera/rgb/image_raw',Image, self.image_callback)
        self.object_pub = rospy.Publisher("/roomba/detector", DetectedObject, queue_size=10)


    #Get a mask to get rid of any colors other than red, blue, and green.
    def get_mask(self, hsv):

        #red mask
        lower_red = np.array([0,120,70])
        upper_red = np.array([10,255,255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170,120,70])
        upper_red = np.array([180,255,255])
        mask2 = cv2.inRange(hsv,lower_red,upper_red)

        
        #green mask
        lower_green = np.array([40,40,40])
        upper_green = np.array([70,255,255])

        mask3 = cv2.inRange(hsv, lower_green , upper_green)

        #blue mask
        lower_blue  = np.array([100, 150, 0])
        upper_blue = np.array([140, 255, 255])

        mask4 = cv2.inRange(hsv, lower_blue  , upper_blue)   

        #merge all masks
        return mask1 + mask2 + mask3 + mask4  
    
    #Sometimes the contour includes image borders
    #which shouldnt be the case. Check for this. 
    def invalid_contour(self, contour):
        x,y,w,h = cv2.boundingRect(contour)
        contour_size=w*h
        error_threshold = 1000
        img_size = self.image.shape[0] * self.image.shape[1]
        
        if  abs(contour_size-img_size)<=error_threshold:
            return True
        else:
            return False

    '''detect, print, and publish the shape that is closest to the 
    camera center.''' 
    def detect_shape(self):

        if self.no_image:
            return

        #save the camera feed for debugging
        cv2.imwrite('temp.jpg', self.image)
        img = self.image

        #mask out irrelevant colors
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = self.get_mask(hsv)
        masked = cv2.bitwise_and(img,img,mask = mask)
        # get the positions of all pixels that are black (i.e. [0, 0, 0])
        black_pixels = np.where(
            (masked[:, :, 0] == 0) & 
            (masked[:, :, 1] == 0) & 
            (masked[:, :, 2] == 0)
        )

        # set those pixels to white
        masked[black_pixels] = [255, 255, 255]

        
        cv2.imwrite('masked.jpg', masked)

        #find the contours in the masked image
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #get the image center and draw a white circle on it
        h,w  = gray.shape
        image_center = np.array([w/2, h/2])
        image_center = tuple(image_center.astype('int32'))
        cv2.circle(img, image_center, 3, (255, 255, 255), 2)

        #iterate over all shapes detected
        shapes = []
        for i, contour in enumerate(contours):        
            #the first shape is the whole image so skip
            if i == 0:
                continue

            #approximate the shape by polygon
            approx = cv2.approxPolyDP(
                contour, 0.0000001 * (cv2.arcLength(contour, True) ** 2), True)

            
            #print(len(approx))
            
            #if invalid skip
            if self.invalid_contour(contour):
                continue
        
            #draw the contour
            cv2.drawContours(img, [contour], 0, (0, 50, 255), 2)

            # finding the center point of shape
            M = cv2.moments(contour)
            if M['m00'] > 0.0:
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])
                contour_center = (x,y)
                cv2.circle(img, contour_center, 3, (100, 255, 0), 2)

                #find the distance of the center of the shape to the center of the image
                distance_to_center = distance.euclidean(image_center, contour_center)
                shapes.append({'contour': contour, 'center': contour_center, 
                                    'approx': approx, 'distance_to_center': distance_to_center})

        #if no shapes found then return 
        if len(shapes) == 0:
            return 

        #sort the shapes by their distance to the center
        sorted_shapes = sorted(shapes, key=lambda i: i['distance_to_center'])
        
        #get the closest shape to the center and the number of its vertices
        closest_shape = sorted_shapes[0]
        num_vertices = len(closest_shape['approx'])

        #publish what is detected
        print(num_vertices)
        if num_vertices < 155:
            print('dumbbell')
            detected_object = DetectedObject()
            detected_object.object = 'dumbbell'
            self.object_pub.publish(detected_object)
        else:
            print('kettlebell')
            detected_object = DetectedObject()
            detected_object.object = 'kettlebell'
            self.object_pub.publish(detected_object)

        # find contour of closest building to center and draw it (blue)
        center_shape_contour = closest_shape['contour']
        cv2.drawContours(img, [center_shape_contour], 0, (255, 0, 0), 2)

        #write the shape outlined image to a file
        cv2.imwrite('processed.jpg', img)
        


    #set the camera feed
    def image_callback(self, msg):
        self.no_image = False
        self.image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
        
        

    def run(self):
        r = rospy.Rate(5)

        while not rospy.is_shutdown():
            self.detect_shape()
            r.sleep()       

if __name__ == '__main__':
    detector = Detector()

    detector.run()