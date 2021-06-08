#!/usr/bin/env python3

import rospy
import numpy as np
import cv2, cv_bridge
import math
import time
import os

import moveit_commander
import cv2, cv_bridge
from roomba.msg import DetectedObject
from geometry_msgs.msg import Twist, Vector3, Pose
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from distances.compute_distances import floyd_warshall
from action_states.generate_action_states import read_objects_and_bins

# Path of directory on where this file is located
path_prefix = os.path.dirname(__file__)

def get_dist(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def get_yaw_from_pose(p):
    """ A helper function that takes in a Pose object (geometry_msgs) 
    and returns yaw """

    yaw = (euler_from_quaternion([
            p.orientation.x,
            p.orientation.y,
            p.orientation.z,
            p.orientation.w])
            [2])

    return yaw

def get_target_angle(p1, p2):
    """ Gets the target angle for the robot to turn to based on robot position (p1)
    and node position (p2) """

    target_angle = 0

    # Get all the x and y coordinates
    x1 = p1.x
    y1 = p1.y
    x2 = p2[0]
    y2 = p2[1]

    # Calculate the angle between p1 and p2 as an absolute value
    raw_angle = math.atan(abs(x2 - x1) / abs(y2 - y1))
    print("raw", raw_angle)

    # Visualize this by positioning the camera angle with x axis on the horizontal
    #   (left -> right increase) and y axis on the vertical (down -> up increase)
    # Note: the yaw of the robot is 0 when facing positive x axis, left turn +yaw,
    #   right turn -yaw
    if x2 < x1 and y2 > y1:
        # print("top left")
        target_angle = math.radians(90) + raw_angle
    elif x2 > x1 and y2 > y1:
        #print("top right")
        target_angle = math.radians(90) - raw_angle
    elif x2 > x1 and y2 < y1:
        #print("bottom right")
        target_angle = -1 * (math.radians(90) - raw_angle)
    elif x2 < x1 and y2 < y1:
        #print("bottom left")
        target_angle = -1 * (math.radians(90) + raw_angle)

    return target_angle

class RobotMovement(object):
    def __init__(self):
        self.initialized = False
        # init node
        rospy.init_node('turtlebot3_movement')

        # This will be set to True once everything is set up
        self.initialized = False

        # init computer vision
        rospy.Subscriber("/camera/rgb/image_raw", Image, self.update_image)
        # init LIDAR
        rospy.Subscriber("scan", LaserScan, self.update_distance)
        # init odom
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        # init detector
        rospy.Subscriber("/roomba/detector", DetectedObject, self.object_detector_callback)

        # init arm
        self.move_arm  = moveit_commander.MoveGroupCommander("arm")
        self.move_grip = moveit_commander.MoveGroupCommander("gripper")
        # init movement
        self.twist = Twist()
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=100)

        # Fetch pre-built action matrix. This is a 2d numpy array where row indexes
        # correspond to the starting state and column indexes are the next states.
        #
        # A value of -1 indicates that it is not possible to get to the next state
        # from the starting state. Values 0-5 correspond to what action is needed
        # to go to the next state.
        #
        # e.g. self.action_matrix[0][1] = 5
        self.action_matrix = np.loadtxt(path_prefix + "/action_states/" + "action_matrix.csv", delimiter = ',')

        # Fetch actions. These are the only 6 possible actions the system can take.
        # self.actions is an array of dictionaries where the row index corresponds
        # to the action number, and the value has the following form:
        # { object: "dumbbell", color: "red", node: 1}
        self.actions = np.genfromtxt(path_prefix + "/action_states/" + "objects.csv", dtype = 'str', delimiter = ',')
        self.actions = list(map(
            lambda x: {"object": str(x[0]), "color": str(x[1]), "node": int(x[2])},
            self.actions
        ))

        # Fetch bins. There are 2 bins, each with a different color and at a different node
        objects, self.bins = read_objects_and_bins(path_prefix + "/action_states/")

        # set up graph-related variables
        self.origin_state = 256
        self.origin_node = 4

        self.num_nonobstacles = 0
        for obj in objects:
            if obj[0] != "obstacle":
                self.num_nonobstacles += 1

        # Set up a list to store the trained Q-matrix
        self.q_matrix = []
        self.load_q_matrix()

        # Set up a list to store the node locations
        self.locations = []
        self.load_locations()

        # Set up pose and orientation variables
        self.curr_pose = Pose()
        self.oriented = False

        # Set up kp for proportional control
        self.kp = 0.2

        # Get the array of shortest paths from node to node
        weight_mat = np.genfromtxt(path_prefix + "/distances/" + "map1_matrix.csv", delimiter=',')
        n = weight_mat.shape[0]
        for i in range(weight_mat.shape[0]):
            for j in range(weight_mat.shape[1]):
                if weight_mat[i,j] == 0:
                    weight_mat[i,j] = 1e9
        distance, self.shortest_paths = floyd_warshall(n, weight_mat)

        # current object and color of object
        self.curr_obj = None
        self.curr_obj_col = None

        # status of robot
        self.holding_object = False
        self.drop_off = False
        self.going_to_pick_up = False
        self.score = 0 # debug counter

        # set up action sequence
        self.action_sequence = []
        self.get_action_sequence()

        # set up the node sequence
        self.node_sequence = []
        self.get_node_sequence()

        # set up ROS / OpenCV bridge
        #ros uses its own image type, cv bridge nicer to work with
        self.bridge = cv_bridge.CvBridge()
        self.image_data = None

        # We are good to go!
        self.initialized = True

    # INIT FUNCS
    def load_locations(self):
        """ Load locations of each node """
        
        # Store the file into self.locations
        self.locations = np.loadtxt(path_prefix + "/locations.csv", delimiter = ',')
    def load_q_matrix(self):
        """ Load the trained Q-matrix csv file """

        # Store the file into self.q_matrix
        self.q_matrix = np.loadtxt(path_prefix + "/q_matrix.csv", delimiter = ',')
    def get_action_sequence(self):
        """ Get the sequence of actions for the robot to move the dumbbells
        to the correct blocks based on the trained Q-matrix """
        
        # Start at the origin
        curr_state = self.origin_state

        # Loop through a specific amount of times to get the action sequence
        for i in range(self.num_nonobstacles):

            # Get row in matrix and select the best action to take with the
            #   maximum Q-value
            q_matrix_row = self.q_matrix[curr_state]
            selected_action = np.where(q_matrix_row == max(q_matrix_row))[0][0]

            # Store the object, color, and node for the action as a tuple
            obj = self.actions[selected_action]["object"]
            clr = self.actions[selected_action]["color"]
            node = self.actions[selected_action]["node"]
            self.action_sequence.append((obj, clr, node))

            # Update the current state
            curr_state = np.where(self.action_matrix[curr_state] == selected_action)[0][0]
                
        print(self.action_sequence)
    def get_node_sequence(self):
        """ Get the sequence of nodes for each action from the action sequence
        with the help of shortest_paths.txt, e.g. the first index is the sequences
        [[4, 1], [1, 4, 5, 6]], meaning the robot has to go from node 4 -> 1 to
        pick up the red dumbbell, and go from node 1 -> 4 -> 5 -> 6 to place the
        red dumbbell at the red bin  """

        # First we start at the origin node
        curr_node = self.origin_node

        for i in range(len(self.action_sequence)):
            # Define an array to hold the pick up object + place object node seqs
            one_sequence = []

            # Find the node of interest and append that shortest path from curr_node
            next_node = self.action_sequence[i][2]
            one_sequence.append(self.shortest_paths[curr_node][next_node])

            # Find where the object belong to and append that shortest path from next_node
            bin_node = self.bins[self.action_sequence[i][1]]
            one_sequence.append(self.shortest_paths[next_node][bin_node])

            # Append the sequence to node sequence and update curr_node to bin_node
            self.node_sequence.append(one_sequence)
            curr_node = bin_node
        
        print("node seq", self.node_sequence)

    # STATE UPDATE FUNCS
    def update_distance(self, data):
        self.distance = data.ranges[0]
    def update_image(self, msg):
        self.image_data = msg

    # CALLBACK FUNCS
    def odom_callback(self, data):
        """ Save the odometry data """

        # Do nothing if not initialized
        if not self.initialized:
            return

        # Save robot pose to self.curr_pose
        self.curr_pose = data.pose.pose
    def object_detector_callback(self, data):
        """ Determine what object perception node detects"""

        if not self.initialized:
            return 
        
        if not self.going_to_pick_up:
            return

        if self.holding_object:
            self.curr_obj = None
            return

        self.curr_obj = data.object
        print(f"The object is {data.object}.")

    # GRANULAR GRIP MOVEMENT FUNCS
    def open_grip(self):
        open_grip = [0.015, 0.015]
        self.move_grip.go(open_grip)
    def close_grip(self):
        close_grip = [0.010, 0.010]
        self.move_grip.go(close_grip)

    # GRANULAR ARM MOVEMENT FUNCS
    def lower_arm(self):
        lower_pos = [0, 0.7, 0, -0.65]
        self.move_arm.go(lower_pos, wait=True)
    def upper_arm(self):
        upper_pos = [0, 0.05, -0.5, -0.65]
        self.move_arm.go(upper_pos, wait=True)
    def angle_arm(self):
        angle_pos = [0.75, 0.05, -0.5, -0.65]
        self.move_arm.go(angle_pos, wait=True)

    # GRANULAR TWIST MOVEMENT FUNCS
    def stop(self):
        self.twist.linear.x = 0
        self.twist.angular.z = 0
        self.cmd_vel_pub.publish(self.twist)
    def move_forward(self, secs=3):
        self.twist.linear.x = 0.1
        self.cmd_vel_pub.publish(self.twist)
        time.sleep(secs)
        self.stop()
    def move_back(self, secs=3):
        self.twist.linear.x = -0.1
        self.cmd_vel_pub.publish(self.twist)
        time.sleep(secs)
        self.stop()
    def turn_left(self):
        print("Turning left.")
        self.twist.angular.z = np.pi/6
        self.cmd_vel_pub.publish(self.twist)
        time.sleep(3)
        self.stop()
    def turn_right(self):
        print("Turning right.")
        self.twist.angular.z = -np.pi/6
        self.cmd_vel_pub.publish(self.twist)
        time.sleep(3)
        self.stop()

    # COMPOUND MOVEMENTS
    def pick_up(self):
        """ Picks up dumbbell and angles out of camera POV
        """
        self.lower_arm()
        self.close_grip()
        self.upper_arm()
        self.angle_arm()
        self.holding_object = True
        self.going_to_pick_up = False
    def let_go(self):
        """ Loosens grip and backs away from dumbbell.
        """
        print("Now letting go of dumbbell")
        self.upper_arm()
        self.lower_arm()
        self.open_grip()
        self.move_back()
        self.holding_object = False
        self.score += 1
    def go_around(self):
        """Takes robot around an object"""
        print("Starting go_around")
        self.turn_left()
        self.move_forward(3)
        self.turn_right()
        self.move_forward(5)
        self.turn_right()
        self.move_forward(3)
        self.turn_left()
        self.move_forward(3)
        self.finished_obj_action = True
        print("Finished go_around")

    # COMPOUND-COMPLEX MOVEMENTS
    def orient(self, p):
        """ Given a node number, orient robot to face that node """

        # Do nothing if not initialized
        if not self.initialized:
            return

        # Sleep for 1 second to make sure the position data is retrieved
        rospy.sleep(1)
        
        # Get the robot/node positions, calculate target angle and diff for 
        #   proportional control
        point = self.locations[p]
        target_angle = get_target_angle(self.curr_pose.position, (point[0], point[1]))
        diff = target_angle - get_yaw_from_pose(self.curr_pose)

        # Set linear velocity to 0 since we only care about angular orientation
        self.twist.linear.x = 0

        # If the angle difference is greater than a threshold, keep turning based
        #   on the robot's angle to the node with proportional control
        if abs(diff) > 0.1:
            self.twist.angular.z = self.kp * diff

        # Otherwise, the robot is now facing the node and we stop turning and 
        #   set self.oriented to True
        else:
            self.twist.angular.z = 0
            self.oriented = True

        # Publish the velocities
        self.cmd_vel_pub.publish(self.twist)
    def follow_yellow_line(self):
        """Drives forward indefinitely along yellow line"""
        # image seup
        yellow = {"lower":np.array([24,127,191]),"upper":np.array([34,255,255])}
        image = self.bridge.imgmsg_to_cv2(self.image_data,desired_encoding='bgr8')
        # boilerplate vars
        h, w, d = image.shape
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, yellow["lower"], yellow["upper"])
        # crop mask
        search_top, search_bot = int(3*h/4), int(3*h/4 + 20)
        mask[0:search_top, 0:w], mask[search_bot:h, 0:w] = 0, 0
        M = cv2.moments(mask)
        # if there are any yellow pixels found
        if M['m00'] > 0:
            # track center of yellow pixels
            cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            cv2.circle(image, (cx, cy), 20, (0,0,255), -1)

            err, k_p = w/2 - cx, 1.0 / 100.0
            self.twist.linear.x = 0.1
            self.twist.angular.z = k_p * err
            self.cmd_vel_pub.publish(self.twist)
        else: # stops if it cannot see yellow pixels
            self.twist.linear.x = 0
            self.twist.angular.z = 0.2
            self.cmd_vel_pub.publish(self.twist)
            print("cant see yellow line here!")
            # self.stop()
        # show the debugging window
        cv2.imshow("window", image)
        cv2.waitKey(3)
    def get_dist_from_node(self, dest):
        """Gets current distance from node"""
        p1 = (self.locations[dest][0], self.locations[dest][1])
        p2 = (self.curr_pose.position.x, self.curr_pose.position.y)
        return get_dist(p1, p2)
    def move_to_node(self, dest):
        print(f"Moving to node {dest}")
        r = rospy.Rate(3)
        self.oriented = False
        while not self.oriented:
            self.orient(dest)
            r.sleep()

        print("Finished orienting, now driving.")
        self.stop()
        time.sleep(1)
        self.object_action_router(dest)
        return

    # OBJECT HANDLER FUNCS
    def object_action_router(self, dest):
        """Decides what action each object"""
        obj, col = self.curr_obj, self.curr_obj_col
        print("Now deciding action to perform")

        dist_from_node = lambda: self.get_dist_from_node(dest)
        
        # Different criteria depending on what's on the node.
        if self.drop_off:
            criteria = 0.4
        elif obj == None:
            print("Moving to node.")
            criteria = 0.5
        elif obj == "dumbbell" or obj == "kettlebell":
            print("Moving to node with object to grab")
            criteria = .9
            self.open_grip()
            self.lower_arm()
        
        # Move along the yellow line to reach node
        while dist_from_node() > criteria:
            print(f"Currently {dist_from_node()} from node.")
            self.follow_yellow_line()

        # Drop off object if robot status says so
        if self.drop_off:
            self.drop_off_object()
            return
        
        # Robot needs to circle around the object if it's not being picked up
        if (not self.going_to_pick_up) and (self.distance < 0.5):
            self.go_around()
            return

        if obj == None: return

        self.finished_obj_action = False
        while not self.finished_obj_action:
            if obj == "kettlebell" or self.score >= 2:
                self.approach_and_pickup_kettlebell(col)
            elif obj == "dumbbell":
                self.approach_and_pickup_dumbbell(col)
            else:
                print(f"Unknown object: {obj}")
                # raise Exception("Unknown object")

    def approach_and_pickup_dumbbell(self, col="red"):
        """Approaches and picks up dumbbell of given color"""
        print(f"Approaching {col} dumbbell")
        # color options
        red = {"lower":np.array([0,190,160]),"upper":np.array([2,255,255])}
        green = {"lower":np.array([60,60,60]),"upper":np.array([65,255,250])}
        blue = {"lower":np.array([94,80,2]),"upper":np.array([126,255,255])}
        select = {"red": red, "green": green, "blue": blue}

        image = self.bridge.imgmsg_to_cv2(self.image_data,desired_encoding='bgr8')
        # boilerplate vars
        h, w, d = image.shape
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, select[col]["lower"], select[col]["upper"])
        M = cv2.moments(mask)
        if M['m00'] > 0:
            # determine the center of the colored pixels in the image
            cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            cv2.circle(image, (cx, cy), 20, (0,0,255), -1)

            # print("Colored pixels were found in the image.")
            print(f"Distance: {self.distance}, cy: {cy}")

            # pick up dumbbell when close enough
            dumbbell_close_enough = 300 < cy < 330 or 0.1 < self.distance < 0.2

            if dumbbell_close_enough: 
                print("Close to dumbbell! Now picking it up.")
                self.twist.linear.x = 0
                self.twist.angular.z = 0
                self.cmd_vel_pub.publish(self.twist)
                self.pick_up()
                self.finished_obj_action = True
                return
            else:
                print("Out of range of dumbbells. Moving forward.")
                # pid control variables!
                k_p, err = .01, w/2 - cx
                # alter trajectory accordingly
                self.twist.linear.x = 0.02
                self.twist.angular.z = k_p * err *.02
                self.cmd_vel_pub.publish(self.twist)
        else:
            print("No colored pixels -- spinning in place.")
            self.twist.linear.x = 0
            self.twist.angular.z = .1
            self.cmd_vel_pub.publish(self.twist)
        # show the debugging window
        cv2.imshow("window", image)
        cv2.waitKey(3)
    def approach_and_pickup_kettlebell(self, col="red"):
        """Approaches and picks up kettlebell of given color"""
        # color options
        red = {"lower":np.array([0,190,160]),"upper":np.array([2,255,255])}
        green = {"lower":np.array([60,60,60]),"upper":np.array([65,255,250])}
        blue = {"lower":np.array([94,80,2]),"upper":np.array([126,255,255])}
        black = {"lower":np.array([ 0, 0, 0]),"upper":np.array([179, 255, 20])}
        select = {"red": red, "green": green, "blue": blue}

        image = self.bridge.imgmsg_to_cv2(self.image_data,desired_encoding='bgr8')
        # boilerplate vars
        h, w, d = image.shape
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, black["lower"], black["upper"])
        M = cv2.moments(mask)
        if M['m00'] > 0:
            # determine the center of the colored pixels in the image
            cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            cv2.circle(image, (cx, cy), 20, (0,0,255), -1)

            # print("Colored pixels were found in the image.")
            print(f"Distance: {self.distance}, cx: {cx}, cy: {cy}")

            # if blue_cond or green_cond or red_cond:
            dumbbell_close_enough = 301 < cy < 330 or 0.1 < self.distance < 0.25

            if dumbbell_close_enough: 
                print("Close to kettle! Now picking it up.")
                self.twist.linear.x = 0
                self.twist.angular.z = 0
                self.cmd_vel_pub.publish(self.twist)
                self.pick_up()
                self.finished_obj_action = True
                return
            else:
                print("Out of range of kettles. Moving forward.")
                # pid control variables!
                k_p, err = .01, 3*w/8 - cx
                # alter trajectory accordingly
                self.twist.linear.x = 0.02
                self.twist.angular.z = k_p * err *.02
                self.cmd_vel_pub.publish(self.twist)
        else:
            print("No colored pixels -- spinning in place.")
            self.twist.linear.x = 0
            self.twist.angular.z = .1
            self.cmd_vel_pub.publish(self.twist)
        # show the debugging window
        cv2.imshow("window", image)
        cv2.waitKey(3)
    def drop_off_object(self):
        col = self.curr_obj_col
        # color options
        red = {"lower":np.array([0,190,160]),"upper":np.array([2,255,255])}
        green = {"lower":np.array([60,60,60]),"upper":np.array([65,255,250])}
        blue = {"lower":np.array([94,80,2]),"upper":np.array([126,255,255])}
        select = {"red": red, "green": green, "blue": blue}
        
        dropped = False
        while not dropped:
            image = self.bridge.imgmsg_to_cv2(self.image_data,desired_encoding='bgr8')
            # boilerplate vars
            h, w, d = image.shape
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, select[col]["lower"], select[col]["upper"])
            M = cv2.moments(mask)
            if M['m00'] > 0:
                # determine the center of the colored pixels in the image
                cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                cv2.circle(image, (cx, cy), 20, (255,255,255), -1)

                # print("Colored pixels were found in the image.")
                print(f"Distance: {self.distance}, cy: {cy}")

                # drop off dumbbell when close enough
                dumbbell_close_enough = cy > 410

                if dumbbell_close_enough: 
                    print("Close to dropoff! Now dropping.")
                    self.twist.linear.x = 0
                    self.twist.angular.z = 0
                    self.cmd_vel_pub.publish(self.twist)
                    self.let_go()
                    self.finished_obj_action = True
                    return
                else:
                    print("Out of range of dropoff. Moving forward.")
                    # pid control variables!
                    k_p, err = .01, w/2 - cx
                    # alter trajectory accordingly
                    self.twist.linear.x = 0.15
                    self.twist.angular.z = k_p * err *.08
                    self.cmd_vel_pub.publish(self.twist)
            else:
                print("No colored pixels -- spinning in place.")
                self.twist.linear.x = 0
                self.twist.angular.z = .1
                self.cmd_vel_pub.publish(self.twist)
            # show the debugging window
            cv2.imshow("window", image)
            cv2.waitKey(3)
            # self.let_go()
            print("dropped off")

    def run_sequence(self, sequence):
        """sequence represents a single group of actions.
        For example [[4,1], [1,4,5,6]]"""

        def set_obj_at_node(dest):
            """Side effect: sets object and object color props"""

            if self.holding_object:
                self.curr_obj = None
                return

            for obj,col,node in self.action_sequence:
                if node == dest:
                    # self.curr_obj = obj
                    self.curr_obj_col = col
                    self.going_to_pick_up = True
                    break
            else:
                self.curr_obj = None
                # self.curr_obj_col = None

        for seq in sequence:
            for dest in seq:
                # ignore first node in sequence; it represents the
                # node that robot is currently sitting on
                if dest == seq[0]: continue
                # last node in last sequence represents drop-off
                if dest == seq[-1] and seq == sequence[-1]: self.drop_off = True
                else: self.drop_off = False
                # execute!
                set_obj_at_node(dest)
                print("Going to pick up:", self.going_to_pick_up)
                print("Current obj:", self.curr_obj)
                self.move_to_node(dest)

    def run(self):
        while True:
            for sequence in self.node_sequence:
                self.run_sequence(sequence)

if __name__ == "__main__":
    node = RobotMovement()
    node.run()