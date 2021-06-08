#!/usr/bin/env python3

import rospy
import os
import numpy as np
from numpy.random import choice

from roomba.msg import QLearningReward, QMatrix, QMatrixRow, RobotAction


# Path of directory on where this file is located
path_prefix = os.path.dirname(__file__) + "/action_states/"

# Header for formatting output
print_header = "=" * 10


def convert_q_matrix_to_list(qmatrix):
    """ Helper function to convert Q-matrix to list """

    res = []

    for qrow in qmatrix:
        res.append(qrow.q_matrix_row)

    return res


def print_state(state, actions):
    """ Helper function to print a state """

    output = ""
    output += "robot: at node " + str(state[0]) + "; "

    is_at_origin = lambda n: "at origin" if n == 0 else "at bin"

    for i in range(len(actions)):
        output += actions[i]["color"] + " " + actions[i]["object"] + ": " + is_at_origin(state[i + 1]) + "; "

    print(output)


class QLearning(object):


    def __init__(self):

        # Once everything is set up this will be set to true
        self.initialized = False

        # Initialize this node
        rospy.init_node("q_learning")

        # Set up publishers
        self.q_matrix_pub = rospy.Publisher("/roomba/q_matrix", QMatrix, queue_size = 10)
        self.robot_action_pub = rospy.Publisher("/roomba/robot_action", RobotAction, queue_size = 10)
        
        # A counter to keep track of how many iterations have passed
        self.cnt = 0

        # Set up subscriber
        rospy.Subscriber("/roomba/reward", QLearningReward, self.reward_received)

        # Fetch pre-built action matrix. This is a 2d numpy array where row indexes
        # correspond to the starting state and column indexes are the next states.
        #
        # A value of -1 indicates that it is not possible to get to the next state
        # from the starting state. Values 0-5 correspond to what action is needed
        # to go to the next state.
        #
        # e.g. self.action_matrix[0][1] = 5
        self.action_matrix = np.loadtxt(path_prefix + "action_matrix.csv", delimiter = ',')

        # Fetch actions. These are the only 6 possible actions the system can take.
        # self.actions is an array of dictionaries where the row index corresponds
        # to the action number, and the value has the following form:
        # { object: "dumbbell", color: "red", node: 1}
        self.actions = np.genfromtxt(path_prefix + "objects.csv", dtype = 'str', delimiter = ',')
        self.actions = list(map(
            lambda x: {"object": str(x[0]), "color": str(x[1]), "node": int(x[2])},
            self.actions
        ))

        # Fetch states. There are 576 states. Each row index corresponds to the
        # state number, and the value is a list of 7 items indicating the state of
        # the robot, the states of the objects in the order defined in objects.csv
        # e.g. [[0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0], ..., [2, 1, 0, 0, 0, 0, 1]]
        # e.g. [2, 1, 0, 0, 0, 0, 1] indicates that the robot is at node 2, red dumbbell
        # is at red bin, blue ball is at blue bin, and the rest are at the origin
        # A value of 0 corresponds to at the origin. 1 corresponds to at red/blue bins 
        # based on the color of the object
        # Note: that not all states are possible to get to.
        self.states = np.loadtxt(path_prefix + "states.csv", delimiter = ',')
        self.states = list(map(lambda x: list(map(lambda y: int(y), x)), self.states))

        # Set the origin state, where the robot first starts (changes with graph)
        # Note: state 256 is where the robot is at starting position + all objects
        #   are at their starting positions (need to change it based on graph)
        self.origin_state = 256

        # Initialize current state and keep track of the next state
        self.curr_state = self.origin_state
        self.next_state = 0

        # Initialize current action index
        self.curr_action = 0

        # Initialize variables to define static status and keep track of how many 
        #   iterations have the Q-matrix remained static
        self.epsilon = 3
        self.static_iter_threshold = 40
        self.static_tracker = 0

        # Initialize and publish Q-matrix
        self.q_matrix = QMatrix()
        self.initialize_q_matrix()
        self.q_matrix_pub.publish(self.q_matrix)

        # Now everything is initialized, sleep for 1 second to make sure
        self.initialized = True
        rospy.sleep(1)
        
        # Start with a random action
        self.select_random_action()


    def initialize_q_matrix(self):
        """ Initialize the Q-matrix with all 0s to start """

        # Loop over 576 rows and 6 columns to set up the matrix
        for i in range(len(self.states)):
            q_matrix_row = QMatrixRow()
            for j in range(len(self.actions)):
                q_matrix_row.q_matrix_row.append(0)
            self.q_matrix.q_matrix.append(q_matrix_row)


    def select_random_action(self):
        """ Select a random action based on current state and publish it """

        self.cnt += 1

        # Do nothing if Q-matrix is not yet initialized
        if not self.initialized:
            print(print_header + "not initialized" + print_header)
            return
        
        # Identify current state and find the row corresponding with that state in the 
        #   action matrix, this is all the valid + invalid actions a robot can take
        curr_state = self.curr_state
        actions_in_row = self.action_matrix[curr_state]

        # Filter out the invalid actions from the row of actions
        filtered_actions_in_row = list(filter(lambda x: x != -1, actions_in_row))

        # If there are no possible actions to take: reset current state to state 256
        while len(filtered_actions_in_row) == 0:
            print(print_header + "no action to take" + print_header)
            self.curr_state = self.origin_state
            curr_state = self.curr_state
            actions_in_row = self.action_matrix[curr_state]
            filtered_actions_in_row = list(filter(lambda x: x != -1, actions_in_row))
            
        # Randomly select an action from the row, assign that action to self.curr_action
        #   and find its index in the row to assign it to self.next_state
        selected_action = int(choice(filtered_actions_in_row))
        self.curr_action = selected_action
        self.next_state = np.where(actions_in_row == selected_action)[0][0]

        # Get the object, color and the node for the selected action
        obj = self.actions[selected_action]["object"]
        clr = self.actions[selected_action]["color"]
        node = self.actions[selected_action]["node"]

        # Set up a RobotAction() msg and publish it
        robot_action = RobotAction()
        robot_action.object = obj
        robot_action.color = clr
        robot_action.node = node
        self.robot_action_pub.publish(robot_action)

        # For testing: print published action
        print(print_header + f"[{self.cnt}] published a new action: {obj}, {clr}, {node}" + print_header)


    def update_q_matrix(self, reward):
        """ Apply the Q-learning algorithm to update and publish the Q-matrix """

        # Initialize variables to be used
        curr_state = self.curr_state
        next_state = self.next_state
        curr_action = self.curr_action

        # Set up parameters for the algorithm
        alpha = 1
        gamma = 0.5

        # Apply algorithm to update the q value for a state-action pair
        old_q_value = self.q_matrix.q_matrix[curr_state].q_matrix_row[curr_action]
        new_q_value = old_q_value + int(alpha * (reward + gamma * max(self.q_matrix.q_matrix[next_state].q_matrix_row) - old_q_value))
        self.q_matrix.q_matrix[curr_state].q_matrix_row[curr_action] = new_q_value

        # Now, move the current state on to the next state
        self.curr_state = next_state

        # For testing: print current state
        print_state(self.states[self.curr_state], self.actions)

        # Check if the change in q-value is static or not and update the tracker
        if abs(old_q_value - new_q_value) <= self.epsilon:
            self.static_tracker += 1
        else:
            self.static_tracker = 0

        # Publish the Q-matrix
        self.q_matrix_pub.publish(self.q_matrix)


    def is_converged(self):
        """ Check if the Q-matrix has converged """

        # If the Q-matrix has remained static for a certain amount of time, 
        #   then it is defined to be convergent
        if self.static_tracker >= self.static_iter_threshold:
            return True

        return False


    def save_q_matrix(self):
        """ Save Q-matrix as a csv file once it's converged to avoid retraining """

        # Save the Q-matrix as a csv file
        data = self.q_matrix.q_matrix
        data = convert_q_matrix_to_list(data)
        print(f"type of data[0]: {type(data[0])}")
        data = np.asarray(data)

        np.savetxt(os.path.dirname(__file__) + "/q_matrix.csv", data, fmt='%5s', delimiter = ',')

    
    def reward_received(self, data):
        """ Process received reward after an action """

        # Update the Q-matrix
        self.update_q_matrix(data.reward)

        if self.is_converged():
            # If the Q-matrix has converged, then we will save it
            self.save_q_matrix()
            print(print_header + f"matrix saved after {self.cnt} actions!" + print_header)
            exit()
        else:
            # If not, we continue to make random actions
            self.select_random_action()

            # Set self.curr_state back to self.origin_state if the world is reset
            if data.reset_world:
                self.curr_state = self.origin_state


    def run(self):
        """ Run the node """

        # Keep the program alive
        rospy.spin()


if __name__ == "__main__":

    # Declare a node and run it
    node = QLearning()
    node.run()
