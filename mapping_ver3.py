import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import Log
from sklearn.cluster import KMeans
import time
import math

class Mapping(Node):
    def __init__(self):
        super().__init__("mapping_node")
        self.is_moving = False
        self.is_finish = False
        self.is_initial = False
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.initial_x = None  # 초기 위치 저장 변수
        self.initial_y = None

        self.sub_map = self.create_subscription(OccupancyGrid, '/map', self.callback_mapping, 10)
        self.pub_goal = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.sub_status = self.create_subscription(Log, '/rosout', self.callback_goal, 10)
        self.sub_pose = self.create_subscription(Odometry, '/odom', self.callback_pose, 10)

    def callback_goal(self, msg):
        if 'Goal succeeded' == msg.msg:
            self.get_logger().info(msg.msg)
            self.is_moving = False

    def callback_pose(self, msg):
        self.robot_x = round(msg.pose.pose.position.x, 3)
        self.robot_y = round(msg.pose.pose.position.y, 3)
        
        # 초기 위치 저장
        if self.initial_x is None and self.initial_y is None:
            self.initial_x = self.robot_x
            self.initial_y = self.robot_y
            self.get_logger().info(f"Initial position stored: ({self.initial_x}, {self.initial_y})")

    def callback_mapping(self, msg):
        if self.is_finish:
            return
        
        mapping = msg
        width = mapping.info.width
        height = mapping.info.height
        origin_x = -round(mapping.info.origin.position.x, 3)
        origin_y = -round(mapping.info.origin.position.y, 3)
        per_pixel = round(mapping.info.resolution, 3)
        np_map = np.array(mapping.data).reshape(height, width)

        init_pose_x = int(origin_x / per_pixel)
        init_pose_y = int(origin_y / per_pixel)

        if not self.is_initial:
            goal_pose = self.goal_pose_detection(np_map, (init_pose_y, init_pose_x))
            self.get_logger().info(f"Init pose: {init_pose_x}, {init_pose_y}")
            self.is_initial = True
        else:
            pose_x = int((origin_x + self.robot_x) / per_pixel)
            pose_y = int((origin_y + self.robot_y) / per_pixel)
            goal_pose = self.goal_pose_detection(np_map, (pose_y, pose_x))
            self.get_logger().info(f"Robot pose {pose_x} {pose_y}")
            self.get_logger().info(f"Robot actual pose {self.robot_x} {self.robot_y}")

        if goal_pose[0] == -1:
            self.return_to_initial_position()
            return
        
        goal_x = (goal_pose[0] - init_pose_x) * per_pixel
        goal_y = (goal_pose[1] - init_pose_y) * per_pixel
        
        self.publish_goal(goal_x, goal_y)

    def return_to_initial_position(self):
        if self.initial_x is not None and self.initial_y is not None:
            self.get_logger().info("Returning to initial position...")
            self.publish_goal(self.initial_x, self.initial_y)
            self.is_finish = True

    def publish_goal(self, x, y):
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.z = 0.0
        goal.pose.orientation.w = 1.0
        goal.header.stamp = self.get_clock().now().to_msg()
        self.pub_goal.publish(goal)
        self.get_logger().info(f"Publishing goal: ({x}, {y})")
        self.is_moving = True

if __name__ == '__main__':
    rclpy.init()
    node = Mapping()
    rclpy.spin(node)
    node.get_logger().info("Exploration finished. Waiting before shutdown...")
    time.sleep(3)
    rclpy.shutdown()
    node.destroy_node()
