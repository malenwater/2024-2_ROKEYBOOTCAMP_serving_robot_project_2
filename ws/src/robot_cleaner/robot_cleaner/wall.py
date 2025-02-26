import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import Log
import time

class Mapping(Node):
    def __init__(self):
        super().__init__("mapping_node")
        self.is_moving = False
        self.is_finish = False
        self.is_inital = False

        self.robot_x = 0.0
        self.robot_y = 0.0

        self.start_x = None
        self.start_y = None

        self.visited_walls = set()  # 방문한 벽 좌표 저장

        self.sub_map = self.create_subscription(OccupancyGrid, '/map', self.callback_mapping, 10)
        self.pub_goal = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.sub_status = self.create_subscription(Log, '/rosout', self.callback_goal, 10)
        self.sub_pose = self.create_subscription(Odometry, '/odom', self.callback_pose, 10)

    def callback_goal(self, msg):
        if 'Goal succeeded' in msg.msg:
            self.get_logger().info(msg.msg)
            self.is_moving = False

    def callback_pose(self, msg):
        self.robot_x = round(msg.pose.pose.position.x, 3)
        self.robot_y = round(msg.pose.pose.position.y, 3)
        
        if self.start_x is None and self.start_y is None:
            self.start_x = self.robot_x
            self.start_y = self.robot_y
            self.get_logger().info(f"Start position saved: ({self.start_x}, {self.start_y})")

    def callback_mapping(self, msg):
        if self.is_finish:
            self.get_logger().info('is_finish is True')
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

        pose_x = int((origin_x + self.robot_x) / per_pixel)
        pose_y = int((origin_y + self.robot_y) / per_pixel)
        goal_pose = self.goal_pose_detection(np_map, (pose_y, pose_x))

        if goal_pose[0] == -1:
            if self.start_x is not None and self.start_y is not None:
                goal_x = self.start_x
                goal_y = self.start_y
            else:
                self.get_logger().warn("Start position not found! Returning to (0,0).")
                goal_x, goal_y = 0.0, 0.0
            self.is_finish = True
        else:
            goal_x = (goal_pose[0] - init_pose_x) * per_pixel
            goal_y = (goal_pose[1] - init_pose_y) * per_pixel

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = goal_x
        goal.pose.position.y = goal_y
        goal.pose.orientation.z = 0.0
        goal.pose.orientation.w = 1.0
        goal.header.stamp = self.get_clock().now().to_msg()
        
        self.pub_goal.publish(goal)
        self.get_logger().info(f"Publishing goal: ({goal_x}, {goal_y})")
        self.is_moving = True

    def goal_pose_detection(self, map, home):
        height, width = map.shape
        free_zone_idx = np.argwhere(map == 0)
        
        dists = []
        poses = []

        for pose in free_zone_idx:
            if tuple(pose) in self.visited_walls:
                continue  # 이미 방문한 벽이라면 제외

            is_near_wall = any(map[pose[0]+dy, pose[1]+dx] == -1 
                               for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                               if 0 <= pose[0]+dy < height and 0 <= pose[1]+dx < width)

            if is_near_wall:
                self.visited_walls.add(tuple(pose))
                continue  # 벽을 방문했으므로 저장 후 제외

            dists.append(self.distance(pose, home))
            poses.append([pose[1], pose[0]])

        if not poses:
            self.get_logger().info('finish mapping!')
            return [-1, -1]

        goal_pose = poses[dists.index(max(dists))]
        self.get_logger().info(f"goal_pose: {goal_pose}")
        return goal_pose

    def distance(self, p1, p2):
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

if __name__ == '__main__':
    rclpy.init()
    node = Mapping()
    rclpy.spin(node)
    rclpy.shutdown()
    node.destroy_node()
