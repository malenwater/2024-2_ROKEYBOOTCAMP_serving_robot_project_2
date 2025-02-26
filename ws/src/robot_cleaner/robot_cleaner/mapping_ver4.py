import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid,Odometry
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import Log
from sklearn.cluster import KMeans
import time

'''

실시간으로 생성된 slam 맵을 분석
미지의 영역을 자율적으로 찾아 탐색하면서 최종적으로 맵 전체를 자동으로 완성하는 코드

'''
class Mapping(Node):
    def __init__(self):
        super().__init__("mapping_node")
        self.is_moving = False
        self.is_finish = False
        self.is_inital = False
        self.robot_x = 0.0
        self.robot_y = 0.0
        # 앞서, slam, => /map (실시간 맵 데이터)
        # nav2 => /odom (로봇의 현재위치 자세정보), /rosout(목표도착여부)
        self.sub_map = self.create_subscription(OccupancyGrid,'/map',self.callback_mapping,10)
        self.sub_pose = self.create_subscription(Odometry,'/odom',self.callback_pose,10)

        self.sub_status = self.create_subscription(Log,'/rosout',self.callback_goal,10)

        # /goal_pose 토픽을 발행한다 => 수신자는 Navigation2
        self.pub_goal = self.create_publisher(PoseStamped,'/goal_pose',10)
    # 로봇의 목표 지점 도착 여부를 확인하는 메서드
    def callback_goal(self,msg):
        if 'Goal succeeded' == msg.msg:
            self.get_logger().info(msg.msg)
            self.is_moving = False
    # 로봇의 현재 위치를 실시간으로 받아 업데이트
    def callback_pose(self,msg):
        self.robot_x = round(msg.pose.pose.position.x,3)
        self.robot_y = round(msg.pose.pose.position.y,3)
    # 맵 데이터를 받을 떄마다 목표를 계산
    #  로봇의 목표 위치를 업데이트하여 이동 명령을 내린다.
        # 초기 위치 저장 (추가 _)
        if self.initial_x is None and self.initial_y is None:
            self.initial_x = self.robot_x
            self.initial_y = self.robot_y
            self.get_logger().info(f"Initial position stored: ({self.initial_x}, {self.initial_y})")

    def callback_mapping(self,msg):
        #if self.is_moving == True:
        #    self.get_logger().info('is_moving is True')
        #    return
        if self.is_finish == True:
            self.get_logger().info('is_finish is True')
            return
        
        mapping = msg
        width = mapping.info.width
        height = mapping.info.height
        origin_x = -round(mapping.info.origin.position.x,3)
        origin_y = -round(mapping.info.origin.position.y,3)
        per_pixel = round(mapping.info.resolution,3)
        np_map = np.array(mapping.data).reshape(height,width)
        # km = KMeans(n_clusters=10)
        # y_pred = km.fit_predict(np_map)
        # print(len(y_pred))
        init_pose_x = int(origin_x/per_pixel)
        init_pose_y = int(origin_y/per_pixel)
        if self.is_inital == False:
            goal_pose = self.goal_pose_detection(np_map,(init_pose_y,init_pose_x))
            self.get_logger().info(f"init pose: {init_pose_x}, {init_pose_y}")
            self.is_inital = True
        else:
            pose_x = int((origin_x + self.robot_x)/per_pixel)
            pose_y = int((origin_y + self.robot_y)/per_pixel)
            goal_pose = self.goal_pose_detection(np_map,(pose_y,pose_x))
            self.get_logger().info(f"robot pose {pose_x} {pose_y}")
            self.get_logger().info(f"robot act pose {self.robot_x} {self.robot_y}")
        # 종료 (수정 )
        if goal_pose[0] == -1:
            goal = PoseStamped()
            goal.header.frame_id = 'map'
            # goal.pose.position.x = 0
            # goal.pose.position.y = 0
            goal.pose.position.x = self.initial_x
            goal.pose.position.y = self.initial_y
            goal.pose.orientation.z = 0.0
            goal.pose.orientation.w = 0.0
            goal.header.stamp = self.get_clock().now().to_msg()
            #for _ in range(5):
            self.pub_goal.publish(goal)
            self.is_finish = True
            self.get_logger().info('go to init pose')
            return
        #self.get_logger().info('origin:',origin_x,origin_y)
        local_map_x = goal_pose[0] - init_pose_x
        local_map_y = goal_pose[1] - init_pose_y
        goal_x = local_map_x * per_pixel
        goal_y = local_map_y * per_pixel
        #print('local map', local_map_x, local_map_y)
        self.get_logger().info(f"goal: {goal_x}, {goal_y}")
        #print("global:",global_goal_x, global_goal_y)
        #print("local:",local_map)
        time.sleep(0.5)
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = goal_x
        goal.pose.position.y = goal_y
        goal.pose.orientation.z = 0.0
        goal.pose.orientation.w = 0.0
        goal.header.stamp = self.get_clock().now().to_msg()
        #for _ in range(5):
        self.pub_goal.publish(goal)
        self.get_logger().info("publish goal")
        self.is_moving = True
        # for raw in np_map:
        #     print(raw)
        #time.sleep(1)
    
    # 두 점 사이 유클리드 거리의 제곱을 계산하는 메서드
    def distance(self,p1,p2):
        return ( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )
    # 맵 데이터에서, 탐색 가능한 가장 효율적인 목표 위치를 찾는다.
    #  미지 영역 옆의 자유 공간을 목표로 선택한다. 다음으로 이동할 가장 최적의 목표 위치를 탐색

    '''
    kante_아이디어
    현재 4방향 검사를 수행하고 있기에, 8군데 방향으로 (대각선 포함) 으로 확장

    goal_pose_detection()코드
    맵 데이터 -> 미탐색영억(-1) 지역 근처의 최적 목표 지점을 찾는 역할이다.


    '''
    def goal_pose_detection(self, map, home):
        home_map = map
        height = home_map.shape[0]
        width = home_map.shape[1]
        init_location = home
        # free_zone은 0인 부분으로서, 이동가능한 영역이다.
        free_zone_idx = np.argwhere(home_map == 0)
        dists = []
        poses = []

        for pose in free_zone_idx:
            is_insert = False
            is_in = [True] * 8  # 8방향 검사할지 여부 (위, 우상, 우, 우하, 하, 좌하, 좌, 좌상)

            # 경계를 벗어나는 경우 검사 X
            # 1. 위쪽 끝, == 0번째 행 == 더이상 좌상으로 탐색이 불가능하다. 
            if pose[0] == 0:
                is_in[0] = is_in[1] = is_in[7] = False  # 위쪽 3방향
            # 2. 우상단, == 더 오른쪽으로 갈수 없다.
            if pose[1] == width-1:
                is_in[1] = is_in[2] = is_in[3] = False  # 오른쪽 3방향
            # 3. 좌하단, == 더 우하,향 불가능하다.
            if pose[0] == height-1:
                is_in[3] = is_in[4] = is_in[5] = False  # 아래쪽 3방향
            # 4. 좌 상단 == 더 좌상,향 불가능하다. 
            if pose[1] == 0:
                is_in[5] = is_in[6] = is_in[7] = False  # 왼쪽 3방향

            # 8방향 탐색 (미탐색 영역 주변의 자유 공간 찾기)
            directions = [
                (-1,  0),  # 상
                (-1,  1),  # 우상
                ( 0,  1),  # 우
                ( 1,  1),  # 우하
                ( 1,  0),  # 하
                ( 1, -1),  # 좌하
                ( 0, -1),  # 좌
                (-1, -1),  # 좌상
            ]
            for i, (dy, dx) in enumerate(directions):
                # 방향 + 미지의 공간(-1)
                if is_in[i] and home_map[pose[0] + dy][pose[1] + dx] == -1:
                    is_insert = True

            # 주변의 자유 공간 개수 계산 (이동 안정성 확보)
            # 0의 개수= 자유공간 개수를 세서, 안정적인 목표 지점을 찾는다. 
            count = sum(
                1 for i, (dy, dx) in enumerate(directions)
                if is_in[i] and home_map[pose[0] + dy][pose[1] + dx] == 0
            )

            # 미탐색 영역 주변 자유 공간이 충분하면 목표로 설정
            # is_insert=True / 미탐색 지역(-1) 근처?
            # count >= 4  / 자유공간(0)이 4개 이상확보?
            if is_insert and count >= 4:
                dists.append(self.distance(pose, init_location))
                poses.append([pose[1], pose[0]])

        if len(poses) == 0:
            self.get_logger().info('finish mapping!')
            return [-1, -1]

        # 가장 먼 위치를 목표로 선택하여 탐색 영역 확장
        goal_pose = poses[dists.index(max(dists))]
        self.get_logger().info(f"goal_pose: {goal_pose}")
        return goal_pose

if __name__ == '__main__':
    #print(np.array(m).reshape(7,6))
    rclpy.init()
    node = Mapping()
    rclpy.spin(node)
    rclpy.shutdown()
    node.destroy_node()