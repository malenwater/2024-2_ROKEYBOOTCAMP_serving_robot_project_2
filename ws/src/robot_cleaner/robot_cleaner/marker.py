import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

class MarkerPublisher(Node):
    def __init__(self):
        super().__init__('marker_publisher')
        self.publisher = self.create_publisher(Marker, 'visualization_marker', 10)
        self.timer = self.create_timer(1.0, self.publish_marker)  # 1초마다 실행

    def publish_marker(self):
        marker = Marker()
        marker.header.frame_id = "map"  # RViz2의 기준 좌표계
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "my_namespace"
        marker.id = 0
        marker.type = Marker.POINTS  # 포인트 타입 마커
        marker.action = Marker.ADD

        # 마커 색상 및 크기 설정
        marker.scale.x = 0.1  # 포인트 크기
        marker.scale.y = 0.1
        marker.color.a = 1.0  # 투명도
        marker.color.r = 1.0  # 빨간색
        marker.color.g = 0.0
        marker.color.b = 0.0

        # 마커 위치 추가
        point = Point()
        point.x = 1.0
        point.y = 2.0
        point.z = 0.0
        marker.points.append(point)

        self.publisher.publish(marker)
        self.get_logger().info('Published marker at (1.0, 2.0, 0.0)')

def main(args=None):
    rclpy.init(args=args)
    node = MarkerPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
