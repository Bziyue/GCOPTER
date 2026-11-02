#!/usr/bin/env python3
"""Integration test against a running direct_corridor_node in an isolated ROS domain."""
import struct
import time
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String, Float64MultiArray
from visualization_msgs.msg import MarkerArray

rclpy.init()
node = rclpy.create_node('direct_corridor_smoke')
received = {'status': '', 'controls': [], 'markers': [], 'reference': []}
qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
subscriptions = [
    node.create_subscription(String, '/direct_corridor/status', lambda m: received.update(status=m.data), qos),
    node.create_subscription(Float64MultiArray, '/direct_corridor/controls', lambda m: received.update(controls=m.data), qos),
    node.create_subscription(MarkerArray, '/direct_corridor/markers', lambda m: received.update(markers=m.markers), qos),
    node.create_subscription(Path, '/direct_corridor/reference', lambda m: received.update(reference=m.poses), qos),
]
cloud_pub = node.create_publisher(PointCloud2, '/direct_corridor/map', 1)
path_pub = node.create_publisher(Path, '/direct_corridor/seed', 1)

def wait(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=.02)
        if predicate():
            return
    raise AssertionError(str(received['status']))

def cloud(points):
    m = PointCloud2()
    m.header.frame_id = 'map'
    m.height = 1
    m.width = len(points)
    m.fields = [PointField(name=name, offset=i*4, datatype=PointField.FLOAT32, count=1)
                for i, name in enumerate(('x', 'y', 'z'))]
    m.point_step = 12
    m.row_step = 12*len(points)
    m.data = b''.join(struct.pack('<fff', *p) for p in points)
    return m

def seed(points, frame='map'):
    m = Path()
    m.header.frame_id = frame
    for p in points:
        pose = PoseStamped()
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = p
        pose.pose.orientation.w = 1.
        m.poses.append(pose)
    return m

try:
    wait(lambda: cloud_pub.get_subscription_count() and path_pub.get_subscription_count())
    cloud_pub.publish(cloud([(0., 4., 2.)]))
    wait(lambda: received['status'] == 'WAITING_FOR_SEED')
    path_pub.publish(seed([(-3., 0., 2.), (0., 0., 2.), (3., 1., 2.)]))
    wait(lambda: received['status'].startswith('SUCCESS') and len(received['controls']) > 0 and len(received['reference']) > 0)
    assert len(received['controls']) % 66 == 0
    first, last = received['reference'][0].pose.position, received['reference'][-1].pose.position
    assert abs(first.x+3) < 1e-10 and abs(last.x-3) < 1e-10 and abs(last.y-1) < 1e-10
    received['status'] = ''
    cloud_pub.publish(cloud([(0., 0., 2.)]))
    wait(lambda: received['status'].startswith('SEED_COLLISION') and not received['controls'] and not received['reference'])
    wait(lambda: len(received['markers']) == 1 and received['markers'][0].action == 3)
    # A new valid map must regenerate the saved seed, not resurrect old coefficients.
    received['status'] = ''
    cloud_pub.publish(cloud([(0., 4., 2.)]))
    wait(lambda: received['status'].startswith('SUCCESS') and len(received['controls']) > 0)
    path_pub.publish(seed([(-3., 0., 2.), (3., 0., 2.)], 'wrong_frame'))
    wait(lambda: received['status'].startswith('INVALID_SEED') and not received['controls'])
    # Malformed payload is rejected before any iterator reads it.
    bad = cloud([(0., 4., 2.)])
    bad.data = b'\x00'
    cloud_pub.publish(bad)
    wait(lambda: received['status'].startswith('INVALID_MAP') and not received['controls'])
    print('PASS: ROS2 seed/map -> controls/reference/markers, exact endpoints, map invalidation, recovery, frame and malformed-cloud rejection')
finally:
    node.destroy_node()
    rclpy.shutdown()
