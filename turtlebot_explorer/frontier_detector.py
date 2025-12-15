#!/usr/bin/env python3
"""
Frontier Detector Node
Subscribes to: /map
Publishes to: /frontiers (PointStamped array as individual messages)
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PointStamped
import numpy as np


class FrontierDetector(Node):
    """
    Detects frontiers (boundaries between explored and unexplored areas).
    DELIBERATIVE component: Processes map to find exploration targets.
    """
    
    def __init__(self):
        super().__init__('frontier_detector')
    
        # Subscribe to map with proper QoS
        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos)
        
        # Publish frontier points
        self.frontier_pub = self.create_publisher(PointStamped, '/frontiers', 10)
        
        self.map_data = None
        self.map_info = None
        self.frontiers = []
        
        # Timer to periodically detect frontiers
        self.create_timer(1.0, self.detect_frontiers)  # 1 Hz
        
        self.get_logger().info('Frontier Detector Node started')
    
    def map_callback(self, msg):
        """Store the current map."""
        self.map_info = msg.info
        self.map_data = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)
    
    def detect_frontiers(self):
        """
        Main frontier detection algorithm.
        Finds cells that are free (0) and adjacent to unknown cells (-1).
        
        
        """
        if self.map_data is None:
            self.get_logger().warn('No map data yet!')
            return
        
        h, w = self.map_data.shape
        frontier_cells = []
        
        # Scan grid for frontier cells - Check all cells
        # Start at 1 and end at h/w-1 to safely check 8 neighbors
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                current = self.map_data[i, j]
                
                # Cell must be Free space (Standard ROS: 0)
                if current == 0:  
                    
                    # Check 8-connected neighbors
                    neighbors = [
                        self.map_data[i-1, j],
                        self.map_data[i+1, j],
                        self.map_data[i, j-1],
                        self.map_data[i, j+1],
                        self.map_data[i-1, j-1],
                        self.map_data[i-1, j+1],
                        self.map_data[i+1, j-1],
                        self.map_data[i+1, j+1]
                    ]
                    
                    # Frontier if any neighbor is Unknown (Standard ROS: -1)
                    has_unknown = any(n == -1 for n in neighbors)
                    
                    if has_unknown:
                        # Convert to world coordinates (using center of cell: +0.5 * resolution)
                        wx = self.map_info.origin.position.x + (j + 0.5) * self.map_info.resolution
                        wy = self.map_info.origin.position.y + (i + 0.5) * self.map_info.resolution
                        frontier_cells.append((wx, wy))
        
        self.frontiers = frontier_cells
        
        if len(frontier_cells) > 0:
            self.get_logger().info(f'✓ Found {len(frontier_cells)} frontiers!')
        else:
            self.get_logger().info('✗ No frontiers detected', throttle_duration_sec=3.0)
        
        # Publish all frontiers
        for wx, wy in self.frontiers:
            point_msg = PointStamped()
            point_msg.header.stamp = self.get_clock().now().to_msg()
            point_msg.header.frame_id = 'map'
            point_msg.point.x = wx
            point_msg.point.y = wy
            point_msg.point.z = 0.0
            self.frontier_pub.publish(point_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FrontierDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()