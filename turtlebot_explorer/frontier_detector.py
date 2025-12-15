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
        Finds cells that are free and adjacent to unknown cells.
        """
        if self.map_data is None:
            self.get_logger().warn('No map data yet!')
            return
        
        h, w = self.map_data.shape
        frontier_cells = []
        
        # DEBUG: Check map statistics
        unique_vals = np.unique(self.map_data)
        self.get_logger().info(f'Map: {h}x{w}, values range: {unique_vals[0]} to {unique_vals[-1]}', 
                            throttle_duration_sec=5.0)
        
        # Scan grid for frontier cells with finer sampling
        for i in range(2, h-2, 3):  # Every 3rd cell (was 5)
            for j in range(2, w-2, 3):
                current = self.map_data[i, j]
                
                # Cell must be definitely free (very low values in 0-99 scale)
                if current >= 0 and current < 10:  # Free space
                    
                    # Check 8-connected neighbors (not just 4)
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
                    
                    # Frontier if any neighbor is unknown/unexplored (mid-to-high values)
                    # In 0-99 scale: unknown typically 40-60, occupied is 70-99
                    has_unknown = any(50 <= n <= 70 for n in neighbors)
                    
                    if has_unknown:
                        # Convert to world coordinates
                        wx = self.map_info.origin.position.x + j * self.map_info.resolution
                        wy = self.map_info.origin.position.y + i * self.map_info.resolution
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