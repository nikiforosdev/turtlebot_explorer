#!/usr/bin/env python3
"""
Frontier Detector Node
Subscribes to: /map
Publishes to: /frontiers (custom message with list of points)

For simplicity, we'll publish to a topic that the controller subscribes to.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PointStamped, Point
import numpy as np


class FrontierDetector(Node):
    """
    Detects frontiers (boundaries between explored and unexplored areas).
    
    DELIBERATIVE component: Processes map to find exploration targets.
    """
    
    def __init__(self):
        super().__init__('frontier_detector')
        
        # Subscribe to map
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        
        # Publish frontiers (we'll use a simple approach - publish via parameter server or topic)
        # For simplicity, store internally and let controller query via a service or shared state
        # But to keep it truly modular with ROS principles, we should use a custom message
        # For now, let's just store it and the controller will access it
        
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
            return
        
        h, w = self.map_data.shape
        frontier_cells = []
        
        # Scan grid for frontier cells
        for i in range(5, h-5, 5):  # Sample every 5th cell for performance
            for j in range(5, w-5, 5):
                # Check if current cell is free
                if self.map_data[i, j] == 0:
                    # Check 4-connected neighbors for unknown cells
                    neighbors = [
                        self.map_data[i-1, j],
                        self.map_data[i+1, j],
                        self.map_data[i, j-1],
                        self.map_data[i, j+1]
                    ]
                    
                    # If any neighbor is unknown, this is a frontier
                    if -1 in neighbors:
                        # Convert to world coordinates
                        wx = self.map_info.origin.position.x + j * self.map_info.resolution
                        wy = self.map_info.origin.position.y + i * self.map_info.resolution
                        frontier_cells.append((wx, wy))
        
        self.frontiers = frontier_cells
        
        if len(self.frontiers) > 0:
            self.get_logger().info(f'Detected {len(self.frontiers)} frontier points', 
                                 throttle_duration_sec=2.0)
        else:
            self.get_logger().info('No frontiers detected', throttle_duration_sec=5.0)
    
    def get_frontiers(self):
        """Returns current list of frontiers."""
        return self.frontiers


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