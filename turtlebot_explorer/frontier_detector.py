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
        
        # Subscribe to map
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        
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
        
        # DEBUG: What values exist in the map?
        unique_values = np.unique(self.map_data)
        self.get_logger().info(f'Map shape: {h}x{w}, Unique values in map: {unique_values}')
        
        # Count each type
        free_cells = np.sum(self.map_data == 0)
        unknown_cells = np.sum(self.map_data == -1)
        occupied_cells = np.sum(self.map_data == 100)
        
        self.get_logger().info(f'Free: {free_cells}, Unknown: {unknown_cells}, Occupied: {occupied_cells}')
        
        # If no unknown cells, we can't find frontiers!
        if unknown_cells == 0:
            self.get_logger().warn('NO UNKNOWN CELLS IN MAP - Cannot detect frontiers!')
            return
        
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
        
        # Publish all frontiers
        for wx, wy in self.frontiers:
            point_msg = PointStamped()
            point_msg.header.stamp = self.get_clock().now().to_msg()
            point_msg.header.frame_id = 'map'
            point_msg.point.x = wx
            point_msg.point.y = wy
            point_msg.point.z = 0.0
            self.frontier_pub.publish(point_msg)
        
        if len(self.frontiers) > 0:
            self.get_logger().info(f'Detected {len(self.frontiers)} frontier points', 
                                 throttle_duration_sec=2.0)
        else:
            self.get_logger().info('No frontiers detected', throttle_duration_sec=5.0)


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