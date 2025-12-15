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
        Main frontier detection algorithm with clustering.
        1. Find all raw frontier cells (Free=0, neighbor Unknown=-1).
        2. Cluster nearby cells.
        3. Publish the centroid of each cluster as a goal.
        """
        if self.map_data is None:
            self.get_logger().warn('No map data yet!')
            return
        
        h, w = self.map_data.shape
        raw_frontier_cells_grid = []
        
        # 1. FIND RAW FRONTIER CELLS (Grid Coordinates)
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                current = self.map_data[i, j]
                
                # Cell must be Free space (Standard ROS: 0)
                if current == 0:  
                    neighbors = [
                        self.map_data[i+di, j+dj] 
                        for di in [-1, 0, 1] 
                        for dj in [-1, 0, 1] 
                        if not (di == 0 and dj == 0)
                    ]
                    
                    # Frontier if any neighbor is Unknown (Standard ROS: -1)
                    has_unknown = any(n == -1 for n in neighbors)
                    
                    if has_unknown:
                        raw_frontier_cells_grid.append((j, i)) # Note: (col, row) -> (x, y)
        
        if not raw_frontier_cells_grid:
            self.get_logger().info('✗ No frontiers detected', throttle_duration_sec=3.0)
            self.frontiers = []
            return
            
        # 2. CLUSTER FRONTIER CELLS
        # This is a simple nearest-neighbor clustering approach
        clusters = []
        
        for p1_x, p1_y in raw_frontier_cells_grid:
            found_cluster = False
            
            for cluster in clusters:
                # Check if p1 is close to the centroid of this cluster
                c_x, c_y = cluster['centroid']
                
                # Use grid distance squared for efficiency
                dist_sq = (p1_x - c_x)**2 + (p1_y - c_y)**2
                
                if dist_sq < self.cluster_dist_sq:
                    # Add to cluster and update centroid
                    cluster['points'].append((p1_x, p1_y))
                    
                    # Update centroid (simple running average)
                    count = len(cluster['points'])
                    new_c_x = (c_x * (count - 1) + p1_x) / count
                    new_c_y = (c_y * (count - 1) + p1_y) / count
                    cluster['centroid'] = (new_c_x, new_c_y)
                    
                    found_cluster = True
                    break
            
            if not found_cluster:
                # Start a new cluster
                clusters.append({
                    'centroid': (p1_x, p1_y), 
                    'points': [(p1_x, p1_y)]
                })

        # 3. PUBLISH CENTROIDS (World Coordinates)
        self.frontiers = []
        for cluster in clusters:
            gx, gy = cluster['centroid']
            
            # Convert centroid grid coordinates to world coordinates
            wx = self.map_info.origin.position.x + gx * self.map_info.resolution + (self.map_info.resolution / 2.0)
            wy = self.map_info.origin.position.y + gy * self.map_info.resolution + (self.map_info.resolution / 2.0)
            self.frontiers.append((wx, wy))
            
            # Publish point
            point_msg = PointStamped()
            point_msg.header.stamp = self.get_clock().now().to_msg()
            point_msg.header.frame_id = 'map'
            point_msg.point.x = wx
            point_msg.point.y = wy
            point_msg.point.z = 0.0
            self.frontier_pub.publish(point_msg)

        self.get_logger().info(f'✓ Found {len(self.frontiers)} viable frontier clusters!')


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