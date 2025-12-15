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
import math
from sklearn.cluster import DBSCAN
# Note: If DBSCAN is not available, you must use the iterative clustering function provided below.
# I will use the iterative approach to avoid external dependencies.

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
        
        # Parameters for clustering: Cluster distance in grid cells (CRITICAL)
        self.cluster_dist = 5  # Max distance in cells to form a cluster
        
        # Timer to periodically detect frontiers
        self.create_timer(1.0, self.detect_frontiers)  # 1 Hz
        
        self.get_logger().info('Frontier Detector Node started')
    
    def map_callback(self, msg):
        """Store the current map."""
        self.map_info = msg.info
        self.map_data = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)

    def cluster_frontiers(self, raw_frontier_cells_grid):
        """Simple iterative clustering algorithm."""
        
        if not raw_frontier_cells_grid:
            return []

        # Convert list of tuples to a NumPy array for efficient distance calculation
        points = np.array(raw_frontier_cells_grid)
        
        # Initial cluster list: list of lists, where each inner list contains indices from `points`
        clusters = []
        is_clustered = np.zeros(len(points), dtype=bool)

        for i in range(len(points)):
            if is_clustered[i]:
                continue
            
            # Start a new cluster with point i
            current_cluster_indices = [i]
            is_clustered[i] = True
            
            # Iteratively find neighbors within the cluster distance
            # Use NumPy's efficient distance calculation
            
            # Calculate distance squared from point i to all other points
            p1 = points[i]
            distances_sq = np.sum((points - p1)**2, axis=1)
            
            # Find neighbors within the cluster distance (squared)
            # cluster_dist_sq = self.cluster_dist * self.cluster_dist
            cluster_dist_sq = self.cluster_dist ** 2

            # Neighbors are points that are close AND have not been clustered yet
            neighbors_indices = np.where((distances_sq < cluster_dist_sq) & (~is_clustered))[0]
            
            # Add all neighbors to the current cluster
            for j in neighbors_indices:
                current_cluster_indices.append(j)
                is_clustered[j] = True
            
            clusters.append(current_cluster_indices)

        # Calculate centroids of the final clusters
        centroids = []
        for cluster_indices in clusters:
            cluster_points = points[cluster_indices]
            # Centroid is the average of x and y coordinates
            centroid_x = np.mean(cluster_points[:, 0])
            centroid_y = np.mean(cluster_points[:, 1])
            centroids.append((centroid_x, centroid_y))
            
        return centroids
    
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
        # Use the corrected clustering function
        cluster_centroids_grid = self.cluster_frontiers(raw_frontier_cells_grid)

        # 3. PUBLISH CENTROIDS (World Coordinates)
        self.frontiers = []
        for gx_float, gy_float in cluster_centroids_grid:
            # Convert centroid grid coordinates (which are floats due to averaging) to world coordinates
            wx = self.map_info.origin.position.x + gx_float * self.map_info.resolution + (self.map_info.resolution / 2.0)
            wy = self.map_info.origin.position.y + gy_float * self.map_info.resolution + (self.map_info.resolution / 2.0)
            
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