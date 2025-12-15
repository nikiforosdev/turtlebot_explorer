#!/usr/bin/env python3
"""
Path Planner Node
Subscribes to: /map, /odom
Provides: Path planning service/functionality

For minimal implementation: Direct line-of-sight planning
Can be upgraded to A* later if needed.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np
import math


class PathPlanner(Node):
    """
    Plans paths from current position to goal.
    
    DELIBERATIVE component: Creates collision-free paths.
    
    For minimal version: Simple direct navigation (can upgrade to A* later)
    """
    
    def __init__(self):
        super().__init__('path_planner')
        
        # Subscribe to map and odometry
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        self.map_data = None
        self.map_info = None
        self.current_x = 0.0
        self.current_y = 0.0
        
        self.get_logger().info('Path Planner Node started')
    
    def map_callback(self, msg):
        """Store the current map."""
        self.map_info = msg.info
        self.map_data = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)
    
    def odom_callback(self, msg):
        """Store current position."""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
    
    def plan_path(self, goal_x, goal_y):
        """
        Plan a path to the goal.
        
        MINIMAL VERSION: Returns goal directly (straight line)
        UPGRADE: Implement A* algorithm here
        
        Args:
            goal_x: Goal x coordinate
            goal_y: Goal y coordinate
        
        Returns:
            List of waypoints [(x1,y1), (x2,y2), ...] or None if unreachable
        """
        if self.map_data is None:
            return None
        
        # For minimal version: just return the goal
        # The reactive layer will handle obstacles
        return [(goal_x, goal_y)]
        
        # TODO: Implement A* here if needed
        # See the A* algorithm in the previous artifact for full implementation
    
    def is_valid_goal(self, x, y):
        """
        Check if a goal position is in free space.
        
        Args:
            x: World x coordinate
            y: World y coordinate
        
        Returns:
            True if goal is valid (in free space)
        """
        if self.map_data is None or self.map_info is None:
            return False
        
        # Convert to grid coordinates
        grid_j = int((x - self.map_info.origin.position.x) / self.map_info.resolution)
        grid_i = int((y - self.map_info.origin.position.y) / self.map_info.resolution)
        
        # Check bounds
        h, w = self.map_data.shape
        if grid_i < 0 or grid_i >= h or grid_j < 0 or grid_j >= w:
            return False
        
        # Check if free or unknown (allow unknown for exploration)
        return self.map_data[grid_i, grid_j] == 0 or self.map_data[grid_i, grid_j] == -1


def main(args=None):
    rclpy.init(args=args)
    node = PathPlanner()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()