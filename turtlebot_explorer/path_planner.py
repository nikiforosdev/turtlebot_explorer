#!/usr/bin/env python3
"""
Path Planner Node
Subscribes to: /map, /odom
Provides: A* path planning functionality
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped
import numpy as np
import math
import heapq


class PathPlanner(Node):
    """
    Plans collision-free paths using A* algorithm.
    DELIBERATIVE component: Creates paths from current position to goal.
    """
    
    def __init__(self):
        super().__init__('path_planner')
        
        # Subscriptions are DISABLED because this instance is not spun as a standalone node.
        # Data will be passed directly from ExplorerController.
        
        # Publish path for visualization (optional)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        
        self.map_data = None
        self.map_info = None
        self.current_x = 0.0
        self.current_y = 0.0
        self.cost_map = None # Initialized here to be created later by inflate_obstacles
        
        # A* parameters
        self.obstacle_cost_threshold = 50  # Cells with value > 50 are obstacles
        self.inflation_radius = 2  # Grid cells to inflate obstacles (safety margin)
        
        self.get_logger().info('Path Planner Node started with A* algorithm (Data received externally)')
    
    def update_data(self, map_data, map_info, current_x, current_y):
        """Update map and pose data from the calling node."""
        self.map_info = map_info
        self.map_data = map_data
        self.current_x = current_x
        self.current_y = current_y
        
        # Re-create cost map every time data is updated
        if self.map_data is not None:
            self.cost_map = self.inflate_obstacles(self.map_data)
    
    # map_callback and odom_callback methods are no longer needed
    
    def inflate_obstacles(self, grid):
        """
        Inflate obstacles for safety margin.
        Returns a cost map where high values = obstacles or near obstacles.
        """
        h, w = grid.shape
        cost_map = np.copy(grid)
        
        # Find obstacle cells
        obstacle_cells = np.argwhere(grid > self.obstacle_cost_threshold)
        
        # Inflate each obstacle
        for obs_i, obs_j in obstacle_cells:
            for di in range(-self.inflation_radius, self.inflation_radius + 1):
                for dj in range(-self.inflation_radius, self.inflation_radius + 1):
                    ni, nj = obs_i + di, obs_j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        # Distance-based cost
                        dist = math.sqrt(di*di + dj*dj)
                        if dist <= self.inflation_radius:
                            # Increase cost near obstacles
                            cost_map[ni, nj] = max(cost_map[ni, nj], 80)
        
        return cost_map
    
    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates."""
        if self.map_info is None:
            return None, None
        
        grid_x = int((x - self.map_info.origin.position.x) / self.map_info.resolution)
        grid_y = int((y - self.map_info.origin.position.y) / self.map_info.resolution)
        
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates."""
        if self.map_info is None:
            return None, None
        
        x = self.map_info.origin.position.x + grid_x * self.map_info.resolution + (self.map_info.resolution / 2.0)
        y = self.map_info.origin.position.y + grid_y * self.map_info.resolution + (self.map_info.resolution / 2.0)
        
        return x, y
    
    def is_valid_cell(self, grid_x, grid_y):
        """Check if grid cell is valid and not an obstacle."""
        if self.cost_map is None:
            return False
        
        h, w = self.cost_map.shape
        
        # Check bounds
        if grid_x < 0 or grid_x >= w or grid_y < 0 or grid_y >= h:
            return False
        
        # Check if not obstacle (allow some cost for inflated areas)
        # Note: We must check for unknown/free (-1, 0) AND inflated cost (< obstacle_cost_threshold)
        map_val = self.cost_map[grid_y, grid_x]
        
        # If it's unknown (-1) or too costly (>= 50), it's invalid for path planning
        return map_val != -1 and map_val < self.obstacle_cost_threshold
    
    def heuristic(self, a, b):
        """Euclidean distance heuristic for A*."""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_neighbors(self, cell):
        """Get valid 8-connected neighbors of a cell."""
        x, y = cell
        neighbors = []
        
        # 8-connected neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                nx, ny = x + dx, y + dy
                
                if self.is_valid_cell(nx, ny):
                    # Cost is higher for diagonal moves
                    cost = 1.414 if (dx != 0 and dy != 0) else 1.0
                    
                    # Add cost from cost_map to traverse through this cell
                    # Normalize map cost (0-100) to a path penalty (e.g., 0-5)
                    map_cost = self.cost_map[ny, nx] / 20.0 
                    
                    neighbors.append(((nx, ny), cost + map_cost))
        
        return neighbors
    
    def reconstruct_path(self, came_from, start, goal):
        """Reconstruct path from A* came_from dict."""
        path = []
        current = goal
        
        while current != start:
            path.append(current)
            current = came_from.get(current)
            if current is None:
                return None
        
        path.append(start)
        path.reverse()
        
        return path
    
    def smooth_path(self, path):
        """
        Simplify path by removing unnecessary waypoints.
        Keep only waypoints where direction changes significantly.
        """
        if len(path) < 3:
            return path
        
        smoothed = [path[0]]
        
        # Use a distance-based sampling to reduce the path complexity
        MIN_DIST_SQUARED = 2**2 # Keep waypoints at least 2 cells apart
        
        for i in range(1, len(path) - 1):
            curr = path[i]
            last_smoothed = smoothed[-1]
            
            dist_sq = (curr[0] - last_smoothed[0])**2 + (curr[1] - last_smoothed[1])**2
            
            # Simple version: Only keep points that are significantly far away
            if dist_sq >= MIN_DIST_SQUARED:
                smoothed.append(curr)
                
        # Always add the final goal
        if path[-1] != smoothed[-1]:
            smoothed.append(path[-1])
        
        return smoothed
    
    def plan_path(self, goal_x, goal_y):
        """
        Plan a path using A* algorithm.
        
        Args:
            goal_x: Goal x coordinate in world frame
            goal_y: Goal y coordinate in world frame
        
        Returns:
            List of waypoints [(x1,y1), (x2,y2), ...] in world coordinates
            or None if no path found
        """
        if self.map_data is None or self.cost_map is None:
            self.get_logger().warn('No map data available for planning')
            return None
        
        # Convert to grid coordinates
        start_gx, start_gy = self.world_to_grid(self.current_x, self.current_y)
        goal_gx, goal_gy = self.world_to_grid(goal_x, goal_y)
        
        if start_gx is None or goal_gx is None:
            self.get_logger().warn('Invalid coordinates for planning')
            return None
        
        # Check if start and goal are valid
        if not self.is_valid_cell(start_gx, start_gy):
            self.get_logger().warn(f'Start position ({start_gx}, {start_gy}) is in obstacle! Map value: {self.cost_map[start_gy, start_gx]}')
            return None
        
        if not self.is_valid_cell(goal_gx, goal_gy):
            self.get_logger().warn(f'Goal position ({goal_gx}, {goal_gy}) is in obstacle! Map value: {self.cost_map[goal_gy, goal_gx]}')
            return None
        
        # A* algorithm
        start = (start_gx, start_gy)
        goal = (goal_gx, goal_gy)
        
        # Priority queue: (f_score, counter, node)
        counter = 0
        open_set = [(0, counter, start)]
        came_from = {}
        
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        closed_set = set()
        
        while open_set:
            current_f, _, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            # Goal reached
            if current == goal:
                grid_path = self.reconstruct_path(came_from, start, goal)
                
                if grid_path is None:
                    return None
                
                # Smooth the path
                grid_path = self.smooth_path(grid_path)
                
                # Convert to world coordinates
                world_path = []
                for gx, gy in grid_path:
                    wx, wy = self.grid_to_world(gx, gy)
                    world_path.append((wx, wy))
                
                self.get_logger().info(f'Path found with {len(world_path)} waypoints')
                
                # Publish path for visualization
                self.publish_path(world_path)
                
                return world_path
            
            closed_set.add(current)
            
            # Check neighbors
            for neighbor, move_cost in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    
                    counter += 1
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
        
        self.get_logger().warn('No path found to goal!')
        return None
    
    def publish_path(self, waypoints):
        """Publish path for visualization in RViz."""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        
        for wx, wy in waypoints:
            pose = PoseStamped()
            pose.header.stamp = path_msg.header.stamp
            pose.header.frame_id = 'map'
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PathPlanner()
    
    try:
        # Note: This file should be launched as a library, but if run standalone, 
        # it will just sit and wait for data.
        rclpy.spin(node) 
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()