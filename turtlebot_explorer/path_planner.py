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
        
        # Publish path for visualization (optional)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        
        self.map_data = None
        self.map_info = None
        self.current_x = 0.0
        self.current_y = 0.0
        self.cost_map = None
        
        # A* parameters - FIXED VALUES
        self.obstacle_cost_threshold = 20  # Lower threshold - treat cells > 20 as obstacles
        self.inflation_radius = 3  # Increased from 2 to 3 for more safety margin
        
        self.get_logger().info('Path Planner Node started with A* algorithm (FIXED VERSION)')
    
    def update_data(self, map_data, map_info, current_x, current_y):
        """Update map and pose data from the calling node."""
        self.map_info = map_info
        self.map_data = map_data
        self.current_x = current_x
        self.current_y = current_y
        
        # Re-create cost map every time data is updated
        if self.map_data is not None:
            self.cost_map = self.inflate_obstacles(self.map_data)
    
    def inflate_obstacles(self, grid):
        """
        Inflate obstacles for safety margin - IMPROVED VERSION
        Returns a cost map where high values = obstacles or near obstacles.
        """
        h, w = grid.shape
        cost_map = np.copy(grid).astype(np.float32)
        
        # Treat unknown cells (-1) as obstacles for path planning
        cost_map[cost_map == -1] = 100
        
        # Find obstacle cells (including unknown)
        obstacle_cells = np.argwhere(grid >= self.obstacle_cost_threshold)
        
        # Also add unknown cells as obstacles
        unknown_cells = np.argwhere(grid == -1)
        obstacle_cells = np.vstack([obstacle_cells, unknown_cells]) if len(unknown_cells) > 0 else obstacle_cells
        
        # Inflate each obstacle
        for obs_i, obs_j in obstacle_cells:
            for di in range(-self.inflation_radius, self.inflation_radius + 1):
                for dj in range(-self.inflation_radius, self.inflation_radius + 1):
                    ni, nj = obs_i + di, obs_j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        # Distance-based cost
                        dist = math.sqrt(di*di + dj*dj)
                        if dist <= self.inflation_radius:
                            # Progressive cost increase near obstacles
                            # Closer to obstacle = higher cost
                            inflation_cost = 100 * (1.0 - dist / self.inflation_radius)
                            cost_map[ni, nj] = max(cost_map[ni, nj], inflation_cost)
        
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
        """Check if grid cell is valid and not an obstacle - IMPROVED VERSION"""
        if self.cost_map is None:
            return False
        
        h, w = self.cost_map.shape
        
        # Check bounds
        if grid_x < 0 or grid_x >= w or grid_y < 0 or grid_y >= h:
            return False
        
        # Check cost - must be low enough to traverse
        # Using lower threshold for stricter obstacle avoidance
        map_val = self.cost_map[grid_y, grid_x]
        
        # Valid if cost is below threshold (free or low-cost inflated area)
        return map_val < self.obstacle_cost_threshold
    
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
                    base_cost = 1.414 if (dx != 0 and dy != 0) else 1.0
                    
                    # Add cost from cost_map - penalize high-cost areas more
                    map_cost = self.cost_map[ny, nx] / 10.0  # Scale cost appropriately
                    
                    total_cost = base_cost + map_cost
                    neighbors.append(((nx, ny), total_cost))
        
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
        Simplify path by removing unnecessary waypoints - IMPROVED VERSION
        """
        if len(path) < 3:
            return path
        
        smoothed = [path[0]]
        
        # Increased minimum distance for better path following
        MIN_DIST_SQUARED = 4**2  # Keep waypoints at least 4 cells apart (was 2)
        
        for i in range(1, len(path) - 1):
            curr = path[i]
            last_smoothed = smoothed[-1]
            
            dist_sq = (curr[0] - last_smoothed[0])**2 + (curr[1] - last_smoothed[1])**2
            
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
            self.get_logger().warn(f'Start position ({start_gx}, {start_gy}) is in obstacle! Cost: {self.cost_map[start_gy, start_gx]:.1f}')
            return None
        
        if not self.is_valid_cell(goal_gx, goal_gy):
            self.get_logger().warn(f'Goal position ({goal_gx}, {goal_gy}) is in obstacle! Cost: {self.cost_map[goal_gy, goal_gx]:.1f}')
            # Try to find nearest valid cell near goal
            goal_gx, goal_gy = self.find_nearest_valid_cell(goal_gx, goal_gy)
            if goal_gx is None:
                self.get_logger().warn('Could not find valid goal nearby')
                return None
            self.get_logger().info(f'Adjusted goal to nearest valid cell: ({goal_gx}, {goal_gy})')
        
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
        
        max_iterations = 10000  # Prevent infinite loops
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
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
                
                self.get_logger().info(f'Path found with {len(world_path)} waypoints after {iterations} iterations')
                
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
        
        self.get_logger().warn(f'No path found to goal after {iterations} iterations!')
        return None
    
    def find_nearest_valid_cell(self, goal_gx, goal_gy, search_radius=5):
        """Find the nearest valid cell to an invalid goal position."""
        best_cell = None
        min_dist = float('inf')
        
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                nx, ny = goal_gx + dx, goal_gy + dy
                
                if self.is_valid_cell(nx, ny):
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < min_dist:
                        min_dist = dist
                        best_cell = (nx, ny)
        
        return best_cell if best_cell else (None, None)
    
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
        rclpy.spin(node) 
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()