import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped, PointStamped
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry, OccupancyGrid
import numpy as np
import math
from nav_msgs.msg import Path


class ExplorerController(Node):
    #COORDINATES REACTIVE DELIVERATIVE
    
    def __init__(self):
        super().__init__('explorer_controller')
        
        # Publishers
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        
        # Subscribers
        self.create_subscription(Bool, '/obstacle_detected', self.obstacle_callback, 10)
        self.create_subscription(TwistStamped, '/reactive_cmd', self.reactive_cmd_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        map_qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)
        self.create_subscription(PointStamped, '/frontiers', self.frontier_callback, 10)
        
        # State variables
        self.obstacle_detected = False
        self.reactive_cmd = None
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.map_data = None
        self.map_info = None
        self.goal = None
        
        # Import path planner
        from turtlebot_explorer.path_planner import PathPlanner
        self.path_planner = PathPlanner()
        
        # Path following
        self.current_path = []
        self.current_waypoint_index = 0
        self.waypoint_threshold = 0.3  # meters
        
        # Store frontiers
        self.frontiers = []
        self.frontier_timeout = 2.0
        self.last_frontier_time = None
        
        # Startup exploration mode
        self.startup_mode = True
        self.startup_move_duration = 5.0
        self.startup_start_time = None
        self.no_frontier_counter = 0
        self.no_frontier_threshold = 10
        
        # Goal parameters
        self.goal_threshold = 0.3
        self.goal_timeout = 30.0
        self.goal_start_time = None
        self.min_goal_distance = 1.0
        
        self.max_linear = 0.15 
        self.max_angular = 2.0
        
        # Control loop
        self.create_timer(0.1, self.control_loop)
    
    def obstacle_callback(self, msg):
        self.obstacle_detected = msg.data
    
    def reactive_cmd_callback(self, msg):
        self.reactive_cmd = msg
    
    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        
        # here we convert quaternion to yaw
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)
    
    def map_callback(self, msg):
        self.map_info = msg.info
        self.map_data = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)
    
    def frontier_callback(self, msg):
        """
        Receive frontier points from frontier_detector node.
        """
        current_time = self.get_clock().now()
        
        # If this is first frontier or timeout expired, reset list
        if self.last_frontier_time is None or \
           (current_time - self.last_frontier_time).nanoseconds / 1e9 > self.frontier_timeout:
            self.frontiers = []
        
        # Add new frontier
        self.frontiers.append((msg.point.x, msg.point.y))
        self.last_frontier_time = current_time
    
    def is_frontier_safe(self, fx, fy):
        """
        Check if a frontier point is safe to navigate to.
        Returns True if the frontier is in free space with some margin from obstacles.
        """
        if self.map_data is None or self.map_info is None:
            return False
        
        # Convert to grid coordinates
        grid_x = int((fx - self.map_info.origin.position.x) / self.map_info.resolution)
        grid_y = int((fy - self.map_info.origin.position.y) / self.map_info.resolution)
        
        h, w = self.map_data.shape
        
        # Check if within bounds
        if grid_x < 0 or grid_x >= w or grid_y < 0 or grid_y >= h:
            return False
        
        # Check the frontier cell and surrounding cells
        safety_radius = 2  # Check 2 cells around the frontier
        
        for di in range(-safety_radius, safety_radius + 1):
            for dj in range(-safety_radius, safety_radius + 1):
                ni, nj = grid_y + di, grid_x + dj
                
                if 0 <= ni < h and 0 <= nj < w:
                    cell_value = self.map_data[ni, nj]
                    # If any nearby cell is an obstacle (> 20) or unknown (-1), reject
                    if cell_value > 20 or cell_value == -1:
                        return False
        
        return True
    
    def select_goal(self):
        #Returns (x, y) or None.
        
        # Check if we have recent frontiers
        if not self.frontiers:
            return None
        
        # Check if frontier data is stale
        if self.last_frontier_time is not None:
            current_time = self.get_clock().now()
            age = (current_time - self.last_frontier_time).nanoseconds / 1e9
            if age > self.frontier_timeout:
                self.get_logger().warn('Frontier data is stale')
                return None
        
        # Filter safe frontiers
        safe_frontiers = []
        for fx, fy in self.frontiers:
            dist = math.sqrt((fx - self.x)**2 + (fy - self.y)**2)
            
            # Check distance and safety
            if dist > 1.0 and dist < 5.0 and self.is_frontier_safe(fx, fy):
                safe_frontiers.append((fx, fy, dist))
        
        if not safe_frontiers:
            self.get_logger().warn('No safe frontiers found!')
            return None
        
        # Select nearest safe frontier
        safe_frontiers.sort(key=lambda x: x[2])  # Sort by distance
        best = safe_frontiers[0]
        
        self.get_logger().info(f'Selected safe frontier at distance {best[2]:.2f}m')
        return (best[0], best[1])
    
    def compute_startup_command(self):
        cmd = self.create_twist_stamped()
        
        if self.startup_start_time is None:
            self.startup_start_time = self.get_clock().now()
        
        elapsed = (self.get_clock().now() - self.startup_start_time).nanoseconds / 1e9
        
        if elapsed < self.startup_move_duration:
            cmd.twist.linear.x = 0.1
            cmd.twist.angular.z = 0.0
            return cmd
        else:
            self.startup_mode = False
            self.startup_start_time = None
            return self.create_twist_stamped()
    
    def compute_random_exploration_command(self):
        # NO FRONTIERS = EXPLORATION
        cmd = self.create_twist_stamped()
        
        import random
        cmd.twist.linear.x = 0.1  # Slower for safety
        cmd.twist.angular.z = random.uniform(-0.5, 0.5)
        
        return cmd
    
    def create_twist_stamped(self):
        """Helper to create TwistStamped message with current timestamp."""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        return msg
    
    def compute_waypoint_navigation_command(self):
        if not self.current_path or self.current_waypoint_index >= len(self.current_path):
            return None
        
        # Get current target waypoint
        target_x, target_y = self.current_path[self.current_waypoint_index]
        
        # Calculate distance to waypoint
        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.sqrt(dx**2 + dy**2)
        
        # Check if waypoint reached
        if dist < self.waypoint_threshold:
            self.current_waypoint_index += 1
            self.get_logger().info(f'Waypoint {self.current_waypoint_index}/{len(self.current_path)} reached')
            
            if self.current_waypoint_index >= len(self.current_path):
                return None
            
            return self.compute_waypoint_navigation_command()
        
        # Navigate to current waypoint
        desired_yaw = math.atan2(dy, dx)
        angle_error = desired_yaw - self.yaw
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
        
        cmd = self.create_twist_stamped()
        
        if abs(angle_error) > 0.3:  # Increased threshold for turning
            # Turn in place
            cmd.twist.linear.x = 0.0  # Stop while turning
            cmd.twist.angular.z = np.clip(1.5 * angle_error, -self.max_angular, self.max_angular)
        else:
            # Move forward with reduced speed
            speed = min(0.3 * dist, self.max_linear)
            cmd.twist.linear.x = speed
            cmd.twist.angular.z = np.clip(1.0 * angle_error, -0.3, 0.3)
        
        return cmd

    def control_loop(self):
        """
        MAIN HYBRID CONTROL LOOP
        """
        # REACTIVE LAYER - Always has priority
        if self.obstacle_detected and self.reactive_cmd is not None:
            self.cmd_pub.publish(self.reactive_cmd)
            self.get_logger().info('REACTIVE MODE: Avoiding obstacle', throttle_duration_sec=1.0)
            
            # If obstacle detected while following path, consider replanning
            if self.current_path:
                self.get_logger().warn('Obstacle detected during path following!')
            
            return
        
        # STARTUP MODE
        if self.startup_mode:
            cmd = self.compute_startup_command()
            self.cmd_pub.publish(cmd)
            self.get_logger().info('STARTUP MODE', throttle_duration_sec=1.0)
            return
        
        # DELIBERATIVE LAYER
        
        # Check if we have a valid path to follow
        if self.current_path and self.current_waypoint_index < len(self.current_path):
            # Check timeout
            if self.goal_start_time is not None:
                elapsed = (self.get_clock().now() - self.goal_start_time).nanoseconds / 1e9
                if elapsed > self.goal_timeout:
                    self.get_logger().info('Path timeout - replanning')
                    self.current_path = []
                    self.goal = None
                    self.goal_start_time = None
            
            # Follow the path
            cmd = self.compute_waypoint_navigation_command()
            
            if cmd is None:
                # Path complete
                self.get_logger().info('Path complete! Goal reached.')
                self.current_path = []
                self.current_waypoint_index = 0
                self.goal = None
                self.goal_start_time = None
                cmd = self.create_twist_stamped()
            
            self.cmd_pub.publish(cmd)
            return
        
        # Need new goal and path
        if self.goal is None:
            self.goal = self.select_goal()
            
            if self.goal and self.map_data is not None:
                self.get_logger().info(f'New goal selected: ({self.goal[0]:.2f}, {self.goal[1]:.2f})')
                
                # Update path planner data
                self.path_planner.update_data(
                    self.map_data, 
                    self.map_info, 
                    self.x, 
                    self.y
                )
                
                # Plan path using A*
                path = self.path_planner.plan_path(self.goal[0], self.goal[1])
                
                if path and len(path) > 0:
                    self.current_path = path
                    self.current_waypoint_index = 0
                    self.goal_start_time = self.get_clock().now()
                    self.no_frontier_counter = 0
                    self.get_logger().info(f'YES, Path planned with {len(path)} waypoints')
                else:
                    self.get_logger().warn('NO, Path planning failed! Trying different goal.')
                    self.goal = None
            else:
                # No frontiers
                self.no_frontier_counter += 1
                
                if self.no_frontier_counter > self.no_frontier_threshold:
                    cmd = self.compute_random_exploration_command()
                    self.cmd_pub.publish(cmd)
                    self.get_logger().info('RANDOM EXPLORATION', throttle_duration_sec=2.0)
                    if self.no_frontier_counter > self.no_frontier_threshold + 20:
                        self.no_frontier_counter = 0
                    return
                else:
                    self.get_logger().info('No frontiers - waiting...', throttle_duration_sec=2.0)
                    self.cmd_pub.publish(self.create_twist_stamped())
                    return


def main(args=None):
    rclpy.init(args=args)
    node = ExplorerController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()