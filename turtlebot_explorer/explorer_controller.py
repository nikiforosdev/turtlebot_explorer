#!/usr/bin/env python3
"""
Explorer Controller Node (Main Coordinator)
Subscribes to: /obstacle_detected, /reactive_cmd, /odom, /map, /frontiers
Publishes to: /cmd_vel (TwistStamped)

Coordinates between reactive and deliberative behaviors.
Implements the HYBRID ARCHITECTURE.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped, PointStamped
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry, OccupancyGrid
import numpy as np
import math
from nav_msgs.msg import Path


class ExplorerController(Node):
    """
    Main coordinator implementing hybrid architecture.
    
    Combines:
    - REACTIVE layer: Obstacle avoidance (high priority)
    - DELIBERATIVE layer: Goal-directed exploration (low priority)
    """
    
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
        self.current_path = []  # List of waypoints
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
        self.goal_timeout = 20.0  # Increased for path following
        self.goal_start_time = None
        self.min_goal_distance = 1.0
        
        # Parameters
        self.max_linear = 0.22
        self.max_angular = 2.84
        
        # Control loop
        self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('Explorer Controller with A* path planning started')
    
    def obstacle_callback(self, msg):
        """Update obstacle detection status."""
        self.obstacle_detected = msg.data
    
    def reactive_cmd_callback(self, msg):
        """Store reactive avoidance command."""
        self.reactive_cmd = msg
    
    def odom_callback(self, msg):
        """Update robot pose."""
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        
        # Convert quaternion to yaw
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)
    
    def map_callback(self, msg):
        """Update map."""
        self.map_info = msg.info
        self.map_data = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)
    
    def frontier_callback(self, msg):
        """
        Receive frontier points from frontier_detector node.
        We accumulate them and reset the list periodically.
        """
        current_time = self.get_clock().now()
        
        # If this is first frontier or timeout expired, reset list
        if self.last_frontier_time is None or \
           (current_time - self.last_frontier_time).nanoseconds / 1e9 > self.frontier_timeout:
            self.frontiers = []
        
        # Add new frontier
        self.frontiers.append((msg.point.x, msg.point.y))
        self.last_frontier_time = current_time
    
    def select_goal(self):
        """
        Select next exploration goal from received frontiers.
        Returns (x, y) or None.
        """
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
        
        # Select nearest frontier
        min_dist = float('inf')
        best = None
        
        for fx, fy in self.frontiers:
            dist = math.sqrt((fx - self.x)**2 + (fy - self.y)**2)
            if dist < min_dist and dist > 0.5:  # At least 0.5m away
                min_dist = dist
                best = (fx, fy)
        
        return best
    
    def compute_startup_command(self):
        """
        Generate exploratory movement when no frontiers exist yet.
        This helps build initial map at startup.
        """
        cmd = self.create_twist_stamped()
        
        # Initialize startup timer
        if self.startup_start_time is None:
            self.startup_start_time = self.get_clock().now()
        
        # Move forward slowly for initial exploration
        elapsed = (self.get_clock().now() - self.startup_start_time).nanoseconds / 1e9
        
        if elapsed < self.startup_move_duration:
            cmd.twist.linear.x = 0.1
            cmd.twist.angular.z = 0.0
            return cmd
        else:
            # After initial movement, disable startup mode
            self.startup_mode = False
            self.startup_start_time = None
            return self.create_twist_stamped()  # Stop
    
    def compute_random_exploration_command(self):
        """
        When stuck with no frontiers for a while, try random exploration.
        This helps escape local minima or find new areas.
        """
        cmd = self.create_twist_stamped()
        
        # Random walk: move forward and turn slightly
        import random
        cmd.twist.linear.x = 0.15
        cmd.twist.angular.z = random.uniform(-0.5, 0.5)
        
        return cmd
    
    def compute_navigation_command(self):
        """
        Compute command to navigate to current goal.
        DELIBERATIVE behavior.
        
        Returns:
            TwistStamped command or None if goal reached
        """
        if self.goal is None:
            return self.create_twist_stamped()
        
        # Calculate distance and angle to goal
        dx = self.goal[0] - self.x
        dy = self.goal[1] - self.y
        dist = math.sqrt(dx**2 + dy**2)
        
        # Check if reached
        if dist < self.goal_threshold:
            return None  # Signal goal reached
        
        # Calculate angle error
        desired_yaw = math.atan2(dy, dx)
        angle_error = desired_yaw - self.yaw
        
        # Normalize to [-pi, pi]
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
        
        # Proportional control
        cmd = self.create_twist_stamped()
        
        if abs(angle_error) > 0.2:  # Need to turn
            cmd.twist.linear.x = 0.05
            cmd.twist.angular.z = np.clip(2.0 * angle_error, -self.max_angular, self.max_angular)
        else:  # Move forward
            cmd.twist.linear.x = np.clip(0.5 * dist, 0.0, self.max_linear)
            cmd.twist.angular.z = np.clip(1.0 * angle_error, -0.5, 0.5)
        
        return cmd
    
    def create_twist_stamped(self):
        """Helper to create TwistStamped message with current timestamp."""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        return msg
    
    def compute_waypoint_navigation_command(self):
        """
        Navigate along the planned path by following waypoints.
        Returns TwistStamped command or None if waypoint reached.
        """
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
            
            # Check if this was the last waypoint
            if self.current_waypoint_index >= len(self.current_path):
                return None  # Path complete
            
            # Move to next waypoint
            return self.compute_waypoint_navigation_command()
        
        # Navigate to current waypoint
        desired_yaw = math.atan2(dy, dx)
        angle_error = desired_yaw - self.yaw
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
        
        cmd = self.create_twist_stamped()
        
        if abs(angle_error) > 0.2:
            # Turn toward waypoint
            cmd.twist.linear.x = 0.05
            cmd.twist.angular.z = np.clip(2.0 * angle_error, -self.max_angular, self.max_angular)
        else:
            # Move forward
            cmd.twist.linear.x = np.clip(0.5 * dist, 0.0, self.max_linear)
            cmd.twist.angular.z = np.clip(1.0 * angle_error, -0.5, 0.5)
        
        return cmd

    def control_loop(self):
        """
        MAIN HYBRID CONTROL LOOP with A* path following
        """
        # REACTIVE LAYER
        if self.obstacle_detected and self.reactive_cmd is not None:
            self.cmd_pub.publish(self.reactive_cmd)
            self.get_logger().info('REACTIVE MODE', throttle_duration_sec=1.0)
            return
        
        # STARTUP MODE
        if self.startup_mode:
            cmd = self.compute_startup_command()
            self.cmd_pub.publish(cmd)
            self.get_logger().info('STARTUP MODE', throttle_duration_sec=1.0)
            return
        
        # DELIBERATIVE LAYER - Path following
        
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
            
            if self.goal and self.map_data is not None: # Check for map data
                self.get_logger().info(f'New goal selected: ({self.goal[0]:.2f}, {self.goal[1]:.2f})')
                
                # --- CRITICAL FIX: Pass data to the Path Planner instance ---
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
                    self.get_logger().info(f'Path planned with {len(path)} waypoints')
                else:
                    self.get_logger().warn('Path planning failed! Trying different goal.')
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