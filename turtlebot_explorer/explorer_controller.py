#!/usr/bin/env python3
"""
Explorer Controller Node (Main Coordinator)
Subscribes to: /obstacle_detected, /reactive_cmd, /odom, /map
Publishes to: /cmd_vel (TwistStamped)

Coordinates between reactive and deliberative behaviors.
Implements the HYBRID ARCHITECTURE.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry, OccupancyGrid
import numpy as np
import math


class ExplorerController(Node):
    """
    Main coordinator implementing hybrid architecture.
    
    Combines:
    - REACTIVE layer: Obstacle avoidance (high priority)
    - DELIBERATIVE layer: Goal-directed exploration (low priority)
    """
    
    def __init__(self):
        super().__init__('explorer_controller')
        
        # Publishers - CHANGED TO TwistStamped
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        
        # Subscribers
        self.create_subscription(Bool, '/obstacle_detected', self.obstacle_callback, 10)
        self.create_subscription(TwistStamped, '/reactive_cmd', self.reactive_cmd_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        
        # State variables
        self.obstacle_detected = False
        self.reactive_cmd = None
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.map_data = None
        self.map_info = None
        self.goal = None
        
        # Parameters
        self.goal_threshold = 0.3  # meters
        self.max_linear = 0.22  # m/s
        self.max_angular = 2.84  # rad/s
        
        # Control loop
        self.create_timer(0.1, self.control_loop)  # 10 Hz
        
        self.get_logger().info('Explorer Controller Node started')
    
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
    
    def find_frontiers(self):
        """
        Detect frontier cells.
        Returns list of (x, y) in world coordinates.
        """
        if self.map_data is None:
            return []
        
        h, w = self.map_data.shape
        frontiers = []
        
        # Sample grid for frontiers
        for i in range(5, h-5, 5):
            for j in range(5, w-5, 5):
                if self.map_data[i, j] == 0:  # Free cell
                    # Check neighbors
                    neighbors = [
                        self.map_data[i-1, j],
                        self.map_data[i+1, j],
                        self.map_data[i, j-1],
                        self.map_data[i, j+1]
                    ]
                    
                    if -1 in neighbors:  # Adjacent to unknown
                        # Convert to world coords
                        wx = self.map_info.origin.position.x + j * self.map_info.resolution
                        wy = self.map_info.origin.position.y + i * self.map_info.resolution
                        frontiers.append((wx, wy))
        
        return frontiers
    
    def select_goal(self):
        """
        Select next exploration goal.
        Returns (x, y) or None.
        """
        frontiers = self.find_frontiers()
        
        if not frontiers:
            return None
        
        # Select nearest frontier
        min_dist = float('inf')
        best = None
        
        for fx, fy in frontiers:
            dist = math.sqrt((fx - self.x)**2 + (fy - self.y)**2)
            if dist < min_dist and dist > 0.5:  # At least 0.5m away
                min_dist = dist
                best = (fx, fy)
        
        return best
    
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
    
    def control_loop(self):
        """
        MAIN HYBRID CONTROL LOOP
        
        Priority hierarchy:
        1. REACTIVE: If obstacle detected, use reactive command
        2. DELIBERATIVE: Otherwise, navigate to exploration goal
        """
        
        # REACTIVE LAYER (Highest Priority)
        if self.obstacle_detected and self.reactive_cmd is not None:
            self.cmd_pub.publish(self.reactive_cmd)
            self.get_logger().info('REACTIVE MODE', throttle_duration_sec=1.0)
            return
        
        # DELIBERATIVE LAYER
        
        # Need new goal?
        if self.goal is None:
            self.goal = self.select_goal()
            
            if self.goal:
                self.get_logger().info(f'DELIBERATIVE: New goal ({self.goal[0]:.2f}, {self.goal[1]:.2f})')
            else:
                self.get_logger().info('No frontiers - exploration complete?', 
                                     throttle_duration_sec=2.0)
                self.cmd_pub.publish(self.create_twist_stamped())  # Stop
                return
        
        # Navigate to goal
        cmd = self.compute_navigation_command()
        
        if cmd is None:  # Goal reached
            self.get_logger().info('Goal reached!')
            self.goal = None
            cmd = self.create_twist_stamped()
        
        self.cmd_pub.publish(cmd)


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
