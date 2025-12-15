#!/usr/bin/env python3
"""
Reactive Controller Node
Subscribes to: /scan
Publishes to: /reactive_cmd (TwistStamped commands when avoiding obstacles)

REACTIVE layer: Fast obstacle avoidance using laser scanner
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool
import numpy as np


class ReactiveController(Node):
    """
    Reactive obstacle avoidance using laser scanner.
    
    REACTIVE component: Fast response to immediate obstacles.
    Runs at high frequency (~10 Hz) independent of map updates.
    """
    
    def __init__(self):
        super().__init__('reactive_controller')
        
        # Subscribe to laser scan
        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1)
        
        self.create_subscription(LaserScan, '/scan', self.laser_callback, qos)
        
        # Publish reactive commands and obstacle status - CHANGED TO TwistStamped
        self.cmd_pub = self.create_publisher(TwistStamped, '/reactive_cmd', 10)
        self.obstacle_pub = self.create_publisher(Bool, '/obstacle_detected', 10)
        
        self.laser_ranges = None
        
        # Parameters
        self.obstacle_threshold = 0.4  # meters
        
        # Timer for reactive control
        self.create_timer(0.1, self.reactive_control_loop)  # 10 Hz
        
        self.get_logger().info('Reactive Controller Node started')
    
    def laser_callback(self, msg):
        """Store laser scan data."""
        self.laser_ranges = np.array(msg.ranges)
        # Replace inf with max range
        self.laser_ranges[np.isinf(self.laser_ranges)] = 10.0
    
    def check_obstacle(self):
        """
        Check if obstacle is too close in front sector.
        
        Returns:
            True if obstacle detected within threshold
        """
        if self.laser_ranges is None:
            return False
        
        # Check front 60 degrees (±30°)
        n = len(self.laser_ranges)
        front_sector = n // 6  # Approximately 60 degrees
        
        front_left = self.laser_ranges[:front_sector]
        front_right = self.laser_ranges[-front_sector:]
        front_ranges = np.concatenate([front_left, front_right])
        
        min_distance = np.min(front_ranges)
        
        # DEBUG
        if min_distance < self.obstacle_threshold:
            self.get_logger().info(f'OBSTACLE! Min distance: {min_distance:.2f}m')
        
        return min_distance < self.obstacle_threshold
    
    def compute_avoidance_command(self):
        """
        Compute velocity command to avoid obstacle.
        
        Strategy: Turn towards more open space while moving slowly forward.
        
        Returns:
            TwistStamped command for obstacle avoidance
        """
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        
        if self.laser_ranges is None:
            return cmd
        
        # Analyze left vs right side
        n = len(self.laser_ranges)
        left_side = self.laser_ranges[:n//3]
        right_side = self.laser_ranges[-n//3:]
        
        avg_left = np.mean(left_side)
        avg_right = np.mean(right_side)
        
        # Slow forward motion
        cmd.twist.linear.x = 0.05
        
        # Turn towards more open space
        if avg_left > avg_right:
            cmd.twist.angular.z = 0.5  # Turn left
        else:
            cmd.twist.angular.z = -0.5  # Turn right
        
        return cmd
    
    def reactive_control_loop(self):
        """
        Main reactive control loop.
        Publishes obstacle status and avoidance commands if needed.
        """
        obstacle_detected = self.check_obstacle()
        
        # Publish obstacle status
        status_msg = Bool()
        status_msg.data = bool(obstacle_detected)
        self.obstacle_pub.publish(status_msg)
        
        # If obstacle detected, compute and publish avoidance command
        if obstacle_detected:
            cmd = self.compute_avoidance_command()
            self.cmd_pub.publish(cmd)
            self.get_logger().info('Obstacle detected - avoiding', 
                                 throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = ReactiveController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
