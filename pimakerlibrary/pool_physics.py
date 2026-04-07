import math
import numpy as np
import cv2

# Physics Utility Functions

def check_ball_collision(b1, b2):
    """Check if two balls are colliding and resolve."""
    # Vector from b1 to b2
    dx = b2.x - b1.x
    dy = b2.y - b1.y
    distance = math.hypot(dx, dy)
    
    # Check if overlapping
    min_dist = b1.radius + b2.radius
    if distance < min_dist:
        # Resolve overlap immediately to prevent getting stuck
        if distance == 0:
            # Prevent division by zero
            dx = 1
            dy = 0
            distance = 1
            
        overlap = min_dist - distance
        
        # Calculate normal vector
        nx = dx / distance
        ny = dy / distance
        
        # Move balls apart based on mass (they have same mass so 0.5 each)
        b1.x -= nx * (overlap * 0.5)
        b1.y -= ny * (overlap * 0.5)
        
        b2.x += nx * (overlap * 0.5)
        b2.y += ny * (overlap * 0.5)
        
        # Calculate relative velocity
        dvx = b2.vx - b1.vx
        dvy = b2.vy - b1.vy
        
        # Calculate velocity along the normal
        vel_along_normal = dvx * nx + dvy * ny
        
        # Only resolve if moving towards each other
        if vel_along_normal > 0:
            return
            
        # Restitution (bounciness) -> 0.85 is stable and realistic
        e = 0.85 
        
        # Calculate impulse scalar
        j = -(1 + e) * vel_along_normal
        # Mass is same for both so m1 + m2 = 2, so divide by 2
        j /= 2.0 
        
        # Apply impulse
        impulse_x = j * nx
        impulse_y = j * ny
        
        b1.vx -= impulse_x
        b1.vy -= impulse_y
        b2.vx += impulse_x
        b2.vy += impulse_y
        
