import pybullet as p
import numpy as np

def lidar_scan(robot_id, num_rays=16, max_dist=3.0, visualize=False):
    """
    Enhanced version with different colors for hits (red) and misses (green)
    """
    pos, ori = p.getBasePositionAndOrientation(robot_id)
    yaw = p.getEulerFromQuaternion(ori)[2]
    
    ray_height = pos[2] + 0.1
    angles = np.linspace(0, 2*np.pi, num_rays, endpoint=False)
    distances = []
    hits = []
    
    for a in angles:
        dx = np.cos(yaw + a)
        dy = np.sin(yaw + a)
        
        ray_from = [pos[0], pos[1], ray_height]
        ray_to = [
            pos[0] + dx * max_dist,
            pos[1] + dy * max_dist,
            ray_height
        ]
        
        hit = p.rayTest(ray_from, ray_to)[0]
        if hit[0] == -1 or hit[0] == robot_id:
            # Nothing hit
            dist = max_dist
            hits.append(False)
        else:
            # Hit wall/obstacle
            dist = hit[2] * max_dist
            hits.append(True)
        
        distances.append(dist)
        
        # Enhanced visualization
        if visualize:
            if hit[0] == -1:
                # No hit - green line to max distance
                line_color = [0, 1, 0]  # Green
                hit_pos = ray_to
            else:
                # Hit detected - red line to hit point
                line_color = [1, 0, 0]  # Red
                hit_pos = hit[3]  # Actual hit position
            
            p.addUserDebugLine(
                ray_from, 
                hit_pos, 
                line_color,
                lifeTime=0.1,
                lineWidth=2
            )
    
    return np.array(distances, dtype=np.float32), np.array(hits, dtype=bool)
