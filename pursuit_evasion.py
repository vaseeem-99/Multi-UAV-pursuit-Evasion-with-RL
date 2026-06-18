"""
============================================================
 IMPROVED PURSUIT-EVASION (WITH QUICK WINS FIXES)
============================================================
"""

import pybullet as p
import pybullet_data
import numpy as np
from gymnasium import spaces
import os


class PursuitEvasionImproved:

    def __init__(
        self,
        gui=False,
        n_pursuers=2,
        arena_size=14.0,
        max_steps=600,
        pursuer_max_speed=3.0,
        evader_speed=0.03,    
        capture_radius=2.5,          
        control_type="velocity",
        use_real_drone=True,
    ):

        self.gui = gui
        if gui:
            self.cid = p.connect(p.GUI)
        else:
            self.cid = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240)

        self.n_pursuers = n_pursuers
        self.arena_size = arena_size
        self.max_steps = max_steps
        self.capture_radius = capture_radius
        self.evader_speed = evader_speed
        self.pursuer_max_speed = pursuer_max_speed
        self.control_type = control_type
        self.use_real_drone = use_real_drone

        self._load_world()
        self._load_drones()
        
        self.obstacle_ids = []

        # observation space 
        self.obs_dim = 28 
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_pursuers, self.obs_dim),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(n_pursuers, 3),
            dtype=np.float32
        )

        self._episode_reset_members()

        print(f"\n=== IMPROVED MAPPO ENVIRONMENT (Quick Wins Applied) ===")
        print(f"Evader speed: {evader_speed:.3f} | Capture radius: {capture_radius:.1f}")

    def _load_world(self):
        self.plane = p.loadURDF("plane.urdf")

        self.wall_ids = []
        thickness = 0.5
        height = 6
        L = self.arena_size/2

        walls = [
            ([ L, 0, height/2], [thickness, L, height/2]),
            ([-L, 0, height/2], [thickness, L, height/2]),
            ([0,  L, height/2], [L, thickness, height/2]),
            ([0, -L, height/2], [L, thickness, height/2]),
        ]

        for pos, half in walls:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half)
            w = p.createMultiBody(0, col, vis, pos)
            self.wall_ids.append(w)

    def _load_drones(self):
        drone_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../gym-pybullet-drones/gym_pybullet_drones/assets/cf2p.urdf"))   
        if not os.path.exists(drone_path):
            print("Real drone not found, instead using spheres.")
            self.use_real_drone = False

        # BLUE PURSUERS
        self.pursuer_ids = []
        for i in range(self.n_pursuers):
            if self.use_real_drone:
                pid = p.loadURDF(drone_path, [0,0,3], globalScaling=12.0)
                p.changeVisualShape(pid, -1, rgbaColor=[0, 0.3, 1, 1])
            else:
                col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
                vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[0, 0.3, 1, 1])
                pid = p.createMultiBody(0, col, vis, [0,0,3])
            self.pursuer_ids.append(pid)

        # RED EVADER
        if self.use_real_drone:
            self.evader_id = p.loadURDF(drone_path, [2,2,3], globalScaling=12.0)
            p.changeVisualShape(self.evader_id, -1, rgbaColor=[1, 0, 0, 1])
        else:
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[1, 0, 0, 1])
            self.evader_id = p.createMultiBody(0, col, vis, [2,2,3])

    def _clear_obstacles(self):
        for obs_id in self.obstacle_ids:
            p.removeBody(obs_id)
        self.obstacle_ids.clear()
    
    def _create_obstacles(self, obstacles):
        self._clear_obstacles()
        
        for obs in obstacles:
            if obs['type'] == 'cylinder':
                pos = obs['position']
                radius, height = obs['size']
                color = obs.get('color', [0.5, 0.5, 0.5, 1.0])
                
                col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
                vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
                obs_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                          baseVisualShapeIndex=vis, basePosition=pos)
                self.obstacle_ids.append(obs_id)

    def _episode_reset_members(self):
        self.step_count = 0
        self.captured = False
        self.min_distance = 9999
        self._clear_obstacles()

    def reset(self, scenario=None):
        self._episode_reset_members()

        if scenario is None:
            for i, pid in enumerate(self.pursuer_ids):
                x = np.random.uniform(-self.arena_size/2+2, -2)
                y = np.random.uniform(-4, 4)
                p.resetBasePositionAndOrientation(pid, [x,y,3], [0,0,0,1])

            ex = np.random.uniform(2, self.arena_size/2-2)
            ey = np.random.uniform(-4, 4)
            p.resetBasePositionAndOrientation(self.evader_id, [ex,ey,3], [0,0,0,1])
            self._clear_obstacles()
        else:
            for pos, pid in zip(scenario["pursuer_positions"], self.pursuer_ids):
                p.resetBasePositionAndOrientation(pid, pos, [0,0,0,1])
            p.resetBasePositionAndOrientation(self.evader_id, scenario["evader_position"], [0,0,0,1])
            
            if "obstacles" in scenario:
                self._create_obstacles(scenario["obstacles"])

        for pid in self.pursuer_ids:
            p.resetBaseVelocity(pid, [0,0,0], [0,0,0])
        p.resetBaseVelocity(self.evader_id, [0,0,0], [0,0,0])

        return self._get_obs()

    def _get_obs(self):
        obs_all = []
        ev_pos, _ = p.getBasePositionAndOrientation(self.evader_id)
        ev_vel, _ = p.getBaseVelocity(self.evader_id)  # NEW
        ev_pos = np.array(ev_pos)
        ev_vel = np.array(ev_vel)

        purs_pos = []
        purs_vel = []

        for pid in self.pursuer_ids:
            pos, _ = p.getBasePositionAndOrientation(pid)
            vel, _ = p.getBaseVelocity(pid)
            purs_pos.append(np.array(pos))
            purs_vel.append(np.array(vel))

        for i in range(self.n_pursuers):
            o = np.zeros(self.obs_dim)

            # Self
            o[0:3] = purs_pos[i]
            o[3:6] = purs_vel[i]
            o[6:9] = ev_pos - purs_pos[i]
            o[9] = np.linalg.norm(ev_pos - purs_pos[i])

            # Other pursuers
            idx = 10
            for j in range(self.n_pursuers):
                if j == i: 
                    continue
                o[idx:idx+3] = purs_pos[j] - purs_pos[i]
                idx += 3

            # Nearest obstacles
            if len(self.obstacle_ids) > 0:
                obs_dists = []
                for obs_id in self.obstacle_ids:
                    obs_pos, _ = p.getBasePositionAndOrientation(obs_id)
                    obs_pos = np.array(obs_pos)
                    dist = np.linalg.norm(purs_pos[i] - obs_pos)
                    obs_dists.append((dist, obs_pos))
                
                obs_dists.sort(key=lambda x: x[0])
                
                for k in range(min(3, len(obs_dists))):
                    o[16 + k*3:16 + k*3 + 3] = obs_dists[k][1] - purs_pos[i]

            # Evader velocity
            o[25:28] = ev_vel

            obs_all.append(o)

        return np.array(obs_all, dtype=np.float32)

    def _too_close_to_wall(self, pos):
        L = self.arena_size/2 - 0.5
        x, y, z = pos
        return abs(x) > L or abs(y) > L

    def _inter_pursuer_collision(self, p1, p2):
        return np.linalg.norm(p1 - p2) < 0.7
    
    def _check_obstacle_collision(self, pos, radius=0.5):
        for obs_id in self.obstacle_ids:
            obs_pos, _ = p.getBasePositionAndOrientation(obs_id)
            obs_pos = np.array(obs_pos)
            dist = np.linalg.norm(pos[:2] - obs_pos[:2])
            if dist < (0.1 + radius):
                return True
        return False

    def step(self, actions):
        dt = 1/30.0
        actions = np.clip(actions, -1, 1)
        actions = actions * self.pursuer_max_speed

        # Move pursuers
        for i, pid in enumerate(self.pursuer_ids):
            pos, orn = p.getBasePositionAndOrientation(pid)
            pos = np.array(pos)
            new = pos + actions[i] * dt
            new = np.clip(new, [-self.arena_size/2+1, -self.arena_size/2+1, 1],
                                [ self.arena_size/2-1,  self.arena_size/2-1, 5])
            p.resetBasePositionAndOrientation(pid, new, orn)

        # EVADER MOVEMENT
        ev_pos, ev_orn = p.getBasePositionAndOrientation(self.evader_id)
        ev_pos = np.array(ev_pos)

        direction = np.zeros(3)
        total_weight = 0
        
        for pid in self.pursuer_ids:
            pos, _ = p.getBasePositionAndOrientation(pid)
            diff = ev_pos - np.array(pos)
            dist = np.linalg.norm(diff)
            
            if dist > 0.1:
                weight = 1.0 / (dist ** 2)
                direction += (diff / dist) * weight
                total_weight += weight
        
        if total_weight > 0:
            direction = direction / total_weight
            if np.linalg.norm(direction) > 1e-6:
                direction = direction / np.linalg.norm(direction)
            
            ev_new = ev_pos + direction * self.evader_speed * dt * 0.25
        else:
            ev_new = ev_pos

        ev_new = np.clip(ev_new, 
                         [-self.arena_size/2+1, -self.arena_size/2+1, 1],
                         [self.arena_size/2-1, self.arena_size/2-1, 5])

        p.resetBasePositionAndOrientation(self.evader_id, ev_new, ev_orn)
        p.stepSimulation()

        # IMPROVED REWARD SHAPING
        # =====================================================
        ev = ev_new
        purs = []
        dists = []

        for pid in self.pursuer_ids:
            pos, _ = p.getBasePositionAndOrientation(pid)
            purs.append(np.array(pos))
            d = np.linalg.norm(ev - purs[-1])
            dists.append(d)

        min_d = min(dists)
        avg_d = np.mean(dists)
        
        prev_min = self.min_distance
        self.min_distance = min(min_d, self.min_distance)

        reward = np.zeros(self.n_pursuers)

        # 1.distance-based reward
        reward -= min_d * 1.5
        reward -= avg_d * 0.3

        # 2.PROXIMITY STAGED REWARDS
        if min_d < 4.0:
            reward += 2.0
        if min_d < 3.0:
            reward += 5.0
        if min_d < 2.0:
            reward += 10.0
        
        # 3.Improvement bonus
        if min_d < prev_min:
            reward += 8.0 

        # 4.SURROUNDING BONUS
        if len(purs) >= 2:
            # Calculate angles between pursuers relative to evader
            purs_vecs = [purs[i] - ev for i in range(len(purs))]
            for i in range(len(purs_vecs)):
                for j in range(i+1, len(purs_vecs)):
                    norm_i = np.linalg.norm(purs_vecs[i])
                    norm_j = np.linalg.norm(purs_vecs[j])
                    if norm_i > 1e-6 and norm_j > 1e-6:
                        angle = np.dot(purs_vecs[i], purs_vecs[j]) / (norm_i * norm_j)
                        # Reward if on opposite sides (angle < -0.5 means >120 degrees)
                        if angle < -0.3:
                            reward += 5.0

        # Collision penalties
        collision_count = 0
        for i in range(len(purs)):
            for j in range(i+1, len(purs)):
                if self._inter_pursuer_collision(purs[i], purs[j]):
                    collision_count += 1
        
        obstacle_collision_count = 0
        for p_pos in purs:
            if self._check_obstacle_collision(p_pos):
                obstacle_collision_count += 1

        reward -= collision_count * 50
        reward -= obstacle_collision_count * 30

        # Capture bonus
        captured = min_d < self.capture_radius
        if captured:
            time_bonus = (self.max_steps - self.step_count) * 0.5
            reward += 300 + time_bonus
            self.captured = True

        # REDUCED step penalty
        reward -= 0.02

        self.step_count += 1
        done = (self.step_count >= self.max_steps) or captured

        return self._get_obs(), reward, done, {
            "captured": captured,
            "min_distance": min_d,
            "drone_collisions": collision_count,
            "obstacle_collisions": obstacle_collision_count,
        }

    def set_camera(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=25,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0,0,2]
        )

    def close(self):
        p.disconnect()