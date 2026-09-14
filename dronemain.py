from __future__ import annotations

import math
import time
import random
import numpy as np
import os
import sys
import matplotlib

# ── Headless / Jetson Nano display detection ──────────────────────────────────
# Activated automatically when no display server is found (SSH, Docker, Jetson).
# Can also be forced explicitly with the --headless CLI flag.
# Windows: Always has display unless SSH/headless. Unix: Check DISPLAY/WAYLAND_DISPLAY
import platform

_is_windows = platform.system() == 'Windows'
_has_display = _is_windows or (
    os.environ.get('DISPLAY', '').strip() != '' or
    os.environ.get('WAYLAND_DISPLAY', '').strip() != ''
)

if not _has_display:
    matplotlib.use('Agg')  # Non-interactive PNG-only backend
elif _is_windows:
    try:
        matplotlib.use('TkAgg')  # Windows-native interactive backend
    except:
        matplotlib.use('WinAgg')  # Fallback Windows backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Optional, Dict
import matplotlib.cm as cm
import argparse



# ══════════════════════════════════════════════════════════════════════════════
# SECTION -1 — GLOBAL CONFIGURATION SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EnvConfig:
    """Dimensions and physical properties of the environment."""
    width: float = 7.0           # x-axis length (m)
    height: float = 1.0          # y-axis width (m)
    depth: float = 0.5           # water depth (m)
    mu_wet: float = 0.45         # friction coefficient
    gravity: float = 9.81        # m/s^2
    obstacles: List[Vec3] = field(default_factory=list) # Cylindrical pillars (x, y, radius)

@dataclass
class ArmConfig:
    """Physical properties of the soft robotic arm."""
    n_segments: int = 10
    L0: float = 0.055            # Natural length (m)
    d: float = 0.014             # Chamber diameter (m)
    r: float = 0.007             # Distance from center to tendon hole (m)
    frequency: float = 0.35      # Wave frequency (Hz)
    amplitude: float = 0.022     # Wave amplitude (m)

@dataclass
class UAVConfig:
    """Flight dynamics and control parameters."""
    hover_alt: float = 1.0
    safe_dist: float = 0.6       # APF repulsion threshold
    k_rep: float = 0.8           # APF repulsion gain
    max_v: float = 1.0           # Max velocity (m/s)
    # PID Gains
    kp_xy: float = 1.6
    ki_xy: float = 0.04
    kd_xy: float = 0.5
    kp_z: float = 2.2
    ki_z: float = 0.08
    kd_z: float = 0.6

@dataclass
class MissionConfig:
    step_size_s: float = 0.0   
    fov_radius: float = float('inf') 
    reach_time: float = 2.5      

    coil_time: float = 3.0       
    retract_time: float = 1.2    
    payload_kg: float = 0.8     

    def resolve_step_size(self, zone_width: float):
        """
        Ensures a sensible search pattern.
        If fov_radius is infinite, we set a clean 4-pass scanning grid per zone.
        If fov_radius is finite, we set step_size to 2 * fov_radius (with a 10% overlap).
        """
        if self.step_size_s > 0.0:
            return  # Use user-defined value if provided
            
        if math.isinf(self.fov_radius):
            # Unlimited FOV: Divide the corridor into 4 clean lanes for visual sweeps
            self.step_size_s = max(0.3, zone_width / 4)
        else:
            # Finite FOV: Set step size to the camera's swath width with 10% overlap
            self.step_size_s = min(zone_width, self.fov_radius * 1.8)


@dataclass
class SimConfig:
    """Master configuration object."""
    env: EnvConfig = field(default_factory=EnvConfig)
    arm: ArmConfig = field(default_factory=ArmConfig)
    uav: UAVConfig = field(default_factory=UAVConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)



# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — ENUMERATIONS & DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

class MissionPhase(Enum):
    IDLE              = auto()
    NAVIGATE_TO_ZONE  = auto()
    HOVER_STABILIZE   = auto()
    BOUSTROPHEDON_SCAN = auto()  # Zigzag scan pattern
    DETECT_DEBRIS     = auto()
    QUAD_ARM_REACH    = auto()  # New: Helical spiral maneuver
    LASSO_COIL        = auto()
    GRIP_VERIFY       = auto()
    ARM_RETRACT       = auto()
    NAVIGATE_TO_BIN   = auto()  # New: Fly to central collection
    EJECT_DEBRIS      = auto()
    RETURN_TO_BASE    = auto()  # New: Post-mission recovery
    MISSION_COMPLETE  = auto()
    FAULT             = auto()



@dataclass
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def dist(self, o: "Vec3") -> float:
        return math.sqrt((self.x-o.x)**2 + (self.y-o.y)**2 + (self.z-o.z)**2)

    def __repr__(self):
        return f"Vec3({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"


@dataclass
class ScanningCell:
    """Geographical range ABCD for a specific robot."""
    p_top_l: Vec3    # Point A
    p_top_r: Vec3    # Point B
    p_bot_l: Vec3    # Point C
    p_bot_r: Vec3    # Point D
    
    @property
    def width(self) -> float: return self.p_top_l.dist(self.p_top_r)
    @property
    def height(self) -> float: return self.p_top_l.dist(self.p_bot_l)


@dataclass
class Segment:
    """
    One continuum segment — constant-curvature arc model.
    """
    index: int
    config: ArmConfig
    l_tendons: List[float] = field(default_factory=lambda: [0.055, 0.055, 0.055])

    @property
    def ell(self) -> float:
        """Arc length of the backbone (average of tendons)."""
        return sum(self.l_tendons) / 3.0

    @property
    def kappa_phi(self) -> Tuple[float, float]:
        """Returns (kappa, phi) based on 3-tendon geometric mapping (120 deg)."""
        L = self.ell
        l1, l2, l3 = self.l_tendons
        
        # Mapping to Cartesian bending plane
        # Tendon 1 at 0 deg, Tendon 2 at 120 deg, Tendon 3 at 240 deg
        ax = L - l1
        ay = (l3 - l2) / math.sqrt(3.0)
        
        A = math.sqrt(ax**2 + ay**2)
        kappa = A / (self.config.r * L) if L > 1e-6 else 0.0
        phi = math.atan2(ay, ax)
        return kappa, phi

    @property
    def kappa(self) -> float:
        return self.kappa_phi[0]

    @property
    def phi_plane(self) -> float:
        return self.kappa_phi[1]

    @property
    def theta_bend(self) -> float:
        return self.kappa * self.ell



    def tip_position(self) -> Tuple[float, float, float]:
        """Returns (x, y, z) in local segment frame using 3-tendon 3D kinematics."""
        k = self.kappa
        theta = self.theta_bend
        phi = self.phi_plane
        L = self.ell
        
        if k < 1e-6:
            return (L, 0.0, 0.0) 
        
        # Depth along initial backbone axis
        depth = (1/k) * math.sin(theta)
        bend_mag = (1/k) * (1 - math.cos(theta))
        
        off_x = bend_mag * math.cos(phi)
        off_y = bend_mag * math.sin(phi)
        
        return (depth, off_x, off_y)


@dataclass
class OctopusArm:
    """
    Single soft continuum arm.
    """
    config: ArmConfig
    env_config: EnvConfig
    segments: List[Segment] = field(default_factory=list)

    def __post_init__(self):
        self.segments = [Segment(index=i, config=self.config, 
                                 l_tendons=[self.config.L0]*3) for i in range(self.config.n_segments)]

    def tip_world_pos(self, base: Vec3, base_angle_rad: float) -> List[Vec3]:

        """Returns list of joint positions, rotated radially by base_angle in the 3D plane."""
        pts = [base]
        # Cumulative orientation for multi-segment chain
        # For simplicity in 10-seg, we assume tangent propagation
        accum_phi = 0.0
        accum_theta = 0.0
        
        for seg in self.segments:
            dx, dy, dz = seg.tip_position()
            
            # Rotate by base_angle_rad (robot's arm mounting position)
            # and offset by base position
            # Visual mapping: Global X, Y is the grid. Z is altitude.
            rx = dx * math.cos(base_angle_rad) - dy * math.sin(base_angle_rad)
            ry = dx * math.sin(base_angle_rad) + dy * math.cos(base_angle_rad)
            rz = dz 
            
            pts.append(Vec3(pts[-1].x + rx, pts[-1].y + ry, pts[-1].z + rz))
        return pts

    def lasso_grip_force(self, tension_N: float = 12.0, wrap_rad: float = math.pi * 2.2) -> float:
        """Capstan Equation: F_hold = (Sum of All 3 Tendons) · e^(μ·φ)"""
        # Summing tension from all 3 strings in your hardware disk
        total_tension = tension_N * 3
        return total_tension * math.exp(self.env_config.mu_wet * wrap_rad)



# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — BEND PROPAGATION CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

class BendPropagationController:
    """
    Generates sinusoidal actuator waves that propagate from base → tip.
    """
    def __init__(self, arm: OctopusArm, config: ArmConfig):
        self.arm = arm
        self.f = config.frequency
        self.dt_seg = 0.25 # Relative phase delay
        self.A = config.amplitude

    def update(self, t: float):
        L0 = self.arm.config.L0

        for i, seg in enumerate(self.arm.segments):
            ph = 2 * math.pi * self.f * t - i * self.dt_seg
            # Drive 3 tendons with 120 deg phase offsets (0, 120, 240 deg)
            for j in range(3):
                seg.l_tendons[j] = max(L0*0.6, min(L0 + self.A * math.sin(ph + j * (2*math.pi/3)), L0*1.8))

    def retract(self):
        for seg in self.arm.segments:
            seg.l_tendons = [seg.config.L0] * 3



# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — UAV PID HOVER CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PIDAxis:
    Kp: float; Ki: float; Kd: float
    _i: float = 0.0; _pe: float = 0.0
    def compute(self, e: float, dt: float) -> float:
        self._i += e * dt
        d = (e - self._pe) / dt if dt > 1e-6 else 0.0
        self._pe = e
        return self.Kp*e + self.Ki*self._i + self.Kd*d
    def reset(self): self._i = 0.0; self._pe = 0.0


class UAVHoverController:
    STABLE_THRESHOLD = 0.05  # metres
    def __init__(self, cfg: UAVConfig):
        self.px = PIDAxis(cfg.kp_xy, cfg.ki_xy, cfg.kd_xy)
        self.py = PIDAxis(cfg.kp_xy, cfg.ki_xy, cfg.kd_xy)
        self.pz = PIDAxis(cfg.kp_z, cfg.ki_z, cfg.kd_z)
        self.target = Vec3()

    def set_target(self, t: Vec3):
        self.target = t; self.px.reset(); self.py.reset(); self.pz.reset()
    def update(self, pos: Vec3, dt: float) -> Vec3:
        return Vec3(self.px.compute(self.target.x - pos.x, dt),
                    self.py.compute(self.target.y - pos.y, dt),
                    self.pz.compute(self.target.z - pos.z, dt))
    def is_stable(self, pos: Vec3) -> bool:
        return pos.dist(self.target) < self.STABLE_THRESHOLD


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DEBRIS DETECTOR (Intensity Centroid)
# ══════════════════════════════════════════════════════════════════════════════

class DebrisDetector:
    """
    Look-Compute-Move: Intensity-weighted sensing.
    Determines centroid of a debris blob from a simulated camera frame.
    """
    def __init__(self, grid: int = 32):
        self.grid = grid

    def detect(self, uav_pos: Vec3, waypoints: List[Vec3], config: MissionConfig, noise_std: float = 0.04) -> Optional[Tuple[float, float]]:
        """
        Returns normalised centroid (Cx, Cy) only if a real waypoint is within 
        the camera's FOV radius.
        """
        # Proximity Check: Is any real debris physically under the drone?
        found = False
        for wp in waypoints:
            dist_2d = math.sqrt((uav_pos.x - wp.x)**2 + (uav_pos.y - wp.y)**2)
            if dist_2d < config.fov_radius: # Configurable camera FOV
                found = True
                break

        
        if not found:
            return None

        # Generate visual centroid for the robot's internal logic
        cx, cy = 0.5 + random.uniform(-0.05, 0.05), 0.5 + random.uniform(-0.05, 0.05)
        frame = np.random.normal(0.04, noise_std, (self.grid, self.grid))
        ci, cj = int(cy * self.grid), int(cx * self.grid)
        for i in range(self.grid):
            for j in range(self.grid):
                frame[i, j] += max(0, 1.0 - math.hypot(i-ci, j-cj) / 5.5)
        frame = np.clip(frame, 0, 1)
        S = frame.sum()
        if S < 1e-6: return None
        rows = np.arange(self.grid).reshape(-1, 1)
        cols = np.arange(self.grid).reshape(1, -1)
        return (float((cols * frame).sum() / S) / self.grid,
                float((rows * frame).sum() / S) / self.grid)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SINGLE-ROBOT MISSION (Dual-Arm Lasso)
# ══════════════════════════════════════════════════════════════════════════════

class RobotUnit:
    """
    One UAV with four octopus arms executing the mission.
    """
    def __init__(self, robot_id: int, start_pos: Vec3, config: SimConfig, colour: str):
        self.id        = robot_id
        self.config    = config
        self.colour    = colour
        # Start position
        self.uav_pos   = Vec3(start_pos.x, start_pos.y, start_pos.z)
        self.phase     = MissionPhase.IDLE
        self.cell: Optional[ScanningCell] = None  # ABCD zone
        self.scan_target: Optional[Vec3] = None

        # Quad-arm setup
        self.arms = [OctopusArm(config=config.arm, env_config=config.env) for _ in range(4)]
        self.S = config.mission.step_size_s

        # Scan state tracking
        self.scan_x = 0.0
        self.scan_dir = 1.0  # 1.0 for UP, -1.0 for DOWN
        self.scan_started = False
        self.is_bridging = False
        
        # Resume state for collection trips
        self.resume_pos = None 
        self.detected_wp: Optional[Vec3] = None


        self.ctrls = [BendPropagationController(arm, config=config.arm) for arm in self.arms]
        self.hover_ctrl = UAVHoverController(config.uav)
        self.detector   = DebrisDetector()

        self._phase_time = 0.0
        self.collected   = 0
        self.total_collected_all_time = 0
        self.lasso_force = 0.0
        self.log: List[str] = []
        self.path_history: List[Vec3] = []
        self.last_collected_wp: Optional[Vec3] = None
        self.carried_wp: Optional[Vec3] = None


    # ── Logging ──────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        entry = f"[R{self.id}][{self.phase.name:<16}] {msg}"
        self.log.append(entry)
        print(entry)

    def _transition(self, p: MissionPhase):
        self._log(f"-> {p.name}")
        self.phase = p
        self._phase_time = 0.0

    def _update_position(self, corr: Vec3, dt: float):
        """Applies velocity clamping to physical limits (config.uav.max_v) and updates position."""
        max_v = self.config.uav.max_v
        vx = max(-max_v, min(corr.x, max_v))
        vy = max(-max_v, min(corr.y, max_v))
        vz = max(-max_v, min(corr.z, max_v))
        
        self.uav_pos.x += vx * dt
        self.uav_pos.y += vy * dt
        self.uav_pos.z += vz * dt
        self.path_history.append(Vec3(self.uav_pos.x, self.uav_pos.y, self.uav_pos.z))

    # ── Phase Handlers ───────────────────────────────────────────────────────

    def _ph_idle(self):
        if self.cell:
            self._transition(MissionPhase.NAVIGATE_TO_ZONE)

    def _ph_navigate(self, dt: float):
        # Target: Bottom-left of assigned cell (Point C)
        tgt = Vec3(self.cell.p_bot_l.x, self.cell.p_bot_l.y, self.config.uav.hover_alt)
        self.hover_ctrl.set_target(tgt)

        
        # Move drone
        corr = self.hover_ctrl.update(self.uav_pos, dt)
        self._update_position(corr, dt)

        # Once reaching corner, stabilize
        if self.uav_pos.dist(tgt) < 0.1:
            self._transition(MissionPhase.HOVER_STABILIZE)

    def _ph_hover(self, dt: float, waypoints: List[Vec3]):
        corr = self.hover_ctrl.update(self.uav_pos, dt)
        self._update_position(corr, dt)
        
        if self.hover_ctrl.is_stable(self.uav_pos):
            if not self.scan_started:
                self.scan_x = self.cell.p_bot_l.x
                self.scan_started = True
                self._transition(MissionPhase.BOUSTROPHEDON_SCAN)
            else:
                self._transition(MissionPhase.BOUSTROPHEDON_SCAN)

    def _ph_boustrophedon(self, dt: float, waypoints: List[Vec3]):
        """Methodical Left-to-Right Debris Pursuit Scan Propagation."""
        # Filter waypoints inside our assigned geo-political corridor, excluding any currently carried debris
        zone_wps = [wp for wp in waypoints if self.cell.p_bot_l.x <= wp.x <= self.cell.p_top_r.x and wp != self.carried_wp]
        
        # Sort targets strictly from left to right (increasing X-coordinate)
        sorted_wps = sorted(zone_wps, key=lambda w: w.x)
        
        if sorted_wps:
            # Target the leftmost uncollected debris in our corridor
            target_wp = sorted_wps[0]
            self.scan_target = Vec3(target_wp.x, target_wp.y, self.config.uav.hover_alt)
            self.hover_ctrl.set_target(self.scan_target)
            
            # Move directly towards target
            corr = self.hover_ctrl.update(self.uav_pos, dt)
            self._update_position(corr, dt)
            
            # Transition to DETECT_DEBRIS when close (within physical reach of 0.4m)
            dist_2d = math.hypot(self.uav_pos.x - target_wp.x, self.uav_pos.y - target_wp.y)
            if dist_2d < 0.4:
                self.detected_wp = target_wp
                self._transition(MissionPhase.DETECT_DEBRIS)
                return
            
            # Safety: if stuck trying to reach target for too long (timeout), skip it
            # This prevents deadlock from APF collision avoidance
            self._phase_time += dt
            if self._phase_time > 15.0:  # 15 seconds = stuck timeout
                self._log(f"Timeout reaching debris at ({target_wp.x:.2f}, {target_wp.y:.2f}). Skipping...")
                self._phase_time = 0.0  # Reset timer for next target
                # Force skip by doing another scan immediately (next iteration will pick next target)
                return
        else:
            # All targets in corridor collected! Return to leftmost bottom base
            self._log(f"Zone {self.id} coverage 100%. Returning to base...")
            self._transition(MissionPhase.RETURN_TO_BASE)

    def _ph_detect(self, dt: float):
        """Phase: Fly directly over the globally detected debris before launching robotic reach."""
        target = Vec3(self.detected_wp.x, self.detected_wp.y, self.config.uav.hover_alt)
        self.hover_ctrl.set_target(target)
        
        corr = self.hover_ctrl.update(self.uav_pos, dt)
        self._update_position(corr, dt)

        if self.uav_pos.dist(target) < 0.1:
            self._log(f"Arrived directly over debris at ({self.detected_wp.x:.2f}, {self.detected_wp.y:.2f}).")
            self._transition(MissionPhase.QUAD_ARM_REACH)



    def _ph_quad_reach(self, dt: float):
        """Phase: Helical spiral maneuver to prepare for lasso."""
        self._phase_time += dt
        f_spiral = 0.5  # Rotation frequency
        A_spiral = 0.02 # Spiral amplitude
        
        for arm_idx, arm in enumerate(self.arms):
            for i, seg in enumerate(arm.segments):
                ph = 2 * math.pi * f_spiral * self._phase_time - i*0.2
                # Generate 3-phase waves for spiralling (0, 120, 240 deg)
                for j in range(3):
                    seg.l_tendons[j] = arm.config.L0 + A_spiral * math.sin(ph + j * (2*math.pi/3))

        
        if self._phase_time >= 2.0: # Spiralling time
            self._transition(MissionPhase.LASSO_COIL)


    def _ph_coil(self, dt: float):
        """Phase: Isometric Cinching. Tighten all tendons to lock the helical grip."""
        self._phase_time += dt
        # Increase tension in all 4 tendons across all segments
        tension_rate = 0.008 
        for arm in self.arms:
            for i, seg in enumerate(arm.segments):
                # Apply more tension to distal (tip) segments for a "lock"
                distal_factor = (i + 1) / arm.config.n_segments

                for t_idx in range(3):
                    seg.l_tendons[t_idx] += tension_rate * distal_factor * dt 
        
        if self._phase_time >= self.config.mission.coil_time:
            self._transition(MissionPhase.GRIP_VERIFY)


    def _ph_verify(self):
        """Phase: Grip Verification. Checks capstan force across all 4 spiralled arms."""
        # 1.9 full helical wraps
        phi = 2.0 * math.pi * 1.9   
        
        # Calculate combined force. 
        # Synergy factor: 4 arms interlocking increases effective friction
        synergy = 1.2 
        self.lasso_force = sum(arm.lasso_grip_force(wrap_rad=phi) for arm in self.arms) * synergy
        
        required = self.config.mission.payload_kg * self.config.env.gravity
        ok = self.lasso_force >= required

        
        self._log(f"QUAD-HELICAL FORCE = {self.lasso_force:.1f} N "
                  f"({'SECURE' if ok else 'WEAK'})")
        
        if ok:
            self.collected += 1
            self.total_collected_all_time += 1
            self.carried_wp = self.detected_wp # Lock the debris as carried
            
            # Save current scan work point to return here later
            self.resume_pos = Vec3(self.uav_pos.x, self.uav_pos.y, self.config.uav.hover_alt)
            self._transition(MissionPhase.ARM_RETRACT)

        else:
            self._transition(MissionPhase.FAULT)

    def _ph_retract(self, dt: float):
        self._phase_time += dt
        for ctrl in self.ctrls:
            ctrl.retract()

        if self._phase_time >= self.config.mission.retract_time:
            self._transition(MissionPhase.NAVIGATE_TO_BIN)


    def _ph_nav_bin(self, dt: float):
        """Phase: Fly to the local bin at the center of the assigned zone."""
        # Calculate local center of the assigned cell
        bin_x = (self.cell.p_top_l.x + self.cell.p_top_r.x) / 2
        bin_y = (self.cell.p_top_l.y + self.cell.p_bot_l.y) / 2
        bin_tgt = Vec3(bin_x, bin_y, self.config.uav.hover_alt + 0.5) # Fly slightly higher

        self.hover_ctrl.set_target(bin_tgt)
        
        corr = self.hover_ctrl.update(self.uav_pos, dt)
        self._update_position(corr, dt)

        if self.uav_pos.dist(bin_tgt) < 0.1:
            self._transition(MissionPhase.EJECT_DEBRIS)

    def _ph_eject(self, dt: float):
        """Phase: Active Ejection. High-speed reverse wave to throw debris."""
        self._phase_time += dt
        # Fast "unspooling" wave
        f_eject = 1.2 
        A_eject = 0.03 # Wider snap
        
        for arm in self.arms:
            for i, seg in enumerate(arm.segments):
                ph = 2 * math.pi * f_eject * self._phase_time - i*0.4
                # Drive tendons to snap straight
                for j in range(3):
                    # We want all tendons to reach L0+A quickly
                    seg.l_tendons[j] = seg.config.L0 + A_eject * math.sin(ph + j * (2*math.pi/3))

        
        if self._phase_time >= 1.5: # Ejection snap time
            self._finish_waypoint()

    def _finish_waypoint(self):
        """Resume scanning for the next leftmost target."""
        # Deposit the carried debris exactly at the local bin center!
        bin_x = (self.cell.p_top_l.x + self.cell.p_top_r.x) / 2.0
        bin_y = (self.cell.p_top_l.y + self.cell.p_bot_l.y) / 2.0
        
        if self.carried_wp:
            self.carried_wp.x = bin_x
            self.carried_wp.y = bin_y
            self.carried_wp.z = 0.0 # Placed on the floor of the bin!
            
            # Now mark it as collected so it is officially added to completed_wps
            self.last_collected_wp = self.carried_wp
            self.carried_wp = None
            
        self._transition(MissionPhase.BOUSTROPHEDON_SCAN)



    def _ph_return_to_base(self, dt: float):
        """Phase: Mission area clear. Return to leftmost bottom point for recovery."""
        start_tgt = Vec3(self.cell.p_bot_l.x, self.cell.p_bot_l.y, self.config.uav.hover_alt + 0.5)
        self.hover_ctrl.set_target(start_tgt)
        
        corr = self.hover_ctrl.update(self.uav_pos, dt)
        self._update_position(corr, dt)

        if self.uav_pos.dist(start_tgt) < 0.1:
            self._transition(MissionPhase.MISSION_COMPLETE)

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self, waypoints_pool: List[Vec3], other_uavs: List['RobotUnit'], dt: float = 0.05):
        if self.phase == MissionPhase.IDLE:
            self._ph_idle()
        elif self.phase == MissionPhase.NAVIGATE_TO_ZONE:
            self._ph_navigate(dt)
        elif self.phase == MissionPhase.HOVER_STABILIZE:
            self._ph_hover(dt, waypoints_pool)
        elif self.phase == MissionPhase.BOUSTROPHEDON_SCAN:
            self._ph_boustrophedon(dt, waypoints_pool)
        elif self.phase == MissionPhase.DETECT_DEBRIS:
            self._ph_detect(dt)

        elif self.phase == MissionPhase.QUAD_ARM_REACH:
            self._ph_quad_reach(dt)
        elif self.phase == MissionPhase.LASSO_COIL:
            self._ph_coil(dt)
        elif self.phase == MissionPhase.GRIP_VERIFY:
            self._ph_verify()
        elif self.phase == MissionPhase.ARM_RETRACT:
            self._ph_retract(dt)
        elif self.phase == MissionPhase.NAVIGATE_TO_BIN:
            self._ph_nav_bin(dt)
        elif self.phase == MissionPhase.EJECT_DEBRIS:
            self._ph_eject(dt)
        elif self.phase == MissionPhase.RETURN_TO_BASE:
            self._ph_return_to_base(dt)
        elif self.phase == MissionPhase.MISSION_COMPLETE:
            pass

        elif self.phase == MissionPhase.FAULT:
            self._log("FAULT — reset scan")
            self.carried_wp = None # Ensure nothing is carried on fault
            self._transition(MissionPhase.BOUSTROPHEDON_SCAN)

        # Physically carry debris with the UAV during transit
        if self.carried_wp:
            self.carried_wp.x = self.uav_pos.x
            self.carried_wp.y = self.uav_pos.y
            self.carried_wp.z = self.uav_pos.z - 0.25 # Dangling below the drone's center

        self._apply_anti_collision(other_uavs, dt)

    def _apply_anti_collision(self, other_uavs: List['RobotUnit'], dt: float):
        if self.phase in (MissionPhase.IDLE, MissionPhase.MISSION_COMPLETE, MissionPhase.FAULT):
            return
        
        # Artificial Potential Field (APF) Anti-Collision with target attraction
        force_x, force_y = 0.0, 0.0
        safe_dist = self.config.uav.safe_dist
        k_rep = self.config.uav.k_rep
        target = self.hover_ctrl.target

        # Get direction to target (attraction)
        tgt_dx = target.x - self.uav_pos.x
        tgt_dy = target.y - self.uav_pos.y
        tgt_dist = math.hypot(tgt_dx, tgt_dy)
        if tgt_dist > 1e-3:
            tgt_dx /= tgt_dist
            tgt_dy /= tgt_dist

        # Drone-to-drone repulsion
        for peer in other_uavs:
            if peer.phase in (MissionPhase.IDLE, MissionPhase.MISSION_COMPLETE):
                continue
            dx = self.uav_pos.x - peer.uav_pos.x
            dy = self.uav_pos.y - peer.uav_pos.y
            dist = math.hypot(dx, dy)
            if 0.01 < dist < safe_dist:
                rep = k_rep * (1.0/dist - 1.0/safe_dist) / (dist**2)
                force_x += rep * (dx/dist)
                force_y += rep * (dy/dist)
                
        # Obstacle repulsion with smart bypass routing
        for obs in self.config.env.obstacles:
            dx = self.uav_pos.x - obs.x
            dy = self.uav_pos.y - obs.y
            dist_center = math.hypot(dx, dy)
            dist = dist_center - obs.z
            
            if 0.01 < dist < safe_dist:
                # Reduced repulsion strength to allow forward progress toward target
                rep = k_rep * 0.8 * (1.0/dist - 1.0/safe_dist) / (dist**2)
                
                # Radial push (away from obstacle center)
                force_x += rep * (dx/dist_center)
                force_y += rep * (dy/dist_center)

                # Tangential bypass (flow around obstacle)
                if dist_center > 1e-6:
                    tx = -dy / dist_center
                    ty = dx / dist_center
                    
                    # Choose bypass direction aligned with target
                    if tx * tgt_dx + ty * tgt_dy < 0:
                        tx = -tx
                        ty = -ty
                    
                    force_x += rep * 0.6 * tx
                    force_y += rep * 0.6 * ty
        
        # Apply forces with smoothing to avoid jitter
        if abs(force_x) > 0.001 or abs(force_y) > 0.001:
            max_v = 0.8  # Slightly reduced to improve precision near targets
            vx = max(-max_v, min(force_x, max_v))
            vy = max(-max_v, min(force_y, max_v))
            
            self.uav_pos.x += vx * dt
            self.uav_pos.y += vy * dt
            
            # Constrain to environment bounds
            self.uav_pos.x = max(0.0, min(self.uav_pos.x, self.config.env.width))
            self.uav_pos.y = max(0.0, min(self.uav_pos.y, self.config.env.height))


    @property
    def is_idle(self) -> bool:
        return self.phase == MissionPhase.IDLE

    @property
    def is_done(self) -> bool:
        return self.phase == MissionPhase.MISSION_COMPLETE


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SWARM COORDINATOR
# ══════════════════════════════════════════════════════════════════════════════

class SwarmCoordinator:
    """
    Manages a fleet of RobotUnit instances.
    """

    def __init__(self, n_robots: int, waypoints: List[Vec3], config: SimConfig = None):
        self.config = config or SimConfig()
        
        # Partition the total channel into N vertical cells
        total_min_x, total_max_x = 0.0, self.config.env.width
        total_min_y, total_max_y = 0.0, self.config.env.height
        cell_width = (total_max_x - total_min_x) / n_robots

        # Resolve dynamic step size for the search corridors
        self.config.mission.resolve_step_size(cell_width)
        print(f"[SwarmCoordinator] Optimized search grid: step_size = {self.config.mission.step_size_s:.3f}m")

        # Dynamic color generation
        cmap = plt.get_cmap('viridis', n_robots)


        
        self.robots: List[RobotUnit] = []
        for i in range(n_robots):
            x1 = total_min_x + i * cell_width
            x2 = x1 + cell_width
            
            # Define ABCD zone for this robot
            cell = ScanningCell(
                p_top_l=Vec3(x1, total_max_y, 0.0),
                p_top_r=Vec3(x2, total_max_y, 0.0),
                p_bot_l=Vec3(x1, total_min_y, 0.0),
                p_bot_r=Vec3(x2, total_min_y, 0.0)
            )
            
            # Start robot at the leftmost bottom point of its zone
            rgb = cmap(i)[:3]
            hex_color = '#%02x%02x%02x' % tuple(int(x*255) for x in rgb)
            
            robot = RobotUnit(i, Vec3(x1, total_min_y, self.config.uav.hover_alt), self.config, hex_color)
            robot.cell = cell
            self.robots.append(robot)


        self._total_collected = 0
        self.completed_wps: List[Vec3] = []

    # ── Swarm Management ──────────────────────────────────────────────────────

    def _assign_tasks(self):
        """Tasks are now pre-assigned by cells."""
        pass

    def _collect_done_robots(self, waypoints: List[Vec3]):
        for robot in self.robots:
            if robot.last_collected_wp:
                if robot.last_collected_wp not in self.completed_wps:
                    self.completed_wps.append(robot.last_collected_wp)
                robot.last_collected_wp = None # reset flag

            if robot.collected > 0:
                self._total_collected += robot.collected
                robot.collected = 0

    # ── Visualisation ─────────────────────────────────────────────────────────

    def _draw(self, ax: plt.Axes, all_wps: List[Vec3]):
        ax.cla() # Clear for fresh animation frame
        ax.set_title(
            f"UAV SWARM — Parametric Mission Scanning\n"
            f"Env: {self.config.env.width}m x {self.config.env.height}m | Robots: {len(self.robots)}",
            fontweight='bold', fontsize=12, color='white', pad=10
        )
        ax.set_xlabel("Length (X)"); ax.set_ylabel("Width (Y)")
        ax.set_facecolor('#0b0b18') # Deep space dark mode
        
        # Dynamic limits
        ax.set_xlim(-0.5, self.config.env.width + 0.5)
        ax.set_ylim(-0.5, self.config.env.height + 0.5)

        # 2D Environment: Sewer Channel
        ax.axhspan(-0.1, 0.0, color='#3c2f2f', alpha=1.0) # Bottom wall
        ax.axhspan(self.config.env.height, self.config.env.height + 0.1, color='#3c2f2f', alpha=1.0)  # Top wall
        ax.axhspan(0.0, self.config.env.height, color='#1e6091', alpha=0.3)  # Water channel


        # Localized Garbage Bins (per robot)
        for robot in self.robots:
            c = robot.cell
            bx = (c.p_top_l.x + c.p_top_r.x) / 2
            by = (c.p_top_l.y + c.p_bot_l.y) / 2
            
            bin_circ = mpatches.Circle((bx, by), 0.2, color=robot.colour, alpha=0.5, 
                                       linestyle='--', linewidth=1.5, fill=False)
            ax.add_patch(bin_circ)
            ax.text(bx, by + 0.25, f"BIN {robot.id}", color=robot.colour, 
                    fontsize=8, ha='center', fontweight='bold')

        # Draw Zone Boundaries
        for robot in self.robots:
            c = robot.cell
            bx = [c.p_top_l.x, c.p_top_r.x, c.p_bot_r.x, c.p_bot_l.x, c.p_top_l.x]
            by = [c.p_top_l.y, c.p_top_r.y, c.p_bot_r.y, c.p_bot_l.y, c.p_top_l.y]
            ax.plot(bx, by, color=robot.colour, alpha=0.3, linewidth=1, linestyle='--')

        # Draw Static Concrete Column Obstacles
        for obs in self.config.env.obstacles:
            pillar = mpatches.Circle((obs.x, obs.y), obs.z, facecolor='#7f8c8d', alpha=0.9, 
                                     edgecolor='#34495e', linewidth=2.0, fill=True, zorder=6)
            ax.add_patch(pillar)
            pillar.set_hatch('///')
            ax.text(obs.x, obs.y, "COLUMN", color='#2c3e50', fontsize=8, ha='center', va='center', fontweight='bold', zorder=7)

        # Debris Waypoints (2D Stars)
        for wp in all_wps:
            is_done = any(wp.dist(done_wp) < 0.01 for done_wp in self.completed_wps)
            ax.scatter(wp.x, wp.y, s=150, marker='*',
                       color='#bdc3c7' if is_done else '#f1c40f',
                       edgecolor='black', linewidth=0.5, zorder=6,
                       alpha=0.1 if is_done else 1.0)

        # Each robot
        legend_patches = []
        for robot in self.robots:
            c = robot.colour

            # 2D Path Trail
            if robot.path_history:
                tx = [p.x for p in robot.path_history]
                ty = [p.y for p in robot.path_history]
                ax.plot(tx, ty, color=c, alpha=0.4, linewidth=1.0)

            # UAV Head (2D Marker)
            ax.scatter(robot.uav_pos.x, robot.uav_pos.y, 
                       s=300, marker='H', color=c, edgecolor='white',
                       linewidth=1.5, zorder=7)
            
            # Mission Status Label
            ax.text(robot.uav_pos.x, robot.uav_pos.y + 0.08,
                    f"R{robot.id}: {robot.phase.name}",
                    color='white', fontsize=7, ha='center',
                    bbox=dict(fc=c, alpha=0.7, pad=1, ec='none'), zorder=8)

            # Arms (Projected to 2D)
            angles = [0.0, math.pi/2, math.pi, 3*math.pi/2]
            for arm, angle in zip(robot.arms, angles):
                pts = arm.tip_world_pos(robot.uav_pos, angle)
                xs = [p.x for p in pts]
                ys = [p.y for p in pts]
                ax.plot(xs, ys, color=c, linewidth=3.5,
                        alpha=0.8, solid_capstyle='round', zorder=5)

            legend_patches.append(mpatches.Patch(color=c, label=f"R{robot.id} ({robot.phase.name})"))

        ax.grid(True, alpha=0.1, color='white')

    # ── Main Loop ─────────────────────────────────────────────────────────────

    def run(self, step_delay: float = 0.01, max_steps: int = 2000,
            waypoints: List[Vec3] = None, headless: bool = False):
        all_wps = waypoints or []
        # Headless if backend is Agg OR caller explicitly forced it
        is_headless = (matplotlib.get_backend().lower() == 'agg') or headless
        
        if not is_headless:
            plt.ion()
        fig, ax = plt.subplots(figsize=(14, 7))
        plt.tight_layout()

        print("=" * 72)
        print(f"  BOUSTROPHEDON SWARM -- {len(self.robots)} Robots | "
              f"Step Pattern: S = L + n")
        print("=" * 4)

        for step in range(max_steps):
            self._assign_tasks()
            
            # Identify uncollected waypoints
            uncollected = [wp for wp in all_wps if wp not in self.completed_wps]
            
            for robot in self.robots:
                peers = [r for r in self.robots if r != robot]
                robot.step(waypoints_pool=uncollected, other_uavs=peers, dt=0.04) # Slower movement
            self._collect_done_robots(all_wps)

            if step % 1 == 0:
                if not is_headless:
                    # Only render live when a display is available
                    self._draw(ax, all_wps)
                    plt.draw()
                    plt.pause(0.08)

            # Done when all robots in mission complete phase
            if all(r.phase == MissionPhase.MISSION_COMPLETE for r in self.robots):
                print(f"  SIMULATION COMPLETED IN {step} LOOPS")
                break

        # Final frame — always render one frame at the end
        self._draw(ax, all_wps)
        if is_headless:
            fig.savefig("simulation_final_state.png", dpi=150)
            print("  [Headless] Saved final state  → simulation_final_state.png")
        else:
            plt.pause(1.0)
            plt.ioff()

        # Summary
        total = self._total_collected
        print("=" * 4)
        print(f"  SWARM MISSION COMPLETE")
        for r in self.robots:
            print(f"    Robot {r.id}: {r.total_collected_all_time} items | "
                  f"Last force: {r.lasso_force:.1f} N")
        print(f"  TOTAL DEBRIS COLLECTED: {total}")
        print("=" * 72)

        # Trajectory plot
        self._plot_trajectories()
        if is_headless:
            plt.savefig("flight_trajectories.png", dpi=150)
            print("  [Headless] Saved flight paths    → flight_trajectories.png")
            plt.close('all')  # Release all figure memory cleanly
        else:
            plt.draw()
            plt.pause(2.0)  # Keep windows visible for 2 seconds before closing
            plt.close('all')  # Auto-close instead of blocking with plt.show()

    # ── Post-mission Trajectory Plot ──────────────────────────────────────────

    def _plot_trajectories(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("UAV Swarm — Flight Path Trajectories", fontweight='bold')
        ax.set_xlabel("X (m)"); ax.set_ylabel("Altitude Z (m)")
        ax.set_facecolor('#0f0f1a')
        ax.grid(True, alpha=0.15, color='white')
        
        for robot in self.robots:
            if robot.path_history:
                xs = [p.x for p in robot.path_history]
                zs = [p.z for p in robot.path_history]
                ax.plot(xs, zs, color=robot.colour, linewidth=1.5, alpha=0.8, label=f"R{robot.id} Path")
                
                # Mark final position
                ax.text(robot.uav_pos.x, robot.uav_pos.z,
                        f"R{robot.id}", color=robot.colour, fontsize=9, fontweight='bold')
        
        ax.set_xlim(-0.5, self.config.env.width + 0.5)
        ax.set_ylim(-0.1, self.config.uav.hover_alt + 1.0)
        ax.legend(fontsize=8, loc='upper right')
        plt.tight_layout()



# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def generate_complex_obstacles(config: EnvConfig, n_robots: int) -> List[Vec3]:
    """Generates complex concrete structural columns (pillars) in each drone's corridor."""
    obs_list = []
    zone_width = config.width / n_robots
    for i in range(n_robots):
        x1 = i * zone_width
        x2 = (i + 1) * zone_width
        x_center = (x1 + x2) / 2.0
        
        # Place 2 concrete pillars (radius 0.18m) in each corridor:
        # Lower pillar
        obs_list.append(Vec3(x_center, config.height * 0.35, 0.18))
        # Upper pillar
        obs_list.append(Vec3(x_center, config.height * 0.65, 0.18))
    return obs_list


def generate_random_debris(n_per_side: int, config: EnvConfig, n_sides: int) -> List[Vec3]:
    """Generates a high density of debris, with exactly `n_per_side` targets in each corridor, avoiding pillars."""
    wps = []
    zone_width = config.width / n_sides
    for side in range(n_sides):
        x_min = side * zone_width + 0.15
        x_max = (side + 1) * zone_width - 0.15
        for _ in range(n_per_side):
            # Keep generating until a point is clear of columns
            while True:
                wp = Vec3(
                    random.uniform(x_min, x_max),
                    random.uniform(0.15, config.height - 0.15),
                    0.0
                )
                too_close = False
                for obs in config.obstacles:
                    if math.hypot(wp.x - obs.x, wp.y - obs.y) < (obs.z + 0.2):
                        too_close = True
                        break
                if not too_close:
                    wps.append(wp)
                    break
    return wps


if __name__ == "__main__":
    # ── Command Line Argument Parsing ─────────────────────────────────────────
    parser = argparse.ArgumentParser(description="UAV Swarm Research Simulation")
    parser.add_id = parser.add_argument
    parser.add_id("--n", type=int, default=3, help="Number of drones in the swarm")
    parser.add_id("--targets", type=int, default=10, help="Number of initial debris targets")
    parser.add_id("--width", type=float, default=None, help="Environment width (auto-scales if None)")
    parser.add_id("--height", type=float, default=2.0, help="Environment height")
    parser.add_id("--fov", type=str, default="inf", help="Camera FOV radius (e.g. 0.35, or 'inf' for unlimited)")
    parser.add_id("--step-size", type=float, default=0.0, help="Custom Boustrophedon step size (0.0 for auto)")
    parser.add_id("--payload", type=float, default=0.8, help="Target payload weight in kg")
    parser.add_id("--headless", action="store_true",
                  help="Force headless mode: skip live GUI, save PNG reports (auto-detected on Jetson)")
    
    args = parser.parse_args()
    
    n_robots = args.n
    n_targets = args.targets
    
    # Parse FOV
    fov_val = args.fov.strip().lower()
    if fov_val == "inf" or fov_val == "infinite":
        fov_radius = float('inf')
    else:
        try:
            fov_radius = float(args.fov)
        except ValueError:
            print(f"Invalid FOV '{args.fov}'. Defaulting to 'inf'.")
            fov_radius = float('inf')

    print("\n" + "="*50)
    print("  UAV SWARM RESEARCH SIMULATION")
    print(f"  Configuration: N={n_robots} | Targets={n_targets}")
    print("="*50)

    # ── Configuration ─────────────────────────────────────────────────────────
    # Scaling width based on number of robots if not provided
    env_width = args.width if args.width else max(7.0, n_robots * 3.0) 
    
    config = SimConfig(
        env=EnvConfig(width=env_width, height=args.height),
        uav=UAVConfig(hover_alt=1.2, safe_dist=0.8),
        mission=MissionConfig(
            step_size_s=args.step_size,
            fov_radius=fov_radius,
            payload_kg=args.payload
        )
    )

    # Generate static concrete column obstacles in each corridor
    config.env.obstacles = generate_complex_obstacles(config.env, n_robots)
    
    # Generate high density debris evenly on each side (corridor)
    debris_waypoints = generate_random_debris(n_targets, config.env, n_robots)

    print(f"\nScenario: {config.env.width}x{config.env.height}m | Robots: {n_robots} | Targets Per Side: {n_targets} (Total: {n_robots * n_targets})")
    print(f"Waypoints: {[str(w) for w in debris_waypoints]}\n")

    swarm = SwarmCoordinator(n_robots=n_robots, waypoints=debris_waypoints, config=config)
    swarm.run(step_delay=0.005, max_steps=5000, waypoints=debris_waypoints,
              headless=args.headless)