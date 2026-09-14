# UAV Swarm Debris-Collection Simulation — `dronemain.py`

A single-file Python simulation of a **multi-drone (UAV) swarm** that searches a
water channel (e.g. a sewer/canal cross-section), finds debris, and uses
**four soft, tentacle-like continuum robotic arms per drone** to lasso-grip and
collect it into a local "bin," while avoiding static obstacles and each other.

This README is written so that another AI assistant (or a human) picking up
this file cold can understand its structure, run it, and know exactly which
parts are broken or simplified before making changes.

---

## 1. What this program actually does

It is a **2D physics-flavored simulation with matplotlib visualization**, not
a real flight controller. There is no hardware I/O — everything (drone
position, arm bending, forces) is computed analytically and plotted.

High-level flow:

1. The environment (a channel) is divided into `n_robots` equal vertical
   corridors ("cells"), each with static cylindrical "concrete pillar"
   obstacles and randomly scattered debris waypoints.
2. Each drone (`RobotUnit`) is assigned one corridor and runs an independent
   **finite-state-machine mission** (`MissionPhase` enum) to scan its
   corridor left-to-right (boustrophedon/zig-zag style), fly to each piece of
   debris, "lasso" it with 4 soft robotic arms, carry it to a bin at the
   center of its corridor, drop it, and repeat until the corridor is clear.
3. Drones repel each other and steer around obstacles using an **Artificial
   Potential Field (APF)** for collision avoidance.
4. A live matplotlib animation shows the swarm operating; on machines with no
   display (SSH/Docker/Jetson) it auto-detects headless mode and saves PNG
   snapshots instead.

## 2. Domain model / architecture

The file is organized into numbered sections (its own comments call these
out with `SECTION -1` … `SECTION 6`):

| Section | Contents |
|---|---|
| -1 | `EnvConfig`, `ArmConfig`, `UAVConfig`, `MissionConfig`, `SimConfig` — all tunable parameters, grouped as dataclasses |
| 0 | `MissionPhase` enum (the FSM states), `Vec3`, `ScanningCell`, `Segment` (one arc-segment of a soft arm), `OctopusArm` |
| 1 | `BendPropagationController` — drives sinusoidal tendon-length waves down an arm (used for the idle "swimming" wave motion) |
| 2 | `PIDAxis` / `UAVHoverController` — simple 3-axis PID controller used to fly the drone toward any XYZ target |
| 3 | `DebrisDetector` — fakes a camera by generating a noisy 32×32 intensity grid and computing an intensity centroid, only triggered if a real waypoint is within `fov_radius` |
| 4 | `RobotUnit` — the big one: one drone + 4 arms + its own mission state machine (all `_ph_*` methods) + APF anti-collision |
| 5 | `SwarmCoordinator` — owns all `RobotUnit`s, partitions the environment into corridors, runs the main loop, and draws every frame |
| 6 | Entry point — CLI arg parsing (`argparse`), obstacle/debris generation, kicks off `SwarmCoordinator.run()` |

### Key classes

- **`Vec3`** — simple x/y/z point with `dist()`. Used for positions, targets,
  and (confusingly) also obstacles, where `x, y` = center and `z` = radius.
- **`Segment` / `OctopusArm`** — a soft continuum-robotics model: each arm has
  10 segments, each driven by 3 tendons (120° apart) whose lengths determine
  a constant-curvature bend (`kappa`, `phi`). This is a real (simplified)
  soft-robotics kinematics model, not just decoration — `lasso_grip_force()`
  implements the **capstan equation** (`F = T·e^(μφ)`) to compute how much
  holding force the wrapped arm generates.
- **`RobotUnit`** — implements the mission as a state machine. Each phase is
  a `_ph_<name>` method; `step()` dispatches to the current phase's handler
  every simulation tick.
- **`SwarmCoordinator`** — divides the channel into `n_robots` vertical
  corridors, gives each robot a `ScanningCell` (an ABCD rectangle), runs the
  step loop, and renders frames via matplotlib.

### Mission state machine (`MissionPhase`)

```
IDLE → NAVIGATE_TO_ZONE → HOVER_STABILIZE → BOUSTROPHEDON_SCAN
     → DETECT_DEBRIS → QUAD_ARM_REACH → LASSO_COIL → GRIP_VERIFY
     → ARM_RETRACT → NAVIGATE_TO_BIN → EJECT_DEBRIS
     → (back to BOUSTROPHEDON_SCAN, loop until corridor is clear)
     → RETURN_TO_BASE → MISSION_COMPLETE
(FAULT can be entered from GRIP_VERIFY if lasso force is insufficient,
 and always transitions back to BOUSTROPHEDON_SCAN)
```

- **BOUSTROPHEDON_SCAN**: picks the left-most uncollected debris in the
  drone's corridor and flies straight to it (despite the name, it's currently
  "nearest-leftmost target," not a literal zig-zag sweep path).
- **QUAD_ARM_REACH → LASSO_COIL → GRIP_VERIFY**: all 4 arms spiral into a
  helix around the debris, tighten (increasing tendon tension over
  `coil_time` seconds), then the grip force is checked against
  `payload_kg * gravity`. If insufficient → `FAULT`.
- **NAVIGATE_TO_BIN → EJECT_DEBRIS**: carries the debris (visually dangling
  0.25 m below the drone) to the corridor's center "bin," then "unspools" the
  arms rapidly to drop it, and the waypoint is permanently placed at the bin
  location and marked collected.
- Anti-collision (`_apply_anti_collision`) runs **every step regardless of
  phase** (except IDLE/COMPLETE/FAULT) using APF: repulsion from other
  drones + repulsion-plus-tangential-bypass steering around static pillar
  obstacles.

## 3. Requirements

```
python3
numpy
matplotlib
```

Install with:
```bash
pip install numpy matplotlib
```

No GPU, ROS, or hardware SDK is required — this is pure simulation.

## 4. How to run

```bash
python3 dronemain.py [options]
```

### CLI options (`argparse`)

| Flag | Default | Meaning |
|---|---|---|
| `--n` | `3` | Number of drones in the swarm (each gets its own corridor) |
| `--targets` | `10` | Number of debris waypoints generated **per corridor** (total = `n * targets`) |
| `--width` | auto (`max(7.0, n*3.0)`) | Environment width in meters |
| `--height` | `2.0` | Environment height (channel width) in meters |
| `--fov` | `"inf"` | Camera FOV radius in meters, or `"inf"` for unlimited detection range |
| `--step-size` | `0.0` (auto) | Boustrophedon scan step size; `0.0` lets `MissionConfig.resolve_step_size()` pick one |
| `--payload` | `0.8` | Target payload weight (kg) the grip force must support |
| `--headless` | off | Force headless mode (skip live GUI, save PNGs). Auto-enabled if no `DISPLAY`/`WAYLAND_DISPLAY` is detected (e.g. SSH, Docker, Jetson Nano) |

### Examples

```bash
# Default: 3 drones, 10 targets each, live GUI if a display exists
python3 dronemain.py

# 5 drones, 6 targets each, force headless (saves PNGs instead of a window)
python3 dronemain.py --n 5 --targets 6 --headless

# Limit the simulated camera's field of view to 0.35 m
python3 dronemain.py --fov 0.35
```

### Output

- **With a display**: opens an interactive matplotlib window animating the
  swarm live, then shows a final flight-trajectory plot.
- **Headless** (no display, or `--headless`): saves two PNGs to the working
  directory:
  - `simulation_final_state.png` — last frame of the swarm animation
  - `flight_trajectories.png` — X vs. altitude(Z) path plot for every drone

Console output logs every phase transition per robot (`[R0][BOUSTROPHEDON_SCAN] → DETECT_DEBRIS`, etc.) and a final summary of items collected per robot and total.

## 5. ⚠️ Known issues in the uploaded file (fix before running)

The file **as uploaded does not run as-is** — two bugs were confirmed by
executing it:

1. **Line 1 has a leading space before `import math`**, which raises
   `IndentationError: unexpected indent` immediately. Fix: remove the leading
   whitespace on line 1 (must be `import math`, not ` import math`).
2. **Forward-reference / definition-order bug**: `EnvConfig` (around line 41)
   declares `obstacles: List[Vec3] = field(default_factory=list)`, but the
   `Vec3` class isn't defined until much later (~line 128), in a file that
   does *not* have `from __future__ import annotations`. Python evaluates
   dataclass field annotations immediately, so this raises
   `NameError: name 'Vec3' is not defined` as soon as `EnvConfig` is defined.
   Fix: either move the `Vec3` class definition above `EnvConfig`, or add
   `from __future__ import annotations` as the first import in the file.

Both were reproduced directly:
```
NameError: name 'Vec3' is not defined
```
at `class EnvConfig: ... obstacles: List[Vec3] = field(...)`.

Once those two fixes are made, the script imports and runs correctly (tested
headless with `--n 1 --targets 2`).

### Other things worth knowing (not bugs, but sharp edges)

- `import sys` appears twice (lines 6 and 25) — harmless but redundant.
- `MissionConfig.reach_time` is defined but never referenced anywhere in the
  simulation logic (dead config field).
- `_assign_tasks()` in `SwarmCoordinator` is a no-op (`pass`) — task
  assignment is actually static, done once in `__init__` via corridor
  partitioning, not dynamic.
- The "boustrophedon" (zig-zag) scan is really "always fly to the nearest
  uncollected leftmost target in my corridor" — there's no actual up/down
  lane-sweeping motion coded, despite `scan_dir`/`scan_x` state fields
  existing (they're set but not used to alter the flight path).
- Obstacles are stored as `Vec3(x, y, radius)` — reusing a position type to
  also mean "circle," so `obs.z` means radius, not altitude, in that context.
- `Segment` objects hold their own `l_tendons` state and are mutated in place
  by controllers (`BendPropagationController`, and the `_ph_*` methods
  directly) — there's no clean separation between "kinematics model" and
  "actuator driver."

## 6. Suggested uses for this file going forward

- Treat it as a **research/prototyping sandbox** for swarm coordination
  strategies (task allocation, collision avoidance, search patterns) and for
  soft-robotic tendon-driven gripper kinematics — independently, since the
  two halves (flight/swarm logic vs. arm kinematics) are only loosely coupled
  through `RobotUnit`.
- If handing this to another AI assistant for further work, point it at:
  - `MissionPhase` + the `_ph_*` methods in `RobotUnit` for mission-logic
    changes,
  - `SwarmCoordinator.__init__` for changing how corridors/zones are
    assigned,
  - `_apply_anti_collision` for collision-avoidance tuning,
  - `SimConfig` and its nested dataclasses for any parameter tuning (avoid
    scattering magic numbers elsewhere in the file).
- Apply the two fixes in Section 5 first — any assistant asked to "run" or
  "extend" this file will hit them immediately otherwise.
