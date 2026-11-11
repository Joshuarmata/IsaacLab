# Live run on this machine (no Isaac Sim)

## Applied

Published on `bag-lift-stagger`:

- `handle_grab/agents/__init__.py` skips the camera-network registration when `rl_games` is not installed. `uv run python -m isaaclab_tasks.core.official_handle_grab.real.deploy` imports without `--extra rl-games`. Training with the extra still registers the network.
- The README driver install is `uv pip install feetech-servo-sdk pyrealsense2 onnxruntime`.
- `policy.onnx` and `policy.onnx.data` are committed at the path below. `policy.pt` and the teacher checkpoint are not.
- `joint_map.json` is still not committed. Set `"verified": true` locally before `home`, as in the steps below.

## Handoff

Handoff for the other machine. The robot computer is the Pop!_OS box (`~/IsaacLab`). Pull `bag-lift-stagger` and rerun the commands below; the camera student does not start Isaac Sim. The issues further down are what failed before this push. The fixes are in "Applied".

Run every command from the Isaac Lab root.

## What this machine is trying to do

`calibrate`, `home`, and `deploy` do not start Isaac Sim. The live loop is 30 Hz: one D415 frame and seven joint angles in, `policy.onnx` through ONNX Runtime, seven joint targets out. `twin --viz kit` and `check_bridge` are the commands that launch the simulator. Skip both on this machine.

The student named in the README is:

```text
logs/rsl_rl/handle_grab_camera_distillation/2026-11-10_17-05-25/exported/policy.onnx
logs/rsl_rl/handle_grab_camera_distillation/2026-11-10_17-05-25/exported/policy.onnx.data
```

`policy.onnx.data` must sit next to `policy.onnx`. The weights are in the `.data` file.

## Commands to run here, once the weights exist

Install the hardware and runtime packages into this checkout's `.venv` (already created by an earlier `uv run`):

```bash
uv pip install feetech-servo-sdk pyrealsense2 onnxruntime
```

Every `python -m isaaclab_tasks...` command on this machine needs the `rl-games` extra. See issue 1.

```bash
uv run --extra rl-games python -m isaaclab_tasks.core.official_handle_grab.real.deploy \
    --policy logs/rsl_rl/handle_grab_camera_distillation/2026-11-10_17-05-25/exported/policy.onnx
```

That is a dry run. It reads the camera and the servo bus, runs the policy, and prints targets. It does not enable torque.

`home` and `deploy --execute` refuse the map until `"verified"` is true. `twin` is what normally sets that, and `twin` starts Isaac Sim, so this machine will not run it. The map is already calibrated (home counts are filled in, every `"sign"` is `1`, `"verified"` is `false`). After the signs are accepted, mark it verified:

```bash
python -c "
import json
from pathlib import Path
p = Path('source/isaaclab_tasks/isaaclab_tasks/core/official_handle_grab/real/joint_map.json')
data = json.loads(p.read_text())
data['verified'] = True
p.write_text(json.dumps(data, indent=2) + '\n')
"
```

`joint_map.json` is local calibration. Do not commit it.

Drive to the simulated home at 60% torque. If a joint turns the wrong way, cut power, flip that joint's `"sign"` between `1` and `-1`, and run `home` again.

```bash
uv run --extra rl-games python -m isaaclab_tasks.core.official_handle_grab.real.home
```

One 20 s episode. Arm torque is capped at 60%, gripper at 30%. Ctrl-C holds the arm where it is. The arm must be within 0.15 rad of home or `--execute` refuses to start.

```bash
uv run --extra rl-games python -m isaaclab_tasks.core.official_handle_grab.real.deploy \
    --policy logs/rsl_rl/handle_grab_camera_distillation/2026-11-10_17-05-25/exported/policy.onnx \
    --execute --log episode.npz
```

Servo bus is `/dev/ttyACM0`, motor ids `1 2 3 4 5 6 7` in order `revolute_1` .. `revolute_7`, gripper last. The user must be in the `dialout` group (`sudo usermod -aG dialout $USER`, then log out and back in).

## Issues

### 1. `uv run python` cannot import the real scripts

First `calibrate` and `twin` attempts:

```text
File "isaaclab_tasks/core/handle_grab/agents/camera_net.py", line 20, in <module>
    from rl_games.algos_torch import network_builder
ModuleNotFoundError: No module named 'rl_games'
```

`uv run python` created `.venv` with the default dependencies. `rl_games` is the `rl-games` extra, not a default dependency. Importing any `isaaclab_tasks` module walks every task package, and `handle_grab/agents/__init__.py` imports `camera_net.py`, which imports `rl_games` at import time. The real-arm scripts never use `rl_games`. `rsl_rl` was already installed.

Workaround that got past this on this machine: prefix every command with `uv run --extra rl-games`. The README's "Sim to real" commands omit that prefix, so they fail on a fresh environment.

Fix to publish: either make that `rl_games` import lazy so `calibrate`, `home`, and `deploy` import without the extra, or change the README commands to `uv run --extra rl-games`.

### 2. Hardware packages were missing from the fresh venv

At the time of the `rl_games` failure, this environment had:

| package | present |
| --- | --- |
| `rl_games` | no |
| `rsl_rl` | yes |
| `scservo_sdk` (`feetech-servo-sdk`) | no |
| `pyrealsense2` | no |
| `onnxruntime` | no |

`onnxruntime` is imported by `real/bridge.py` `OnnxStudent` and is not in `pyproject.toml` (only `onnx` is). The README's driver line installs `feetech-servo-sdk` and `pyrealsense2` and does not mention `onnxruntime`.

`deploy` later reached `onnxruntime.InferenceSession`, so `onnxruntime` is installed now. The camera and the servo bus were not opened on that attempt, because loading the model failed first. Confirm `feetech-servo-sdk` and `pyrealsense2` before the next dry run.

Fix to publish: add `onnxruntime` to the documented driver install.

### 3. The student weights are not on this machine

Latest command:

```bash
uv run --extra rl-games python -m isaaclab_tasks.core.official_handle_grab.real.deploy \
    --policy logs/rsl_rl/handle_grab_camera_distillation/2026-11-10_17-05-25/exported/policy.onnx
```

Result:

```text
onnxruntime.capi.onnxruntime_pybind11_state.NoSuchFile: [ONNXRuntimeError] : 3 : NO_SUCHFILE :
Load model from logs/rsl_rl/handle_grab_camera_distillation/2026-11-10_17-05-25/exported/policy.onnx failed. File doesn't exist
```

There is no `policy.onnx` under this home directory. `.gitignore` line `**/logs/*` ignores that path, so `cc190c4d1` did not contain the export. The code path itself got as far as opening the model.

Fix to publish: put both of these in the repo and push this branch.

```text
logs/rsl_rl/handle_grab_camera_distillation/2026-11-10_17-05-25/exported/policy.onnx
logs/rsl_rl/handle_grab_camera_distillation/2026-11-10_17-05-25/exported/policy.onnx.data
```

`git add -f` is required, because `**/logs/*` ignores them. If those files are large, a non-ignored path is fine as long as `LIVE_RUN.md` and the README name the path this machine should pass to `--policy`.

The teacher checkpoint named in the README is also absent here, and is only needed to retrain:

```text
logs/rsl_rl/handle_grab_state/2026-11-10_16-22-41/model_400.pt
```

Do not retrain for this handoff. This machine is doing a first run of the existing student.

### 4. Joint map is calibrated and not verified

`source/isaaclab_tasks/isaaclab_tasks/core/official_handle_grab/real/joint_map.json` exists only on this machine (untracked). Port `/dev/ttyACM0`, motor ids 1 through 7, home counts filled in, every sign `+1`, `"verified": false`.

`home.py` and `deploy --execute` exit until `"verified"` is true. The check that normally sets it is `twin --viz kit`, which this machine is not running. Leave the file untracked. This machine will set `"verified": true` locally before `home`.

## What not to change

Do not commit `joint_map.json`. Do not require Isaac Sim for `deploy`. The other machine should publish the two ONNX files and the import or documentation fixes above, then push `bag-lift-stagger`.
