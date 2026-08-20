# Installation Guide

This guide explains how to install DonaNet and prepare the local environment for training, testing and inspecting YOLO-based mammal detection models for Doñana National Park.
## Prerequisites

Before installing DonaNet, make sure the following software is available on your system:

- Git
- Python 3.11 or newer
- A CUDA-capable NVIDIA GPU is strongly recommended for training
- CPU-only execution is supported, but training will be significantly slower

---

## 1. Clone the Repository

```bash
git clone https://github.com/wildintelproject/donanet.git
cd donanet
```

---

## Option A — uv (local environment)

This option runs DonaNet directly on your machine using a Python virtual environment managed by
[uv](https://docs.astral.sh/uv/).

### Run the setup script

The provided `setup.sh` script handles everything: it installs `uv` automatically if it is not
already present, creates the virtual environment and installs all required dependencies.

```bash
./setup.sh
```

### Activate the virtual environment

After setup, activate the virtual environment to access the `donanet` CLI:

```bash
source .venv/bin/activate
```

### Run DonaNet

With the environment activated, you can now run DonaNet commands. For example, to see the help

```bash
python donanet.py --help
```

---

## Option B — Docker (containerised environment)

This option runs DonaNet inside a Docker container based on
[`ultralytics/ultralytics`](https://hub.docker.com/r/ultralytics/ultralytics), which already
bundles PyTorch, CUDA and Ultralytics — no local GPU drivers or Python packages are required
beyond Docker itself.

The `configure.py` wizard generates a fully self-contained profile with a Docker Compose stack
and a set of ready-to-use helper scripts.

### Step 1 — Prepare the environment

The wizard depends on a small set of Python packages. Run setup first and activate the environment:

```bash
./setup.sh
source .venv/bin/activate
```

### Step 2 — Run the configuration wizard

```bash
python configure.py interactive
```

The wizard will guide you through the following sections:

#### Profile name
A profile is an independent, self-contained deployment. You can create multiple profiles
(e.g. `cpu-dev`, `gpu-production`) and switch between them freely. Each profile is stored
under `./profiles/<name>/`.

#### Timezone
Auto-detected from your system. Used to set the container's timezone correctly.

#### Storage paths
The three directories that are mounted as volumes inside the container:

| Host path   | Mounted at      | Purpose                                    |
|-------------|-----------------|--------------------------------------------|
| `./dataset` | `/app/dataset`  | Training images and YOLO labels            |
| `./weights` | `/app/weights`  | Model weight files (`best.pt`, `last.pt`)  |
| `./runs`    | `/app/runs`     | Ultralytics training artefacts and metrics |

Paths that do not yet exist are created automatically.

#### GPU support
Enables NVIDIA GPU acceleration via the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html).
When enabled, an additional `docker-compose.gpu.yml` overlay is generated.
Disable this option to run in CPU-only mode.

#### Resource limits
Sets the memory (RAM) and CPU core limits for the container. The wizard detects your system
resources and proposes sensible defaults (75 % of available RAM and CPU cores).

### Step 3 — What gets generated

All files are written to `./profiles/<profile-name>/`:

```
profiles/
└── <profile-name>/
    ├── .env                   ← environment variables (paths, timezone, resource limits)
    ├── docker-compose.yml     ← main Compose file
    ├── docker-compose.gpu.yml ← GPU overlay (only when GPU is enabled)
    ├── build.sh               ← builds the Docker image
    ├── train.sh               ← runs a training job
    ├── test.sh                ← runs inference / evaluation
    ├── prepare.sh             ← splits a raw dataset into train / val / test
    ├── info.sh                ← shows available weights and dataset summary
    ├── exec.sh                ← opens a shell (or runs a command) inside the container
    └── ps.sh                  ← lists running containers for this profile
```

### Helper scripts reference

| Script       | Purpose                                                | Example                                                                |
|--------------|--------------------------------------------------------|------------------------------------------------------------------------|
| `build.sh`   | Builds (or rebuilds) the Docker image for this profile | `./build.sh`                                                           |
| `train.sh`   | Launches a training run inside the container           | `./train.sh --model yolov8n.pt --epochs 100 --name my_run`             |
| `test.sh`    | Runs inference or evaluation on the test partition     | `./test.sh --weights weights/my_run/best.pt --conf 0.25`               |
| `prepare.sh` | Splits raw images + labels into `train / val / test`   | `./prepare.sh --source /path/to/raw --train 0.7 --val 0.2 --test 0.1` |
| `info.sh`    | Displays available weights and dataset statistics      | `./info.sh`                                                            |
| `exec.sh`    | Opens a bash shell inside the container for debugging  | `./exec.sh`                                                            |
| `ps.sh`      | Shows the status of the containers in this profile     | `./ps.sh`                                                              |

### Step 4 — Build and run

```bash
# Build the Docker image
./profiles/<profile-name>/build.sh

# Train
./profiles/<profile-name>/train.sh --model yolov8n.pt --epochs 100 --name my_run

# Inference
./profiles/<profile-name>/test.sh --weights weights/my_run/best.pt
```

---

## Directory Layout

```
donanet/
├── donanet.py          ← CLI entry point
├── configure.py        ← Docker configuration wizard
├── setup.sh            ← environment setup script
├── Dockerfile
├── pyproject.toml
├── uv.lock             ← locked dependency versions
├── profiles/           ← generated Docker profiles (one directory per profile)
├── dataset/ 
│ ├── images/ 
│ │ ├── train/ 
│ │ ├── val/ 
│ │ └── test/ 
│ ├── labels/ 
│ │ ├── train/ 
│ │ ├── val/ 
│ │ └── test/ 
│ ├── data.yaml         ← YOLO dataset configuration file 
│ └── annotations.csv   ← original annotation metadata 
├── output/ 
│   ├── train/          ← own training outputs, metrics, plots and weights 
│   └── test/           ← own testing outputs, predictions and evaluation summaries
├── weights/            ← pretrained donanet_weights.pt
├── runs/               ← storage for testing donanet_weights.pt
└── docs/
```
Training outputs are saved under:

```text
output/train/YYYYMMDD/

The main trained weights are saved as:

```text
output/train/YYYYMMDD/weights/best.pt
output/train/YYYYMMDD/weights/last.pt

Testing outputs are saved under:

```text
output/test/predictions_YYYYMMDD_HHMM/