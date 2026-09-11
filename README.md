# AGT Map Reconstruction

**Offline LiDAR map reconstruction and navigation-map validation toolkit for agricultural / structured outdoor environments.**

**面向农业与结构化户外场景的离线 LiDAR 地图重建、可通行性分析与导航地图验证工具。**

[中文](#中文) · [English](#english)

---

<a id="中文"></a>

# 中文

## 1. 项目定位

AGT Map Reconstruction 是一个**独立于机器人导航运行时的离线地图实验与验证仓库**。

它面向 FAST-LIO2 / FAST-LIVO2 / 其他 LIO-SLAM 输出的 PCD 点云地图，研究如何把“几何上可视的三维点云”逐步转换成“适合移动机器人导航使用、并且能够被验证的二维地图资产”。

当前仓库重点解决：

- PCD 离线预处理；
- 地面分割 baseline 对比；
- 高程与相对高程建图；
- 几何可通行性分析；
- 农业行/走廊结构恢复；
- 语义静态障碍与候选障碍分层；
- Nav2 兼容 PGM/YAML 导出；
- 机器人真实多边形 footprint 验证；
- 行内横向偏移路线搜索；
- 平滑横向路线几何验证；
- 实验结果、失败案例和参数的可追踪记录。

> 本仓库**不是完整导航栈**。定位、实时局部感知、Nav2 Planner/Controller、Behavior Tree、车辆控制与在线避障应放在导航运行时仓库中。

## 2. 为什么存在这个仓库

对于农业、温室、林区、厂区等非平整场景，简单地把三维点云直接压成 PGM 往往会产生问题：坡地被误判为障碍、植被与立柱混杂、ground/non-ground 无法直接回答“车能不能通过”，以及手工修改 PGM 后失去地图来源和实验可解释性。

因此本项目采用逐级验证：

~~~text
3D PCD
  |
  v
Ground / terrain understanding
  |
  v
Elevation + relative elevation
  |
  v
Traversability reasoning
  |
  v
Agricultural row / corridor recovery
  |
  v
Semantic navigation layers
  |
  v
Nav2-compatible static map
  |
  v
Robot footprint validation
  |
  v
In-aisle route geometry validation
~~~

目标不是“尽快得到一张黑白图”，而是能够解释：地图为什么这样生成、哪些区域是真正静态障碍、机器人尺寸是否能通过，以及失败究竟来自地图、未知区域、走廊几何还是路径本身。

## 3. 当前能力状态

| 能力 | 状态 | 说明 |
|---|---|---|
| PCD 加载 | ✅ 已实现 | Open3D 离线加载 |
| PCD 离线清理 | ✅ 已实现 | voxel + radius outlier + optional cluster-size filter |
| Height Threshold | ✅ 已实现 | 简单地面分割 baseline |
| PMF-inspired baseline | ✅ 已实现 | 轻量近似基线，不等同于完整标准 PMF |
| CSF | 🧭 计划/待适配 | 当前 main 未实现 |
| Patchwork / Patchwork++ | 🧭 计划/待适配 | 当前 main 未实现 adapter |
| LineFit | 🧭 计划/待适配 | 早期设计项，当前 main 未实现 |
| Elevation / Relative Elevation | ✅ 已实现 | 离线高程分析 |
| Traversability Grid | ✅ 已实现 | 几何可通行性 baseline |
| Row Direction Estimation | ✅ 已实现 | 几何/PCA 方向估计 |
| Agricultural Corridor Recovery | ✅ 已实现 | 行/走廊结构恢复 |
| Navigation Map V2 | ✅ 已实现 | PGM/YAML + semantic masks |
| Polygon Footprint Validation | ✅ 已实现 | EXP004-A |
| Constant Lateral Offset Search | ✅ 已实现 | EXP004-B1 |
| Smooth Lateral Route Search | ✅ 已实现 | EXP004-B2 |
| Headland Handoff Validation | 🚧 下一阶段 | EXP004-C |
| Ackermann / turning-radius constraint | 🚧 下一阶段 | 需要真实车辆几何参数 |
| ROS 2 / Nav2 runtime | ❌ 不属于本仓库 | 由外部导航系统集成 |

## 4. 核心流水线

~~~text
PCD
 |
 +-- preprocessing
 |
 +-- ground / terrain baselines
 |       |- height threshold
 |       └- PMF-inspired baseline
 |
 +-- elevation / relative elevation
 |
 +-- traversability grid
 |
 +-- row direction + corridor recovery
 |
 +-- semantic navigation geometry
 |       |- aisle prior
 |       |- ridge / wall / pillar
 |       └- obstacle / step candidates
 |
 +-- navigation map v2
 |       |- navigation_base_map.pgm
 |       |- navigation_base_map.yaml
 |       |- candidate_mask.npy
 |       |- static_obstacle_mask.npy
 |       └- validation.json
 |
 +-- EXP004-A polygon footprint validation
 |
 +-- EXP004-B1 constant lateral-offset search
 |
 └-- EXP004-B2 smooth lateral route search
~~~

## 5. Navigation Map V2 语义策略

静态地图采用明确优先级，而不是把所有检测结果永久烧进 PGM。

**静态硬障碍：** ridge、wall、pillar。它们始终优先于 aisle/free prior，并进入 static_obstacle_mask。

**候选障碍：** obstacle candidate、step candidate。它们保留为 advisory candidate layer，后续可由局部感知、在线 costmap 或地图更新逻辑确认。

PGM 约定：

~~~text
0   = occupied
205 = unknown
254 = free
~~~

Nav2 YAML：

~~~yaml
mode: trinary
occupied_thresh: 0.65
free_thresh: 0.196
~~~

核心原则：

> 静态结构不能被走廊先验擦除；不确定障碍也不应在没有足够证据时永久写死。

## 6. 实验阶段

- **EXP001 — Ground Segmentation Benchmark**：验证简单地面分割，并确认 ground/non-ground 只是中间表达。
- **EXP002 — Agricultural Corridor Recovery**：恢复行方向、走廊和中心线。
- **EXP003 / EXP003.1 — Navigation Map V2**：生成 Nav2 静态地图，并修正 pillar 被 aisle prior 覆盖成 free 的安全问题。
- **EXP004-A — Polygon Footprint Validation**：用真实多边形 footprint 验证中心线。
- **EXP004-B1 — Constant Lateral Offset Search**：中心线失败时，在走廊内搜索平行偏移路线，而不是修改 PGM。
- **EXP004-B2 — Smooth Lateral Route**：允许横向偏移沿纵向平滑变化，并进行连续 footprint sweep。
- **EXP004-C — Headland Handoff & Kinematics**：下一阶段，处理行入口/出口、headland 交接和车辆运动学。

当前实验显示，大量剩余失败集中在**行入口/出口与 headland 的交接区域**，因此下一步应优先做 EXP004-C，而不是无限增加行内规划复杂度。

完整实验记录见 docs/EXPERIMENTS.md。

## 7. 快速开始

~~~bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
~~~

主要依赖：NumPy、Open3D、SciPy、OpenCV、Matplotlib、PyYAML。

EXP003/EXP004 是纯离线 Python 测试，不要求 ROS 2 runtime。若 shell 已 source ROS 2 Humble，建议：

~~~bash
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
~~~

## 8. 常用入口

### PCD 离线清理

历史 `agt_pointcloud_map_tools` 的有效需求已收敛到本仓库 preprocessing 层；旧 ROS 2 skeleton 不再作为独立运行时维护。在线 self-filter / crop / voxel / diagnostics 属于 `agt_pointcloud_pipeline`。

~~~bash
PYTHONPATH=src python tools/clean_pcd.py \
  --pcd /path/to/global_map.pcd \
  --config configs/preprocessing/outdoor_default.yaml \
  --output results/preprocessing/global_map_clean.pcd \
  --removed-output results/preprocessing/global_map_removed.pcd \
  --report results/preprocessing/cleaning_report.json
~~~

单张融合 PCD 不足以可靠判断人/车动态残影，因此本工具不会把“dynamic ghost cleaner”伪装成已实现能力；动态/静态判断应使用带时间证据的注册帧或专门 benchmark。

迁移审计见 [docs/MIGRATION_POINTCLOUD_MAP_TOOLS.md](docs/MIGRATION_POINTCLOUD_MAP_TOOLS.md)。

### 地面分割 baseline

~~~bash
PYTHONPATH=src python tools/run_benchmark.py --pcd /path/to/map.pcd

# 可选：先清理，再进入同一 benchmark
PYTHONPATH=src python tools/run_benchmark.py \
  --pcd /path/to/map.pcd \
  --preprocess-config configs/preprocessing/outdoor_default.yaml
~~~

当前入口默认运行 height-threshold baseline。PMF-inspired 以及未来 Patchwork++ 等算法建议后续统一接入 registry/benchmark runner。

### 生成 Navigation Map V2

~~~bash
python tools/build_navigation_map.py \
  --semantic-labels /path/to/semantic_labels.npy \
  --aisles /path/to/aisle_rectangles.json \
  --output results/EXP003/navigation-map-v2 \
  --resolution 0.05
~~~

### Polygon footprint 验证

~~~bash
python tools/validate_robot_footprint.py \
  --map-pgm results/EXP003/navigation-map-v2/navigation_base_map.pgm \
  --map-yaml results/EXP003/navigation-map-v2/navigation_base_map.yaml \
  --aisles /path/to/aisle_rectangles.json \
  --footprint /path/to/robot_footprint.json \
  --candidate-mask results/EXP003/navigation-map-v2/candidate_mask.npy \
  --output results/EXP004/robot-footprint-v1
~~~

Footprint 单位为米，位于机器人 base_link 坐标系：

~~~json
{
  "name": "my_robot",
  "polygon_xy_m": [
    [0.50, 0.30],
    [0.50, -0.30],
    [-0.50, -0.30],
    [-0.50, 0.30]
  ]
}
~~~

正式验收前请替换为真实车辆尺寸。

### 行内固定横向偏移

~~~bash
python tools/search_in_aisle_offsets.py \
  --map-pgm results/EXP003/navigation-map-v2/navigation_base_map.pgm \
  --map-yaml results/EXP003/navigation-map-v2/navigation_base_map.yaml \
  --aisles /path/to/aisle_rectangles.json \
  --footprint /path/to/robot_footprint.json \
  --candidate-mask results/EXP003/navigation-map-v2/candidate_mask.npy \
  --output results/EXP004/in-aisle-route-search-v1
~~~

### 平滑横向路线

~~~bash
python tools/search_smooth_lateral_routes.py \
  --map-pgm results/EXP003/navigation-map-v2/navigation_base_map.pgm \
  --map-yaml results/EXP003/navigation-map-v2/navigation_base_map.yaml \
  --aisles /path/to/aisle_rectangles.json \
  --footprint /path/to/robot_footprint.json \
  --candidate-mask results/EXP003/navigation-map-v2/candidate_mask.npy \
  --output results/EXP004/smooth-lateral-route-v1
~~~

## 9. 目录结构

~~~text
.
├── docs/
│   ├── EXPERIMENTS.md
│   ├── DEVELOPMENT_LOG.md
│   ├── benchmark_design.md
│   └── experiments/
├── src/agt_map_reconstruction/
│   ├── algorithms/
│   ├── preprocessing/
│   ├── io/
│   ├── maps/
│   └── visualization/
├── tests/
├── tools/
├── pyproject.toml
└── README.md
~~~

- src/：可复用算法与地图逻辑；
- tools/：离线实验和命令行入口；
- tests/：EXP003/EXP004 核心行为测试；
- docs/EXPERIMENTS.md：当前最权威的实验状态记录；
- results/：生成物，应视为实验输出而不是源代码。

## 10. 与导航系统的边界

本仓库负责离线地图推理、地图生成和地图资产验证；外部导航仓库负责 localization、local perception、costmaps、planner、controller、behavior tree 与 vehicle interface。

这样可以让地图算法独立 benchmark、复现和回归验证，同时避免主导航仓库持续膨胀。

## 11. 下一阶段建议

1. EXP004-C：Headland Handoff & Ackermann Transition；
2. 引入真实 wheelbase / steering / footprint / minimum turning radius；
3. 统一 ground segmentation plugin/registry 接口；
4. 接入 Patchwork++ 等生产级地面分割方法；
5. 增加 slope / roughness / local height variation 等 traversability 指标；
6. 定义稳定的 map-package 输出契约，供 MapManager / localization / Nav2 使用；
7. 增加 CI、依赖版本约束和 results 目录治理。

## 12. 设计原则

**地图不是为了让 planner 看起来能跑，而是为了保留可解释、可复现、可验证的几何与语义。**

- 不通过随意修改 PGM 来“修复”实验失败；
- 静态障碍与不确定候选障碍分层；
- 机器人真实 footprint 必须进入地图验收；
- 每次算法修改记录数据集、参数、输出、指标和失败案例；
- 离线地图生成与在线导航运行时保持解耦。

---

<a id="english"></a>

# English

## 1. Project Definition

AGT Map Reconstruction is an **offline LiDAR map reconstruction, traversability analysis, and navigation-map validation toolkit**, intentionally decoupled from the robot navigation runtime.

It processes PCD maps generated by FAST-LIO2, FAST-LIVO2, or other LIO-SLAM systems and converts 3D geometric maps into validated 2D navigation assets.

Current scope includes offline PCD processing, ground-segmentation baselines, elevation maps, geometry-based traversability analysis, agricultural corridor recovery, semantic obstacle separation, Nav2-compatible map export, physical robot-footprint validation, and offline in-aisle route validation.

> This repository is **not a complete navigation stack**. Localization, online perception, Nav2 planners/controllers, behavior trees, vehicle control, and runtime obstacle avoidance belong in external runtime repositories.

## 2. Motivation

Direct PCD-to-PGM projection is often insufficient in agricultural, greenhouse, forest, factory, or uneven outdoor environments. Slopes may become false obstacles, structural geometry may be mixed with vegetation, and ground/non-ground segmentation cannot directly answer whether the physical vehicle can pass.

The repository therefore uses staged validation:

~~~text
3D PCD
  |
  v
Terrain understanding
  |
  v
Elevation / relative elevation
  |
  v
Traversability
  |
  v
Row / corridor recovery
  |
  v
Semantic navigation layers
  |
  v
Nav2-compatible static map
  |
  v
Robot footprint validation
  |
  v
Route geometry validation
~~~

The goal is not simply to produce a black-and-white map, but to preserve provenance, interpretability, and robot-scale validation.

## 3. Implementation Status

| Capability | Status | Notes |
|---|---|---|
| PCD loading | ✅ Implemented | Offline Open3D loading |
| Height threshold | ✅ Implemented | Simple segmentation baseline |
| PMF-inspired baseline | ✅ Implemented | Lightweight approximation, not a canonical full PMF |
| CSF | 🧭 Planned | Not implemented on main |
| Patchwork / Patchwork++ | 🧭 Planned | Adapter not implemented on main |
| LineFit | 🧭 Planned | Historical design target |
| Elevation / relative elevation | ✅ Implemented | Offline terrain representation |
| Traversability grid | ✅ Implemented | Geometry-based baseline |
| Row direction estimation | ✅ Implemented | Geometric/PCA-based reasoning |
| Agricultural corridor recovery | ✅ Implemented | Row-aware corridor geometry |
| Navigation Map V2 | ✅ Implemented | PGM/YAML + semantic masks |
| Polygon footprint validation | ✅ Implemented | EXP004-A |
| Constant lateral-offset search | ✅ Implemented | EXP004-B1 |
| Smooth lateral route search | ✅ Implemented | EXP004-B2 |
| Headland handoff validation | 🚧 Next stage | EXP004-C |
| Ackermann / turning-radius constraints | 🚧 Next stage | Requires measured vehicle geometry |
| ROS 2 / Nav2 runtime | ❌ Out of scope | Integrate externally |

## 4. Navigation Map V2 Policy

Permanent structural geometry has higher priority than the aisle/free-space prior.

Hard static obstacles: ridge, wall, and pillar.

Advisory candidates: obstacle candidate and step candidate.

Static PGM values:

~~~text
0   = occupied
205 = unknown
254 = free
~~~

Nav2 map YAML:

~~~yaml
mode: trinary
occupied_thresh: 0.65
free_thresh: 0.196
~~~

> Permanent structures must never be erased by an aisle prior, while uncertain observations should not automatically become permanent map obstacles.

## 5. Experiment Progression

- **EXP001** — ground segmentation benchmark;
- **EXP002** — agricultural corridor recovery;
- **EXP003 / EXP003.1** — navigation-map interface and static-obstacle policy correction;
- **EXP004-A** — polygon footprint centerline validation;
- **EXP004-B1** — constant lateral-offset route search;
- **EXP004-B2** — smooth lateral route search;
- **EXP004-C** — planned headland handoff and vehicle-kinematics validation.

The latest experiments indicate that many unresolved failures occur at aisle entry/exit handoff boundaries rather than in the aisle interior. Headland transition validation is therefore a higher-priority next step than adding more in-aisle planner complexity.

See docs/EXPERIMENTS.md for the authoritative experiment record.

## 6. Quick Start

~~~bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
~~~

Main dependencies: NumPy, Open3D, SciPy, OpenCV, Matplotlib, and PyYAML.

~~~bash
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
~~~

The pytest plugin guard is useful when ROS 2 Humble has already been sourced and ROS testing plugins would otherwise be auto-loaded.

## 7. Common Commands

Ground-segmentation baseline:

~~~bash
python tools/run_benchmark.py --pcd /path/to/map.pcd
~~~

Build Navigation Map V2:

~~~bash
python tools/build_navigation_map.py \
  --semantic-labels /path/to/semantic_labels.npy \
  --aisles /path/to/aisle_rectangles.json \
  --output results/EXP003/navigation-map-v2 \
  --resolution 0.05
~~~

Validate a polygon footprint:

~~~bash
python tools/validate_robot_footprint.py \
  --map-pgm results/EXP003/navigation-map-v2/navigation_base_map.pgm \
  --map-yaml results/EXP003/navigation-map-v2/navigation_base_map.yaml \
  --aisles /path/to/aisle_rectangles.json \
  --footprint /path/to/robot_footprint.json \
  --candidate-mask results/EXP003/navigation-map-v2/candidate_mask.npy \
  --output results/EXP004/robot-footprint-v1
~~~

Search constant in-aisle offsets:

~~~bash
python tools/search_in_aisle_offsets.py \
  --map-pgm results/EXP003/navigation-map-v2/navigation_base_map.pgm \
  --map-yaml results/EXP003/navigation-map-v2/navigation_base_map.yaml \
  --aisles /path/to/aisle_rectangles.json \
  --footprint /path/to/robot_footprint.json \
  --candidate-mask results/EXP003/navigation-map-v2/candidate_mask.npy \
  --output results/EXP004/in-aisle-route-search-v1
~~~

Search smooth lateral routes:

~~~bash
python tools/search_smooth_lateral_routes.py \
  --map-pgm results/EXP003/navigation-map-v2/navigation_base_map.pgm \
  --map-yaml results/EXP003/navigation-map-v2/navigation_base_map.yaml \
  --aisles /path/to/aisle_rectangles.json \
  --footprint /path/to/robot_footprint.json \
  --candidate-mask results/EXP003/navigation-map-v2/candidate_mask.npy \
  --output results/EXP004/smooth-lateral-route-v1
~~~

## 8. Repository Layout

~~~text
.
├── docs/
│   ├── EXPERIMENTS.md
│   ├── DEVELOPMENT_LOG.md
│   ├── benchmark_design.md
│   └── experiments/
├── src/agt_map_reconstruction/
│   ├── algorithms/
│   ├── io/
│   ├── maps/
│   └── visualization/
├── tests/
├── tools/
├── pyproject.toml
└── README.md
~~~

- src/ contains reusable algorithm and map logic.
- tools/ contains offline experiment and CLI entry points.
- tests/ protects core EXP003/EXP004 behavior.
- docs/EXPERIMENTS.md is the authoritative experiment status record.
- results/ should be treated as generated experiment artifacts rather than source code.

## 9. Runtime Boundary

This repository owns offline reconstruction, semantic map reasoning, map export, and validation. A navigation runtime should own localization, local perception, costmaps, planner, controller, behavior tree, and vehicle interface.

Keeping this boundary explicit allows map algorithms to evolve and be benchmarked without continuously expanding the production navigation stack.

## 10. Roadmap

1. EXP004-C headland handoff and Ackermann transition validation;
2. introduce measured wheelbase, steering, footprint, and minimum-turning-radius constraints;
3. unify ground-segmentation plugins under one registry/benchmark interface;
4. integrate Patchwork++ or other production-grade segmentation methods;
5. extend traversability with slope, roughness, and local height variation;
6. define a stable map-package handoff contract for MapManager, localization, and Nav2;
7. add CI, dependency constraints, and generated-results repository hygiene.

## 11. Design Principles

- Do not edit the PGM merely to force a PASS.
- Keep permanent structural obstacles separate from uncertain candidates.
- Validate maps using the physical robot footprint.
- Record dataset, parameters, outputs, metrics, failures, and conclusions for every experiment.
- Keep offline map reconstruction decoupled from online navigation runtime.
