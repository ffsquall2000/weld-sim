# 超声波焊接 FEA 增强系统 -- 统一设计文档

> **日期**: 2026-02-27
> **状态**: 已批准
> **目标**: 工程验证级 FEA 系统，替代 ANSYS/COMSOL 日常焊头/变幅杆设计工作流
> **部署**: 服务器端计算 (180.152.71.166:8001)，结果通过 REST API 返回前端
> **精度目标**: 频率误差 <1%, 应力误差 <5%
> **频率范围**: 15--40 kHz 全覆盖
> **几何输入**: 参数化生成 + STEP/IGES 导入

---

## 0. 架构总览

### 0.1 混合求解器架构 (Solver A + Solver B)

```
前端 (浏览器)
  │
  │  REST API (JSON)
  ▼
FastAPI 服务 (uvicorn, port 8001)
  │
  ├── 同步路径 (<5K 节点): 直接返回结果
  │
  └── 异步路径 (≥5K 节点): 返回 job_id → 轮询/WebSocket
        │
        ▼
  ┌─────────────────────────────────┐
  │       SolverDispatcher          │
  │  (solver_backend.py)            │
  ├─────────────┬───────────────────┤
  │  Solver A   │    Solver B       │
  │ numpy/scipy │  FEniCSx/DOLFINx  │
  │  本地进程    │   Docker 容器     │
  │  日常分析    │   高级物理/验证    │
  └─────────────┴───────────────────┘
        │               │
        └───────┬───────┘
                ▼
        交叉验证 (可选)
        MAC > 0.95, Δf < 1%
```

### 0.2 部署架构

所有 FEA 计算在服务器端执行，前端不做任何计算：

```
服务器 (180.152.71.166)
├── weld-sim.service (uvicorn, port 8001)
│   ├── FastAPI 应用 (web/app.py)
│   ├── FEA 计算引擎 (Solver A, 本进程内)
│   ├── 异步任务队列 (asyncio + ProcessPoolExecutor)
│   └── 结果缓存 (SQLite)
│
├── fenicsx-worker (Docker, 可选)
│   ├── dolfinx:v0.8.0 镜像
│   ├── FastAPI 内部 API (port 8002, 仅 localhost)
│   └── PETSc + SLEPc + MUMPS
│
├── Gmsh (系统安装, Python API)
├── CadQuery (pip, 可选)
└── 数据存储
    ├── /opt/weld-sim/data/components/   # 组件定义 JSON
    ├── /opt/weld-sim/data/meshes/       # 缓存的网格文件
    ├── /opt/weld-sim/data/results/      # 分析结果
    └── /opt/weld-sim/data/materials/    # 材料 YAML 文件
```

**前端交互流程:**
1. 前端发送分析请求 → `POST /api/v1/analysis/modal`
2. 小模型 (<5K节点): 服务器同步计算, 直接返回 JSON 结果
3. 大模型 (≥5K节点): 返回 `{job_id}`, 前端轮询 `GET /api/v1/jobs/{job_id}`
4. 前端渲染: 3D 模态动画, 应力云图, 频率响应曲线, 疲劳报告

### 0.3 分析能力矩阵

| 分析类型 | Solver A | Solver B | 适用场景 |
|---------|----------|----------|---------|
| 模态分析 | ✅ | ✅ | 所有: 焊头/变幅杆/装配体 |
| 谐响应分析 | ✅ | ✅ | 振幅分布, 频率扫描 |
| 静力分析 | ✅ | ✅ | 螺栓预紧, 重力 |
| 疲劳评估 | ✅ | -- | S-N, Goodman, 安全系数 |
| 压电耦合 | -- | ✅ | 换能器阻抗, k_eff |
| 非线性接触 | -- | ✅ | 螺栓连接, 预应力模态 |
| 热机耦合 | -- | ✅ | 热漂移, 热失控预测 |
| 交叉验证 | A+B 联合 | A+B 联合 | 关键设计确认 |

---

## 1. 几何与网格层

> **详细设计**: [geometry-meshing-design.md](./2026-02-27-geometry-meshing-design.md) (1,100 行)

### 1.1 参数化几何生成器

**文件**: `ultrasonic_weld_master/plugins/geometry_analyzer/parametric_geometry.py` (~1,400 LOC)

支持组件类型:
- **焊头**: 7种 (flat, cylindrical, exponential, catenoidal, blade, stepped, block)
- **变幅杆**: 4种 (stepped, exponential, catenoidal, uniform)
- **换能器**: back mass + PZT stack + front mass

半波长自动计算: `L_half = c_bar / (2*f)`, 其中 `c_bar = sqrt(E/rho)`

特征支持: 槽缝 (through/blind/tapered), 螺纹连接 (简化为光滑埋头孔), 圆角/倒角

装配体构建: 沿共享轴线平移组件, 重合配合面

### 1.2 CAD 导入管线

**文件**: `cad_import.py` (~650 LOC)

- STEP/IGES 导入 (CadQuery/OCP, 降级到文本解析器)
- 几何验证: 水密检查, 自交检测, 退化面修复
- 特征识别: 20截面切片 → 面积变化分析 → 自动分类 (焊头/变幅杆/换能器)
- 单位检测: 基于包围盒尺度的启发式判断

### 1.3 Gmsh 网格策略

**文件**: `gmsh_mesher.py` (~1,100 LOC)

| 项目 | 规格 |
|------|------|
| 单元类型 | HEX20 (简单/参数化) / TET10 (开槽/导入) |
| 网格密度 | ≥6 二阶单元/波长, λ_min = c_bar / (2*f_operating) |
| 自适应细化 | 7种区域: 几何过渡/圆角/槽端/螺纹根/接触面/安装环/装配界面 |
| 质量控制 | 最小纵横比 ≥0.2, 最小 Jacobian ≥0.3 |
| 收敛研究 | 4级 (4/6/8/12 单元/波长), 收敛于 <0.1% 频率变化 |
| 装配网格 | 共形 (Gmsh fragment, 共享节点) 或 绑定接触 (KD-tree 配对) |

### 1.4 边界面识别

**文件**: `boundary_identifier.py` (~500 LOC)

8种边界角色: `input`, `output`, `mounting_ring`, `thread_bore`, `lateral`, `symmetry_xz`, `symmetry_yz`, `interface`

自动检测算法: 主轴对齐 (dot product >0.95) + 轴向位置 + 对称性反射测试

---

## 2. 核心求解器 A (numpy/scipy)

> **详细设计**: [fea-core-solver-a-design.md](./2026-02-27-fea-core-solver-a-design.md) (~5,550 LOC)

### 2.1 模块布局

```
fea/
├── mesh_io.py              # Gmsh .msh 读取                    400 LOC
├── elements.py             # TET10, HEX20 形函数 + 积分         900 LOC
├── assembly.py             # 全局 K, M 装配                     500 LOC
├── boundary_conditions.py  # BC: free-free, clamped, MPC        350 LOC
├── modal_solver.py         # 模态分析 (特征值)                   600 LOC
├── harmonic_solver.py      # 谐响应分析                         550 LOC
├── static_solver.py        # 线性静力                           400 LOC
├── fatigue.py              # 疲劳/寿命评估                       450 LOC
├── assembly_coupling.py    # 多体耦合                           500 LOC
├── postprocess.py          # 后处理                             500 LOC
├── solver_config.py        # 配置数据类                         200 LOC
└── utils.py                # 工具函数                           200 LOC
```

### 2.2 核心求解能力

**模态分析**: `[K - ω²M]φ = 0`
- `scipy.sparse.linalg.eigsh`, shift-invert at σ = (2πf_target)²
- 自由-自由: 丢弃 f < 10 Hz 的6个刚体模态
- 模态分类: 纵向/弯曲/扭转 (基于位移分量参与因子)
- 寄生模态检测: 频率间距 ≥500 Hz @20kHz

**谐响应**: `[K - ω²M + jωC]u = F`
- Rayleigh 阻尼: C = αM + βK, 从 Q 因子推导
- 频率扫描: f_target ± 5%, 200+ 频率点
- 输出: 振幅分布, 均匀性 U = min_avg/max_avg

**静力分析**: `Ku = F`
- 螺栓预紧力 (分布面力)
- 几何刚度矩阵 K_σ → 预应力模态

**装配体耦合**:
- Bonded: 共享 DOF (界面节点合并)
- Tied contact: MPC 约束 (主从节点配对)
- 支持: 焊头+变幅杆, 焊头+变幅杆+换能器

---

## 3. FEniCSx 插件 B

> **详细设计**: [fenicsx-plugin-design.md](./2026-02-27-fenicsx-plugin-design.md) (2,082 行)

### 3.1 插件架构

`SolverBackend` 抽象基类统一两个求解器:
- `NumpyScipyBackend` -- 包装现有 Solver A
- `FEniCSxBackend` -- 懒导入, FEniCSx 不可用时优雅降级
- `SolverDispatcher` -- 根据分析类型路由请求

`mesh_bridge.py` -- Gmsh 模型同时转换为 numpy 数组和 DOLFINx 网格

### 3.2 高级物理

| 功能 | 方程/方法 |
|------|----------|
| 压电耦合 | σ = cᴱε - eᵀE, D = eε + εˢE; 混合有限元 (P1向量+P1标量) |
| 阻抗扫描 | 频率扫描 → f_r, f_a, k_eff = √(1-(f_r/f_a)²) |
| 非线性接触 | 增广拉格朗日 + Coulomb 摩擦, Newton 求解器 + MUMPS |
| 预应力模态 | 收敛切线刚度 → SLEPc shift-invert 特征值提取 |
| 热机耦合 | 滞后发热 + 摩擦发热, BDF2 时间积分, 频率温漂 |

### 3.3 交叉验证

```
MAC_ij = |φ_A_i^T · φ_B_j|² / ((φ_A_i^T · φ_A_i)(φ_B_j^T · φ_B_j))
```

| 指标 | PASS | WARNING | FAIL |
|------|------|---------|------|
| MAC 对角线 | ≥0.95 | 0.90--0.95 | <0.90 |
| 频率偏差 | <0.5% | 0.5--1.0% | >1.0% |
| 应力峰值偏差 | <5% | 5--10% | >10% |

### 3.4 Docker 部署

```dockerfile
FROM dolfinx/dolfinx:v0.8.0
# 内部 API: port 8002, 仅 localhost 访问
# 主服务通过 FEniCSxClient HTTP 调用
```

---

## 4. 材料数据库与疲劳评估

> **详细设计**: [material_database_fatigue_assessment.md](../design/material_database_fatigue_assessment.md) (2,768 行)

### 4.1 增强材料模式

Pydantic 验证模型层次: `ElasticProperties` → `AcousticProperties` → `ThermalProperties` → `StrengthProperties` → `SNcurve` → `DampingModel` → `PiezoelectricProperties` → `FEAMaterial`

向后兼容: `flatten_to_legacy()` 输出现有 `FEA_MATERIALS` 格式

### 4.2 材料数据库 (14种材料)

| 材料 | E (GPa) | ρ (kg/m³) | Q | σ_e @10⁹ (MPa) | 用途 |
|------|---------|-----------|---|-----------------|------|
| Ti-6Al-4V (退火) | 113.8 | 4430 | 5000 | 350 | 焊头首选 |
| Ti-6Al-4V (STA) | 114 | 4430 | 5000 | -- | 高强度焊头 |
| Al 7075-T6 | 71.7 | 2810 | 9000 | 105 | 轻型焊头 |
| D2 Tool Steel | 210 | 7700 | 3000 | -- | 耐磨焊头 |
| M2 HSS | 220 | 8160 | 3000 | -- | 高速钢焊头 |
| CPM 10V | 210 | 7450 | 2500 | -- | 粉末冶金 |
| PM60 | 215 | 7850 | 2500 | -- | 粉末冶金 |
| HAP40 | 225 | 8000 | 2200 | -- | 高硬度 |
| HAP72 | 230 | 8100 | 2000 | 560 | 极高硬度 |
| Steel 4140 | 200 | 7850 | 3500 | -- | 后质量块/前质量块 |
| PZT-4 | -- | 7500 | Qm=500 | -- | 压电换能器 |
| PZT-8 | -- | 7600 | Qm=1000 | -- | 大功率压电 |
| Ferro-Titanit WFN | 290 | 6900 | 2000 | -- | 硬质合金/金属陶瓷 |
| CPM Rex M4 | 220 | 8000 | 2500 | -- | 粉末冶金 |

### 4.3 疲劳评估模块

5个核心组件:
1. **SNInterpolator** -- log-log S-N 曲线插值/外推
2. **StressConcentrationDB** -- 6种特征 Kt (圆角/沟槽/横孔/螺纹/槽/键槽), Peterson/Neuber Kf
3. **GoodmanDiagram** -- 4种平均应力理论 (Goodman/Gerber/Soderberg/Morrow)
4. **SafetyFactorCalculator** -- Marin 修正 (表面/尺寸/可靠性/温度) + 安全系数 ≥ 2.0
5. **rainflow_count()** -- ASTM E1049 雨流计数 + Palmgren-Miner 累积损伤

### 4.4 阻尼模型

关键设计: Ti-6Al-4V 在应变 >0.3% 时 Q 急剧下降 (`strain_threshold`)，这是焊头高振幅热失控的主要原因

三轴依赖: 频率 × 应变幅值 × 温度

### 4.5 数据存储

每种材料一个 YAML 文件 → `materials/` 目录 → 启动时 Pydantic 校验

---

## 5. API 端点与分析场景

> **详细设计**: [fea-api-endpoints-analysis-scenarios-design.md](./2026-02-27-fea-api-endpoints-analysis-scenarios-design.md)

### 5.1 端点概览 (50+)

| 模块 | 端点数 | 说明 |
|------|--------|------|
| 组件管理 (Horn/Booster/Transducer) | ~25 | CRUD + 生成 + 导入 + 下载 |
| 分析 (Modal/Harmonic/Static/Fatigue) | ~10 | 同步/异步双模式 |
| 装配体分析 | ~5 | 创建装配 + 装配分析 |
| 高级分析 (Piezo/Contact/Thermal) | ~5 | Solver B 专属 |
| 任务管理 | ~5 | 轮询/取消/结果下载 |

### 5.2 同步/异步双模式

```
POST /api/v1/analysis/modal
  ├── mesh nodes < 5000 → 同步: 直接返回 ModalResult JSON
  └── mesh nodes ≥ 5000 → 异步: 返回 {job_id, status: "queued"}
                                   → GET /api/v1/jobs/{job_id}
                                   → {status: "running", progress: 65}
                                   → {status: "completed", result: {...}}
```

### 5.3 三大分析场景

**场景 1: 独立焊头分析**
1. 创建/导入焊头几何
2. 模态分析 → 找到工作频率附近的纵向模态
3. 谐响应 → 振幅分布 + 均匀性
4. 应力分析 → 热点定位
5. 疲劳评估 → 安全系数 + 预期寿命

**场景 2: 独立变幅杆分析**
1. 创建变幅杆 (选型 + 增益)
2. 模态分析 → 确认谐振频率
3. 增益验证 → 输入/输出面振幅比
4. 应力分析 → 节面区域应力集中

**场景 3: 全装配体分析 (焊头+变幅杆+换能器)**
1. 创建各组件 + 装配
2. 装配体模态 → 系统谐振频率
3. 压电阻抗扫描 (Solver B) → f_r, f_a, k_eff
4. 耦合谐响应 → 端到端振幅传递
5. 交叉验证 (可选) → Solver A vs B 对比

---

## 6. 分阶段实施路线图

> **详细设计**: [fea-phased-implementation-roadmap.md](./2026-02-27-fea-phased-implementation-roadmap.md) (1,529 行)

### 6.1 阶段依赖

```
Phase 1 ────► Phase 2 ────► Phase 3 ────► Phase 4
(基础)        (模态)        (谐响应)       (应力/疲劳)
                                              │
Phase 1 ────► Phase 5 ◄──────────────────────┘
              (装配体)
                  │
                  ▼
              Phase 6 ────► Phase 7
              (FEniCSx)     (高级物理)
```

### 6.2 各阶段摘要

| 阶段 | 周次 | 目标 | 关键交付物 | 代码量 |
|------|------|------|-----------|--------|
| 1 基础 | 1-2 | Gmsh + TET10 + 材料库 | mesher.py, elements.py, 增强materials | ~2,050 |
| 2 模态 | 3-4 | 特征值求解 + 分类 | modal_solver.py, postprocess.py | ~1,800 |
| 3 谐响应 | 5-6 | 频率扫描 + 振幅 | harmonic_solver.py | ~1,600 |
| 4 应力/疲劳 | 7-8 | 静力 + S-N + 安全系数 | static_solver.py, fatigue.py | ~1,700 |
| 5 装配体 | 9-10 | 多体耦合 + 变幅杆 | assembly_coupling.py, booster params | ~2,100 |
| 6 FEniCSx | 11-12 | Docker + 压电 + 交叉验证 | Dockerfile, fenicsx_backend.py | ~2,200 |
| 7 高级 | 13-14 | 接触 + 热机 + 收敛 | contact.py, thermal.py | ~1,260 |
| **总计** | **14周** | | | **~12,710 新 + ~1,330 改** |

### 6.3 服务器部署计划

| 阶段 | 部署方式 | 新增依赖 |
|------|---------|---------|
| Phase 1-5 | `pip install` 到现有 venv | gmsh, cadquery (可选) |
| Phase 6+ | 新增 Docker 容器 | docker, dolfinx:v0.8.0 镜像 |
| 生产环境 | docker-compose 全栈 | weld-sim + fenicsx-worker |

每个阶段结束后可独立部署到服务器并投入使用，不需要等待全部完成。

### 6.4 验证标准

| 指标 | 目标 | 验证方法 |
|------|------|---------|
| 频率误差 | <1% | 对比 ANSYS 参考值 / 实测数据 |
| 应力误差 | <5% | 对比 ANSYS 参考值 |
| 振幅均匀性 | U >0.85 | 输出面节点统计 |
| 交叉验证 MAC | >0.95 | Solver A vs B 模态对比 |
| 求解时间 | <60s @50K 节点 | 模态分析基准测试 |

---

## 7. 详细设计文档索引

| # | 模块 | 文件 | 行数 |
|---|------|------|------|
| 1 | 几何与网格 | [geometry-meshing-design.md](./2026-02-27-geometry-meshing-design.md) | 1,100 |
| 2 | 核心求解器 A | [fea-core-solver-a-design.md](./2026-02-27-fea-core-solver-a-design.md) | ~2,000 |
| 3 | FEniCSx 插件 B | [fenicsx-plugin-design.md](./2026-02-27-fenicsx-plugin-design.md) | 2,082 |
| 4 | 材料数据库与疲劳 | [material_database_fatigue_assessment.md](../design/material_database_fatigue_assessment.md) | 2,768 |
| 5 | API 端点与场景 | [fea-api-endpoints-analysis-scenarios-design.md](./2026-02-27-fea-api-endpoints-analysis-scenarios-design.md) | ~2,000 |
| 6 | 实施路线图 | [fea-phased-implementation-roadmap.md](./2026-02-27-fea-phased-implementation-roadmap.md) | 1,529 |
