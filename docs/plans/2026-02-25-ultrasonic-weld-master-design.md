# UltrasonicWeldMaster 设计文档

> **版本**: v1.0
> **日期**: 2026-02-25
> **状态**: 已批准
> **技术栈**: Python 3.11+ / PySide6 / SQLite
> **架构**: 插件式微内核
> **目标平台**: macOS (MacBook 独立运行)

---

## 1. 产品定位

**UltrasonicWeldMaster** 是一款面向超声波金属焊接应用工程师和研发工程师的专业参数生成与试验报告软件。

### 1.1 核心功能

1. **焊接参数自动生成** — 根据材料、工装、焊接面积等输入，基于物理模型和工艺知识库自动推荐焊接参数
2. **多专家校验机制** — 生成的参数经过物理合理性、工艺安全性、一致性三重校验
3. **试验报告生成** — 输出 PDF / Excel / JSON 三种格式的专业报告
4. **全操作日志** — 每次运行的输入/输出/中间过程全部记录，作为 AI 自适应系统的数据底座
5. **插件可扩展** — 新应用场景、新算法、新报告模板均可通过插件机制添加

### 1.2 目标用户

| 角色 | 使用方式 | 关注点 |
|------|---------|--------|
| 应用工程师 | GUI 操作 | 快速获取推荐参数、生成报告交付客户 |
| 研发工程师 | GUI + 配置文件 + API | 自定义算法、调试参数模型、分析历史数据 |

### 1.3 支持的应用场景

**第一版 (v1.0):**
- 锂电池极耳焊接 (Al-Al, Cu-Cu, Cu-Ni, Cu-Al)
- 锂电池 Busbar 焊接
- 锂电池集流盘焊接
- 通用金属超声波焊接 (自定义材料组合)

---

## 2. 系统架构

### 2.1 插件式微内核架构

```
┌────────────────────── GUI Shell (PySide6) ──────────────────────┐
│  主窗口 | 项目管理 | 参数配置面板 | 结果可视化 | 报告预览        │
└────────────────────────────┬────────────────────────────────────┘
                             | Qt Signal/Slot
┌────────────────────────────┴────────────────────────────────────┐
│                    核心引擎 (Core Engine)                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │ 插件管理器 │ │ 事件总线  │ │ 日志系统  │ │ 数据仓库(SQLite) │   │
│  │ PluginMgr│ │ EventBus │ │ Logger   │ │ DataWarehouse    │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             | Plugin API (标准接口)
┌────────────────────────────┴────────────────────────────────────┐
│                        插件层 (Plugins)                           │
│                                                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐   │
│  │ 锂电池参数引擎    │  │ 通用金属焊接引擎  │  │ 报告生成器      │   │
│  │ (li_battery)    │  │ (general_metal) │  │ (reporter)     │   │
│  └─────────────────┘  └─────────────────┘  └────────────────┘   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐   │
│  │ 材料数据库插件    │  │ 工艺知识库插件    │  │ AI自适应插件    │   │
│  │ (material_db)   │  │ (knowledge)     │  │ (ai_adaptive)  │   │
│  └─────────────────┘  └─────────────────┘  └────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 核心设计原则

1. **微内核只做 4 件事**: 插件生命周期管理、事件分发、日志记录、数据持久化
2. **所有业务逻辑在插件中**: 参数计算、报告生成、AI 推理全部是插件
3. **插件间通过事件总线通信**: 解耦，任何插件可被替换或禁用
4. **每次操作产生结构化日志**: 写入 SQLite + JSON 文件，为未来 AI 训练做数据底座
5. **GUI Shell 与 Core 分离**: Core 可以无 GUI 运行（命令行模式）

### 2.3 目录结构

```
ultrasonic_weld_master/
├── core/                    # 微内核
│   ├── __init__.py
│   ├── engine.py            # 核心引擎（启动、插件管理）
│   ├── plugin_api.py        # 插件标准接口定义 (ABC)
│   ├── event_bus.py         # 事件总线（发布-订阅）
│   ├── logger.py            # 结构化日志系统
│   ├── database.py          # SQLite 数据仓库
│   └── config.py            # 全局配置管理 (YAML)
├── plugins/                 # 插件目录
│   ├── __init__.py
│   ├── li_battery/          # 锂电池参数引擎插件
│   │   ├── __init__.py
│   │   ├── plugin.py        # 插件入口
│   │   ├── calculator.py    # 参数计算核心
│   │   ├── physics.py       # 物理模型
│   │   ├── validators.py    # 参数校验器
│   │   └── config.yaml      # 插件配置
│   ├── general_metal/       # 通用金属焊接插件
│   │   ├── __init__.py
│   │   ├── plugin.py
│   │   ├── calculator.py
│   │   └── config.yaml
│   ├── material_db/         # 材料数据库插件
│   │   ├── __init__.py
│   │   ├── plugin.py
│   │   ├── materials.yaml   # 材料数据
│   │   └── properties.py    # 材料属性计算
│   ├── knowledge_base/      # 工艺知识库插件
│   │   ├── __init__.py
│   │   ├── plugin.py
│   │   └── rules/           # 工艺规则 YAML 文件
│   ├── reporter/            # 报告生成器插件
│   │   ├── __init__.py
│   │   ├── plugin.py
│   │   ├── pdf_generator.py
│   │   ├── excel_generator.py
│   │   ├── json_exporter.py
│   │   └── templates/       # 报告模板
│   └── ai_adaptive/         # AI自适应插件（v2.0）
│       ├── __init__.py
│       └── plugin.py
├── gui/                     # PySide6 GUI Shell
│   ├── __init__.py
│   ├── main_window.py       # 主窗口
│   ├── panels/              # 各功能面板
│   │   ├── project_panel.py
│   │   ├── input_wizard.py  # 分步输入向导
│   │   ├── result_panel.py  # 结果展示
│   │   ├── report_panel.py  # 报告预览
│   │   ├── history_panel.py # 历史记录
│   │   └── settings_panel.py
│   ├── widgets/             # 自定义控件
│   │   ├── material_selector.py
│   │   ├── sonotrode_editor.py
│   │   └── chart_widget.py
│   ├── resources/           # 图标、样式表
│   │   ├── icons/
│   │   ├── styles/
│   │   └── images/
│   └── themes.py            # 深色/浅色主题
├── data/                    # 数据目录
│   ├── database.sqlite      # 主数据库
│   ├── materials/           # 材料数据文件
│   ├── knowledge/           # 工艺知识 YAML
│   └── logs/                # 操作日志
│       ├── app.log          # 应用日志
│       ├── operations.jsonl # 操作日志 (JSON Lines)
│       └── calculations.jsonl # 计算日志 (JSON Lines)
├── reports/                 # 生成的报告输出
├── tests/                   # 测试
│   ├── test_core/
│   ├── test_plugins/
│   └── test_gui/
├── docs/                    # 文档（已有专家分析文档）
│   ├── plans/
│   ├── experts/
│   ├── versions/
│   └── user-guide/
├── main.py                  # 应用入口
├── cli.py                   # 命令行接口
├── requirements.txt         # Python 依赖
├── pyproject.toml           # 项目配置
├── CHANGELOG.md             # 版本变更日志
└── README.md                # 项目说明
```

---

## 3. 核心引擎详细设计

### 3.1 插件标准接口 (plugin_api.py)

```python
from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass

@dataclass
class PluginInfo:
    name: str           # 唯一标识 "li_battery"
    version: str        # 语义化版本 "1.0.0"
    description: str    # 插件描述
    author: str         # 作者
    dependencies: list[str]  # 依赖的其他插件名

class PluginBase(ABC):
    """所有插件必须继承的基类"""

    @abstractmethod
    def get_info(self) -> PluginInfo: ...

    @abstractmethod
    def activate(self, context: 'EngineContext') -> None: ...

    @abstractmethod
    def deactivate(self) -> None: ...

    def get_config_schema(self) -> Optional[dict]:
        """返回插件配置的 JSON Schema"""
        return None

    def get_ui_panels(self) -> list:
        """返回插件提供的 GUI 面板"""
        return []

class ParameterEnginePlugin(PluginBase):
    """参数引擎类插件的扩展接口"""

    @abstractmethod
    def get_input_schema(self) -> dict:
        """返回用户需要输入的字段定义 (JSON Schema)"""
        ...

    @abstractmethod
    def calculate_parameters(self, inputs: dict) -> 'WeldRecipe':
        """核心方法：根据输入计算焊接参数"""
        ...

    @abstractmethod
    def validate_parameters(self, recipe: 'WeldRecipe') -> 'ValidationResult':
        """校验参数合理性"""
        ...

    @abstractmethod
    def get_supported_applications(self) -> list[str]:
        """返回支持的应用场景列表"""
        ...
```

### 3.2 事件总线 (event_bus.py)

```python
# 发布-订阅模式，插件间解耦通信
# 支持同步和异步事件
# 所有事件自动记录到日志
```

关键事件定义：

| 事件名 | 触发时机 | 携带数据 |
|--------|---------|---------|
| `project.created` | 新建项目 | 项目配置 |
| `project.opened` | 打开项目 | 项目 ID |
| `inputs.changed` | 用户修改输入 | 变更的字段和值 |
| `calculation.started` | 开始计算 | 输入数据快照 |
| `calculation.completed` | 计算完成 | WeldRecipe 结果 |
| `validation.started` | 开始校验 | WeldRecipe |
| `validation.completed` | 校验完成 | ValidationResult |
| `report.requested` | 请求生成报告 | 报告配置 |
| `report.generated` | 报告生成完毕 | 文件路径列表 |
| `plugin.activated` | 插件激活 | PluginInfo |
| `plugin.deactivated` | 插件停用 | 插件名 |

### 3.3 结构化日志系统 (logger.py)

**三层日志体系：**

| 层级 | 内容 | 存储 | 用途 |
|------|------|------|------|
| 应用日志 | 程序运行状态、错误、警告 | `data/logs/app.log` (滚动) | 调试排错 |
| 操作日志 | 用户每次操作的完整记录 | SQLite `operations` 表 + `data/logs/operations.jsonl` | 审计追溯 |
| 计算日志 | 参数计算的输入/输出/中间过程 | SQLite `calculations` 表 + `data/logs/calculations.jsonl` | AI 训练数据底座 |

**操作日志记录格式 (JSON Lines):**

```json
{
  "timestamp": "2026-02-25T15:30:45.123Z",
  "session_id": "sess_abc123",
  "event_type": "calculation.completed",
  "user_action": "generate_parameters",
  "inputs": { "...完整输入..." },
  "outputs": { "...完整输出..." },
  "intermediate": { "...中间计算步骤..." },
  "validation": { "...校验结果..." },
  "metadata": {
    "engine_version": "1.0.0",
    "plugin_versions": {"li_battery": "1.0.0"},
    "calculation_time_ms": 45,
    "confidence_score": 0.87
  }
}
```

### 3.4 数据仓库 (database.py)

SQLite 数据库核心表：

```sql
-- 项目表
CREATE TABLE projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    application TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    config JSON
);

-- 操作会话表
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    project_id TEXT REFERENCES projects(id),
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    user_name TEXT
);

-- 操作记录表
CREATE TABLE operations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    event_type TEXT NOT NULL,
    user_action TEXT,
    data JSON,
    metadata JSON
);

-- 焊接参数配方表
CREATE TABLE recipes (
    id TEXT PRIMARY KEY,
    project_id TEXT REFERENCES projects(id),
    session_id TEXT REFERENCES sessions(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    application TEXT NOT NULL,
    inputs JSON NOT NULL,
    parameters JSON NOT NULL,
    safety_window JSON,
    validation_result JSON,
    risk_assessment JSON,
    notes TEXT
);

-- 报告记录表
CREATE TABLE reports (
    id TEXT PRIMARY KEY,
    recipe_id TEXT REFERENCES recipes(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    report_type TEXT,  -- 'pdf', 'excel', 'json'
    file_path TEXT,
    metadata JSON
);

-- 材料数据表
CREATE TABLE materials (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,      -- 'foil', 'tab', 'busbar'
    material_type TEXT,  -- 'Cu', 'Al', 'Ni'
    properties JSON NOT NULL,
    source TEXT,
    updated_at TIMESTAMP
);

-- 焊头/砧座数据表
CREATE TABLE sonotrodes (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    sonotrode_type TEXT, -- 'sonotrode', 'anvil'
    material TEXT,
    knurl_type TEXT,
    knurl_pitch REAL,
    knurl_depth REAL,
    contact_area_w REAL,
    contact_area_l REAL,
    properties JSON,
    updated_at TIMESTAMP
);

-- 试焊结果反馈表（AI数据底座）
CREATE TABLE weld_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recipe_id TEXT REFERENCES recipes(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    actual_parameters JSON,  -- 实际使用的参数
    quality_results JSON,    -- 拉剥力、电阻等实测值
    notes TEXT,
    operator TEXT
);
```

---

## 4. 参数生成引擎详细设计

### 4.1 用户输入流程 (4 步向导)

```
Step 1: 选择应用场景
  ├── 锂电池
  │   ├── 极耳焊接 (Tab Welding)
  │   ├── Busbar 焊接
  │   └── 集流盘焊接
  └── 通用金属焊接
      └── 自定义材料组合

Step 2: 输入材料信息
  ├── 上层材料: 材质、厚度、层数、退火状态
  │   (从材料库选择或手动输入)
  └── 下层材料: 材质、厚度

Step 3: 输入工装信息
  ├── 焊头: 材料、齿形、齿距、齿深、工作面尺寸
  │   (从焊头库选择或手动输入)
  └── 砧座: 类型、齿纹参数

Step 4: 输入焊接约束
  ├── 焊接区域: 宽度 x 长度
  ├── 设备参数: 频率、最大功率、驱动类型
  └── 质量要求: 目标拉剥力、目标电阻、CPK
```

### 4.2 三层计算模型

**Layer 1: 物理模型计算**
- 声阻抗匹配分析 -> 推荐焊头材料、能量传递效率
- 界面功率密度估算 -> P = 4*f*mu*sigma_n * A * A_contact
- 多层能量衰减模型 -> 底层可达能量比
- 热扩散估算 -> 界面温升范围、热影响区

**Layer 2: 经验模型修正**
- 材料组合查表 -> 推荐参数范围
- 层数修正 -> 调整能量/振幅/压力
- 焊头齿纹修正 -> 调整接触效率
- 安全约束检查 -> 过焊/穿孔风险

**Layer 3: 输出参数配方 (WeldRecipe)**
- 核心参数: 振幅、压力、能量/时间、控制模式
- 安全窗口: 各参数的上下限
- 预估质量: 拉剥力范围、电阻范围
- 风险评估: 过焊/虚焊/穿孔风险等级
- 建议: 试焊方案、微调指南

### 4.3 三重校验机制 (多 Agent 审核)

```
Validator 1: 物理合理性校验
  - 功率密度范围 (0.5-5 W/mm2)
  - 界面温度 < 0.6 * T_melt
  - 声阻抗匹配效率 > 70%

Validator 2: 工艺安全性校验
  - 振幅 < 材料承受极限
  - 塌陷量估算 < 穿孔阈值
  - 总热输入 < 隔膜安全阈值 (锂电池)
  - 金属碎屑风险等级

Validator 3: 一致性校验
  - 参数窗口宽度满足 CPK 要求
  - 与历史相似案例对比
  - 参数间耦合关系自洽
```

输出: PASS / WARNING / FAIL + 详细原因

---

## 5. 试验报告设计

### 5.1 报告内容结构

```
试验报告 -- 超声波金属焊接参数推荐书
-----------------------------------------------
1. 项目概况
   - 应用场景、日期、操作者、软件版本

2. 输入条件
   - 材料信息（含材料属性表）
   - 焊头/砧座信息
   - 焊接面积与约束条件

3. 推荐焊接参数
   - 核心参数表
   - 安全窗口表
   - 参数配方编号

4. 参数推导依据
   - 物理模型计算过程
   - 经验修正说明
   - 关键假设

5. 风险评估
   - 过焊/虚焊/穿孔风险矩阵
   - 安全建议

6. 校验报告
   - 3 个校验器结果
   - 警告与建议

7. 试焊建议
   - 推荐试焊方案
   - 参数微调指南
   - 质量检测方法

附录: 材料物性数据、计算公式
```

### 5.2 输出格式

| 格式 | 用途 | 技术实现 |
|------|------|---------|
| PDF | 交付客户、归档 | ReportLab 或 WeasyPrint |
| Excel | 产线录入、数据分析 | openpyxl |
| JSON | AI 训练数据、系统集成 | 标准 json 模块 |

---

## 6. GUI 设计

### 6.1 主界面布局

```
+---------+--------------------------------------------------+
|         |                                                  |
| 导航栏   |              主工作区                              |
|         |                                                  |
| > 新建   |  [Step1] [Step2] [Step3] [Step4]                 |
|   项目列表|  (分步向导 tabs)                                   |
| > 材料库  |                                                  |
| > 焊头库  |  [ 输入表单 / 参数结果 ]                            |
| > 知识库  |                                                  |
| > 历史    |  +----------+-----------+-------------+          |
| > 报告    |  | 参数结果  |  风险评估   |  校验结果    |          |
|         |  +----------+-----------+-------------+          |
| ------- |                                                  |
| > 设置   |  [计算参数]  [生成报告]  [导出]                     |
| > 插件   |                                                  |
| > 日志   |                                                  |
+---------+--------------------------------------------------+
| 状态栏: 引擎状态 | 插件数 | 数据库大小 | 最后操作              |
+----------------------------------------------------------+
```

### 6.2 设计风格
- 专业工程软件风格
- 深色/浅色主题可切换
- 中文界面为主，关键术语保留英文

---

## 7. 版本与文档管理

### 7.1 版本策略
- 语义化版本号 MAJOR.MINOR.PATCH
- CHANGELOG.md 记录所有变更
- 每个版本在 docs/versions/ 下有 release notes

### 7.2 自动生成的文档

```
docs/
├── versions/           # 版本发布说明
├── user-guide/         # 用户手册
│   ├── 01-安装指南.md
│   ├── 02-快速入门.md
│   ├── 03-参数生成使用说明.md
│   ├── 04-报告说明.md
│   └── 05-插件开发指南.md
├── api/                # API 文档
├── plans/              # 设计文档
└── experts/            # 专家分析文档
```

---

## 8. 操作日志与 AI 数据底座

### 8.1 日志体系

| 层级 | 内容 | 格式 | 存储 |
|------|------|------|------|
| 应用日志 | 程序运行状态 | 标准 logging | data/logs/app.log |
| 操作日志 | 用户操作记录 | JSON Lines | SQLite + .jsonl |
| 计算日志 | 参数计算全过程 | JSON Lines | SQLite + .jsonl |

### 8.2 AI 数据底座

每次参数计算自动生成训练数据三元组：

```
(输入条件, 推荐参数, [实际焊接结果])
```

实际焊接结果可后续人工回填，形成闭环学习数据。

---

## 9. 技术依赖

### 9.1 Python 依赖

```
PySide6>=6.6            # GUI 框架
numpy>=1.24             # 数值计算
scipy>=1.11             # 科学计算
pyyaml>=6.0             # 配置文件
reportlab>=4.0          # PDF 生成
openpyxl>=3.1           # Excel 生成
matplotlib>=3.8         # 图表
pydantic>=2.0           # 数据校验
```

### 9.2 系统要求
- macOS 12+ (Monterey 及以上)
- Python 3.11+
- 最低 4GB RAM
- 最低 500MB 磁盘空间

---

## 10. 开发阶段规划 (概要)

| 阶段 | 内容 | 预计 |
|------|------|------|
| Phase 1 | 微内核 + 插件框架 + 基础 GUI | 基础架构 |
| Phase 2 | 锂电池参数引擎 + 材料数据库 | 核心功能 |
| Phase 3 | 通用金属焊接引擎 + 知识库 | 扩展场景 |
| Phase 4 | 报告生成器 (PDF/Excel/JSON) | 输出能力 |
| Phase 5 | 三重校验 + 历史对比 | 质量保障 |
| Phase 6 | 完善 GUI + 主题 + 国际化 | 用户体验 |
| Phase 7 | 测试 + 文档 + 打包发布 | 交付准备 |
