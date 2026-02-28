# Assistant 项目精简重构设计

> 日期: 2026-02-27
> 方案: B (精简重构)

## 目标

将 assistant 项目从 40+ 模块的全功能企业平台，精简为聚焦于 **ERP 数据 + 邮件分析 → 决策报表** 的轻量系统。

## 保留的核心能力

1. **完整邮件客户端** — IMAP 收取、SMTP 发送、AI 分析摘要、实体提取、自动回复生成与审批
2. **ERP 数据获取** — 通过新 SinnexAdapter 对接自研 ERP (localhost:3002) REST API
3. **CEO 日报 + BI 报表** — 每日简报、风险检测、趋势分析、周/月/季度 BI 报表
4. **AI Chat** — 自然语言查询 ERP 数据和邮件分析结果

---

## 一、模块保留/删除清单

### 保留的前端路由 (src/app/)

| 路由 | 用途 |
|------|------|
| `mail/` | 完整邮件客户端 |
| `chat/` | AI 对话 |
| `ceo-dashboard/` | CEO 仪表盘 |
| `dashboard/` | 主面板 |
| `daily-report/` | 每日报告 |
| `business-intelligence/` | BI 分析报表 |
| `reports/` | 定期报表查看 |
| `decisions/` | 决策建议 |
| `ai/` | AI 提取/统计 (为邮件分析服务) |
| `ai-settings/` | AI 配置 |
| `login/` | 登录 |
| `settings/` | 系统设置 (精简) |
| `admin/` | 管理后台 (仅保留用户管理、LLM 配置、邮件账户) |
| `notifications/` | 通知中心 |
| `search/` | 统一搜索 (精简到邮件+报表范围) |

### 删除的前端路由 (~25 个)

| 路由 | 删除原因 |
|------|----------|
| `welding/`, `welding-intelligence/` | 焊接专用，不需要 |
| `experiments/`, `ml/` | 实验/ML，不需要 |
| `contracts/`, `contract-review/` | 合同管理，ERP 有 |
| `customers/`, `suppliers/`, `sales/` | CRM/销售，ERP 有 |
| `bom/` | 物料清单，ERP 有 |
| `qc/`, `tracking/` | 质检/追踪，ERP 有 |
| `workflow/` | 工作流引擎 |
| `executive/` | 指挥中心 |
| `ace/` | 主动推送系统 |
| `intelligence/` | 技术情报爬取 |
| `notes/`, `calendar/`, `meetings/` | 办公协作 |
| `tasks/`, `todo-tasks/` | 任务管理 |
| `projects/` | 项目管理 |
| `import/`, `knowledge/` | 数据导入/知识库 |
| `help/`, `system-status/` | 帮助/系统状态 |

### 保留的 src/lib/ 核心模块

| 模块 | 用途 |
|------|------|
| `mail.ts`, `mail-sync.ts` | 邮件收发同步 |
| `llm/` | LLM 多模型路由 (整个目录) |
| `erp/` | ERP 适配器 (重写) |
| `erp-ai-interface.ts` | ERP 向 assistant 请求 AI 能力 |
| `decision-engine/` | 决策引擎 (风险/趋势/建议/简报) |
| `reports/` | 报表生成 |
| `auth.ts` | 认证 |
| `prisma.ts` | 数据库 |
| `redis.ts` | 缓存/消息 |
| `queue.ts` | BullMQ 队列 |
| `encryption.ts` | 加密 |
| `sse.ts` | 实时推送 |
| `gpu-resource.ts` | GPU 资源管理 |
| `entity-resolution.ts` | 实体消歧 |

### 删除的 src/lib/ 模块

| 模块 | 删除原因 |
|------|----------|
| `ace/` | 主动推送系统 |
| `intelligence/` | 技术情报爬取 |
| `executive/` | 指挥中心 |
| `workflow/` | 工作流引擎 |
| `virtual-employees/` | 虚拟员工 |
| `learning/` | 行为学习系统 |
| `welding-intelligence/` | 焊接分析 |
| `ontology/` | 语义本体 |
| `wecom/` | 企业微信 |
| `system/` | 服务器监控 |
| `mcp/` | MCP 协议 |

---

## 二、ERP 适配器重写

### 文件变更

```
src/lib/erp/
  types.ts          ← 保留，扩展数据类型
  index.ts          ← 重写 factory，默认返回 SinnexAdapter
  adapters/
    sinnex.ts       ← 新建，对接自研 ERP REST API
    kingdee.ts      ← 删除
    mock.ts         ← 删除
```

### SinnexAdapter 对接的 ERP API

| 方法 | 调用 ERP 端点 | 用途 |
|------|-------------|------|
| `getCustomers()` | `GET /api/customers` | 客户列表 |
| `getSuppliers()` | `GET /api/suppliers` | 供应商列表 |
| `getSalesOrders()` | `GET /api/sales-orders` | 销售订单 |
| `getPurchaseOrders()` | `GET /api/purchase-orders` | 采购订单 |
| `getInventory()` | `GET /api/inventory` | 库存数据 |
| `getFinanceSummary()` | `GET /api/finance/reports` | 财务汇总 |
| `getWorkOrders()` | `GET /api/work-orders` | 生产工单 (新增) |
| `getQualityStats()` | `GET /api/reports/ultrasonic-kpis` | 质量统计 (新增) |
| `getProjectStatus()` | `GET /api/projects` | 项目状态 (新增) |
| `getReceivables()` | `GET /api/finance/receivables` | 应收账款 (新增) |
| `getPayables()` | `GET /api/finance/payables` | 应付账款 (新增) |

### 认证方式

自研 ERP 通过 `ExternalApiKey` Bearer Token 认证：

```env
ERP_TYPE=sinnex
ERP_BASE_URL=http://127.0.0.1:3002
ERP_API_KEY=<在 ERP 中创建的 ExternalApiKey>
```

### 数据流向

```
自研 ERP (3002)  ──REST API──►  SinnexAdapter  ──►  Decision Engine / Dashboard / Chat
自研 ERP (3002)  ──/api/erp/ai──►  ERPAIService  (反向：ERP 调 assistant AI，保持不变)
```

---

## 三、数据库 Schema 精简

### 删除的 Prisma 模型 (~30 个)

| 模型组 | 具体模型 |
|--------|---------|
| 合同 | `Contract`, `ContractReview` |
| 客户/项目 | `Customer`, `Project` (assistant 自己的，ERP 有) |
| 焊接 | `Experiment` 及相关模型 |
| 工作流 | `WorkflowDefinition`, `WorkflowInstance`, `WorkflowStep` |
| 情报 | `TechIntelSource`, `TechIntelArticle`, `CollectedArticle` |
| 知识库 | `KnowledgeItem`, `DocumentChunk`, `Document` |
| ACE | `ACEConfig`, `LearningFeedback`, `PredictionModelPerformance`, `PushEffectiveness` |
| 学习 | `LearningRule`, `LearningRuleApplication` |
| 办公 | `TodoTask` |
| 本体 | `OntologyEntityType`, `RelationType`, `OntologyRelation` |
| 代码生成 | `CodeGenSession`, `CodeGenMessage`, `GeneratedFile` |
| 销售 | `Opportunity`, `SalesRecord` |
| 质检 | `QCReport` |

### 保留的 Prisma 模型 (~20 个)

| 模型 | 用途 |
|------|------|
| `User` | 认证用户 |
| `Mail`, `MailAnalysis`, `MailAttachment`, `MailExtraction` | 邮件全套 |
| `EmailAccount` | 多邮箱配置 |
| `AutoReply` | 自动回复跟踪 |
| `DailyReport` | 每日报告 |
| `DailyBrief` | CEO 简报 |
| `RiskAlert` | 风险预警 |
| `BusinessInsight` | 商业洞察 |
| `AIPeriodicReport` | 定期 BI 报表 |
| `BusinessRule` | 业务规则 |
| `ChatSession` | AI 对话历史 |
| `Embedding` | 向量嵌入 |
| `SystemNotification` | 系统通知 |
| `AuditLog` | 审计日志 |

### 迁移策略

1. 备份数据库
2. 从 schema.prisma 中移除模型
3. `prisma migrate dev` 生成迁移 (DROP 对应表)
4. 清理孤立数据

---

## 四、PM2 生态精简

### 保留的进程

| 进程 | 实例数 | 用途 |
|------|--------|------|
| `assistant` | 4 → **2** | 主应用 (减少实例节省内存) |
| `queue-worker` | 2 | BullMQ 任务处理 |
| `cron-mail-sync` | 1 | IMAP 邮件同步 |
| `cron-daily-digest` | 1 | 每日报告生成 |

### 删除的进程

| 进程 | 删除原因 |
|------|----------|
| `cron-auto-crawl` | 技术情报爬取 |
| `cron-mail-analyzer` | 功能已合并在 mail-sync 中 |

---

## 五、AI Chat 增强 — ERP 数据查询

### 工具函数注册 (Function Calling)

```
- query_erp_customers(filter?) → 查客户
- query_erp_sales_orders(filter?) → 查销售订单
- query_erp_inventory(filter?) → 查库存
- query_erp_finance(type) → 查财务数据
- query_erp_production(filter?) → 查生产工单
- query_erp_quality(filter?) → 查质量数据
- query_mail_analysis(filter?) → 查邮件分析结果
- query_reports(type, period?) → 查历史报表
- query_risk_alerts(filter?) → 查风险预警
```

---

## 六、精简后架构

```
┌─────────────────────────────────────────────────────────┐
│                  精简后 Assistant (3000)                  │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐             │
│  │ 邮件客户端│  │ AI Chat  │  │CEO 仪表盘 │             │
│  │ 收/发/分析│  │ERP+邮件  │  │日报/BI报表│             │
│  └────┬─────┘  └────┬─────┘  └─────┬─────┘             │
│       │              │              │                    │
│  ┌────┴──────────────┴──────────────┴─────┐             │
│  │          决策引擎 (Decision Engine)      │             │
│  │  风险检测 | 趋势分析 | 建议生成 | 日报   │             │
│  └────┬──────────────┬────────────────────┘             │
│       │              │                                   │
│  ┌────┴────┐    ┌────┴────────┐                         │
│  │邮件数据  │    │ERP Adapter  │                         │
│  │本地 DB   │    │(SinnexAPI)  │                         │
│  └─────────┘    └──────┬──────┘                         │
│                        │                                 │
│  LLM 路由 (Ollama/DeepSeek/Zhipu/OpenAI/Claude)        │
└─────────────────────────────────────────────────────────┘
          │ REST API
          ▼
┌──────────────────┐     ┌──────────────┐
│ 自研 ERP (3002)  │     │ IMAP Servers │
│ 180+ API         │     │ 邮件收取     │
└──────────────────┘     └──────────────┘
```

### 精简效果预估

| 指标 | 精简前 | 精简后 |
|------|--------|--------|
| 前端路由 | ~40 | ~15 |
| API 路由 | ~80 | ~30 |
| lib 模块 | ~80 | ~30 |
| DB 模型 | ~50 | ~20 |
| PM2 实例 | 8 进程 | 5 进程 |
| 内存占用 | ~900MB | ~400MB |
| Heap 压力 | 90%+ | ~50-60% |
