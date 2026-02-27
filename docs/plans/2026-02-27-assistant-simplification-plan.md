# Assistant 精简重构实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 assistant 项目从 40+ 模块的全功能企业平台精简为聚焦于 ERP 数据 + 邮件分析 → 决策报表的轻量系统。

**Architecture:** 所有操作在远程服务器 192.168.110.241 (公网 180.152.71.166) 上执行，代码位于 `/home/squall/project`。SSH 密钥：`/Users/jialechen/.ssh/lab_deploy_180_152_71_166`，用户：`squall`。通过 SSH 执行所有文件操作和命令。

**Tech Stack:** Next.js 16.1.6, Prisma + PostgreSQL (pgvector), Redis + BullMQ, PM2, TypeScript

**设计文档:** `docs/plans/2026-02-27-assistant-simplification-design.md`

---

## SSH 快捷变量

以下所有命令中用 `$SSH` 代替完整 SSH 前缀：

```bash
SSH="ssh -i /Users/jialechen/.ssh/lab_deploy_180_152_71_166 squall@192.168.110.241"
PROJECT="/home/squall/project"
```

---

## Phase 0: 备份与准备

### Task 0.1: 数据库备份

**Step 1: 备份 PostgreSQL 数据库**

```bash
$SSH "pg_dump -U squall private_assistant > /home/squall/backups/private_assistant_$(date +%Y%m%d_%H%M%S).sql"
```

Expected: SQL dump 文件生成成功

**Step 2: 验证备份文件**

```bash
$SSH "ls -lh /home/squall/backups/private_assistant_*.sql | tail -1"
```

Expected: 文件大小 > 0

### Task 0.2: 代码备份

**Step 1: 创建代码 git tag 作为回滚点**

```bash
$SSH "cd $PROJECT && git add -A && git stash && git tag pre-simplification-backup"
```

Expected: Tag 创建成功

**Step 2: 创建工作分支**

```bash
$SSH "cd $PROJECT && git checkout -b feat/simplification"
```

Expected: 新分支创建成功

**Step 3: Commit (标记备份点)**

```bash
$SSH "cd $PROJECT && git commit --allow-empty -m 'chore: start assistant simplification refactor'"
```

---

## Phase 1: 删除前端路由页面 (~25 个目录)

### Task 1.1: 删除不需要的前端路由

**Files to delete:**

```
src/app/welding/
src/app/welding-intelligence/
src/app/experiments/
src/app/ml/
src/app/contracts/
src/app/contract-review/
src/app/customers/
src/app/suppliers/
src/app/sales/
src/app/bom/
src/app/qc/
src/app/tracking/
src/app/workflow/
src/app/executive/
src/app/ace/
src/app/intelligence/
src/app/notes/
src/app/calendar/
src/app/meetings/
src/app/tasks/
src/app/todo-tasks/
src/app/projects/
src/app/import/
src/app/knowledge/
src/app/help/
src/app/system-status/
```

**Step 1: 删除所有不需要的前端路由目录**

```bash
$SSH "cd $PROJECT && rm -rf \
  src/app/welding \
  src/app/welding-intelligence \
  src/app/experiments \
  src/app/ml \
  src/app/contracts \
  src/app/contract-review \
  src/app/customers \
  src/app/suppliers \
  src/app/sales \
  src/app/bom \
  src/app/qc \
  src/app/tracking \
  src/app/workflow \
  src/app/executive \
  src/app/ace \
  src/app/intelligence \
  src/app/notes \
  src/app/calendar \
  src/app/meetings \
  src/app/tasks \
  src/app/todo-tasks \
  src/app/projects \
  src/app/import \
  src/app/knowledge \
  src/app/help \
  src/app/system-status"
```

**Step 2: 验证删除**

```bash
$SSH "ls $PROJECT/src/app/ | sort"
```

Expected: 只剩下 `admin/`, `ai/`, `ai-settings/`, `api/`, `business-intelligence/`, `ceo-dashboard/`, `chat/`, `daily-report/`, `dashboard/`, `decisions/`, `favicon.ico`, `fonts/`, `globals.css`, `layout.tsx`, `login/`, `mail/`, `notifications/`, `page.tsx`, `reports/`, `search/`, `settings/`

**Step 3: Commit**

```bash
$SSH "cd $PROJECT && git add -A && git commit -m 'refactor: remove 25 unused frontend route directories'"
```

---

## Phase 2: 删除 API 路由 (~13 个路由组，~54 route 文件)

### Task 2.1: 删除不需要的 API 路由

**API route groups to delete:**

```
src/app/api/workflow/           (7 routes)
src/app/api/intelligence/       (14 routes)
src/app/api/welding/            (4 routes)
src/app/api/welding-intelligence/ (2 routes)
src/app/api/contract/           (5 routes - 注意是单数)
src/app/api/contracts/          (2 routes - 注意是复数)
src/app/api/ontology/           (6 routes)
src/app/api/executive/          (2 routes)
src/app/api/virtual-employees/  (2 routes)
src/app/api/wecom/              (4 routes)
src/app/api/ace/                (8 routes)
src/app/api/mcp/                (3 routes)
src/app/api/cron/intelligence/  (1 route - 在 cron 目录内)
```

**同时删除不再需要的 API 路由：**

```
src/app/api/bom/                (BOM 管理)
src/app/api/materials/          (物料管理)
src/app/api/qc-reports/         (质量报告)
src/app/api/meetings/           (会议)
src/app/api/notes/              (笔记)
src/app/api/todo-tasks/         (待办)
src/app/api/tasks/              (任务)
src/app/api/projects/           (项目管理)
src/app/api/import/             (数据导入)
src/app/api/knowledge/          (知识库)
src/app/api/graphrag/           (图 RAG)
src/app/api/learning/           (学习系统)
src/app/api/experiments/        (实验)
src/app/api/rd-experiments/     (研发实验)
src/app/api/sales-experiments/  (销售实验)
src/app/api/trial-equipments/   (试用设备)
src/app/api/opportunities/      (商机)
src/app/api/customers/          (客户 - ERP 有)
src/app/api/suppliers/          (供应商 - ERP 有)
src/app/api/supplier-evaluations/ (供应商评估)
src/app/api/supplier-prices/    (供应商报价)
src/app/api/sales/              (销售 - ERP 有)
src/app/api/tracking/           (追踪)
src/app/api/quality/            (质量管理)
src/app/api/documents/          (文档管理)
src/app/api/entity/             (实体管理)
src/app/api/asr/                (语音识别)
src/app/api/ml/                 (机器学习)
src/app/api/rules/              (业务规则)
```

**Step 1: 删除所有不需要的 API 路由**

```bash
$SSH "cd $PROJECT && rm -rf \
  src/app/api/workflow \
  src/app/api/intelligence \
  src/app/api/welding \
  src/app/api/welding-intelligence \
  src/app/api/contract \
  src/app/api/contracts \
  src/app/api/ontology \
  src/app/api/executive \
  src/app/api/virtual-employees \
  src/app/api/wecom \
  src/app/api/ace \
  src/app/api/mcp \
  src/app/api/cron/intelligence \
  src/app/api/bom \
  src/app/api/materials \
  src/app/api/qc-reports \
  src/app/api/meetings \
  src/app/api/notes \
  src/app/api/todo-tasks \
  src/app/api/tasks \
  src/app/api/projects \
  src/app/api/import \
  src/app/api/knowledge \
  src/app/api/graphrag \
  src/app/api/learning \
  src/app/api/experiments \
  src/app/api/rd-experiments \
  src/app/api/sales-experiments \
  src/app/api/trial-equipments \
  src/app/api/opportunities \
  src/app/api/customers \
  src/app/api/suppliers \
  src/app/api/supplier-evaluations \
  src/app/api/supplier-prices \
  src/app/api/sales \
  src/app/api/tracking \
  src/app/api/quality \
  src/app/api/documents \
  src/app/api/entity \
  src/app/api/asr \
  src/app/api/ml \
  src/app/api/rules"
```

**Step 2: 验证保留的 API 路由**

```bash
$SSH "ls $PROJECT/src/app/api/ | sort"
```

Expected: 只剩下核心 API：`admin/`, `ai/`, `audit/`, `auth/`, `boss/`, `business-intelligence/`, `ceo/`, `chat/`, `cron/`, `daily-report/`, `decision/`, `decisions/`, `erp/`, `export/`, `extractions/`, `health/`, `mail/`, `metrics/`, `notifications/`, `proxy/`, `reports/`, `search/`, `seed/`, `settings/`, `sse/`, `system-status/`, `user/`, `users/`

**Step 3: Commit**

```bash
$SSH "cd $PROJECT && git add -A && git commit -m 'refactor: remove ~40 unused API route groups'"
```

---

## Phase 3: 删除 lib 模块 (~85 个文件)

### Task 3.1: 删除不需要的 lib 目录和文件

**Directories to delete:**

```
src/lib/ace/                    (28 files - ACE 引擎)
src/lib/executive/              (3 files - 指挥中心)
src/lib/ontology/               (7 files - 语义本体)
src/lib/learning/               (6 files - 行为学习)
src/lib/virtual-employees/      (5 files - 虚拟员工)
src/lib/wecom/                  (5 files - 企业微信)
src/lib/welding-intelligence/   (8 files - 焊接分析)
src/lib/workflow/               (3 files - 工作流)
src/lib/intelligence/           (2 files - 情报)
src/lib/intelligence-gathering/ (10 files - 情报爬取)
src/lib/business-intelligence/  (3 files - BI 生成器)
src/lib/bom/                    (4 files - BOM)
src/lib/mcp/                    (4 files - MCP)
src/lib/documents/              (4 files - 文档处理)
src/lib/quality/                (6 files - 质量管理)
src/lib/tracking/               (4 files - 追踪)
src/lib/rules/                  (8 files - 业务规则)
src/lib/ml/                     (5 files - ML)
src/lib/graphrag/               (7 files - 图 RAG)
src/lib/server-monitor/         (3 files - 服务器监控)
src/lib/storage/                (2 files - 云存储)
```

**Standalone files to delete:**

```
src/lib/wecom.ts
src/lib/entity-resolution.ts    (不再需要，CRM 在 ERP)
src/lib/assistant/              (assistant-tools.ts 需要重写，先删再重建)
```

**Step 1: 删除所有不需要的 lib 目录**

```bash
$SSH "cd $PROJECT && rm -rf \
  src/lib/ace \
  src/lib/executive \
  src/lib/ontology \
  src/lib/learning \
  src/lib/virtual-employees \
  src/lib/wecom \
  src/lib/welding-intelligence \
  src/lib/workflow \
  src/lib/intelligence \
  src/lib/intelligence-gathering \
  src/lib/business-intelligence \
  src/lib/bom \
  src/lib/mcp \
  src/lib/documents \
  src/lib/quality \
  src/lib/tracking \
  src/lib/rules \
  src/lib/ml \
  src/lib/graphrag \
  src/lib/server-monitor \
  src/lib/storage \
  src/lib/wecom.ts \
  src/lib/entity-resolution.ts"
```

**Step 2: 验证保留的 lib 模块**

```bash
$SSH "ls $PROJECT/src/lib/ | sort"
```

Expected: 保留核心模块：`assistant/`, `auth.ts`, `decision-engine/`, `encryption.ts`, `env.ts`, `erp/`, `erp-ai-interface.ts`, `gpu-resource.ts`, `llm/`, `mail-sync.ts`, `mail.ts`, `prisma.ts`, `queue.ts`, `redis.ts`, `reports/`, `sse.ts`, `system/`, `task-runner.ts`, 以及其它基础工具文件

**Step 3: Commit**

```bash
$SSH "cd $PROJECT && git add -A && git commit -m 'refactor: remove ~85 unused lib module files'"
```

---

## Phase 4: 修复保留模块中的断裂引用

### Task 4.1: 修复 task-runner.ts (P0)

`task-runner.ts` 在 `runDocumentProcessingTask()` 中动态导入了已删除的 `virtual-employees/coordinator`。

**Step 1: 查看当前代码**

```bash
$SSH "grep -n 'virtual-employee\|virtualEmployee' $PROJECT/src/lib/task-runner.ts"
```

**Step 2: 移除 virtual-employees 动态导入**

找到 `runDocumentProcessingTask()` 中的虚拟员工 review 步骤（约在第 669 行附近），删除整个 try/catch 块，或替换为空操作：

```typescript
// 删除类似这样的代码块:
// const { virtualEmployeeCoordinator } = await import("./virtual-employees/coordinator");
// 替换为：
const virtualReviewResult = { total: 0, confirmed: 0, rejected: 0, escalated: 0, errors: 0 };
```

**Step 3: 验证修复**

```bash
$SSH "grep -n 'virtual-employee\|virtualEmployee' $PROJECT/src/lib/task-runner.ts"
```

Expected: 无匹配（或仅剩 fallback 结果）

### Task 4.2: 修复 reports/periodic-reports.ts (P0)

该文件的 `collectReportData()` 查询了已删除的 Prisma 模型 (experiments, welding patterns)。

**Step 1: 查看当前代码中的 Prisma 查询**

```bash
$SSH "grep -n 'prisma\.\|Experiment\|welding\|WeldingParameter' $PROJECT/src/lib/reports/periodic-reports.ts"
```

**Step 2: 将 experiments 和 welding 数据收集替换为空数据**

将引用已删除模型的数据库查询注释掉或替换为空数组/零值。保留 LLM 报表生成逻辑，只修改数据收集部分。

**Step 3: 验证修复**

```bash
$SSH "grep -n 'prisma\.experiment\|prisma\.welding\|prisma\.salesExperiment' $PROJECT/src/lib/reports/periodic-reports.ts"
```

Expected: 无匹配

### Task 4.3: 修复 decision-engine/ (P1)

决策引擎中有已禁用的 customer/project 相关代码。

**Step 1: 检查当前状态**

```bash
$SSH "grep -n 'DISABLED\|Customer\|Project\|analyzeCustomer\|analyzeProject' $PROJECT/src/lib/decision-engine/*.ts"
```

**Step 2: 清理 dead code**

在以下文件中移除已注释/禁用的代码：
- `risk-detector.ts` — 移除 disabled customer/project risk metrics
- `daily-briefing.ts` — 从 `DailyMetrics` 接口移除 `activeProjects`, `riskyProjects`, `activeCustomers`, `newCustomers` 字段
- `trend-analyzer.ts` — 移除 `analyzeCustomerTrends()` 和 `analyzeProjectTrends()` 桩函数
- `suggestion-generator.ts` — 移除注释掉的 customer/project suggestion 块

**Step 3: 验证无残留引用**

```bash
$SSH "grep -rn 'Customer\|Project\b' $PROJECT/src/lib/decision-engine/*.ts | grep -v '//'"
```

Expected: 无活跃引用（注释可忽略）

### Task 4.4: 修复 sidebar 导航 (P2)

**File:** `src/components/app-shell.tsx`

**Step 1: 查看当前导航项**

```bash
$SSH "grep -n 'href.*=\|label.*=\|icon.*=' $PROJECT/src/components/app-shell.tsx | head -60"
```

**Step 2: 移除以下导航项**

删除指向已删除路由的导航项（约 12 个）:
- ACE引擎 (`/ace`)
- Intelligence (`/intelligence`)
- Business Intelligence 如果有单独入口
- 客户管理 (`/customers`)
- 供应商管理 (`/suppliers`)
- 销售管道 (`/sales`)
- 焊接数据 (`/welding`)
- 焊接智能 (`/welding-intelligence`)
- BOM管理 (`/bom`)
- 试验管理 (`/experiments`)
- 质量检查 (`/qc`)
- Workflow (`/workflow`)
- 以及其它已删除路由的导航项

**Step 3: 验证导航项只指向保留的路由**

```bash
$SSH "grep 'href' $PROJECT/src/components/app-shell.tsx | grep -vE 'mail|chat|ceo-dashboard|dashboard|daily-report|business-intelligence|reports|decisions|ai|login|settings|admin|notifications|search'"
```

Expected: 只有 `/` (首页) 和其它基础链接

### Task 4.5: 修复其他有断裂引用的保留文件

以下文件导入了已删除模块，需逐一修复：

**关键文件列表：**

| 文件 | 断裂引用 | 修复方式 |
|------|---------|---------|
| `src/app/api/user/context/route.ts` | workflow, intelligence, contract, executive, ace, knowledge, todo-task, calendar | 移除相关数据收集逻辑 |
| `src/app/api/erp/ai/route.ts` | workflow, contract, ace | 移除对应处理分支 |
| `src/app/api/seed/route.ts` | workflow, contract | 移除对应 seed 逻辑 |
| `src/app/api/mail/*/route.ts` (多个) | executive, knowledge, ace | 移除相关 import 和调用 |
| `src/app/api/reports/*/route.ts` | contract, ace | 移除相关数据源 |
| `src/app/api/search/*/route.ts` | contract, ace | 缩小搜索范围 |
| `src/app/api/boss/pending-reviews/route.ts` | contract | 移除合同审批部分 |
| `src/app/api/export/csv/route.ts` | contract, ace | 移除相关导出类型 |
| `src/app/reports/page.tsx` | `@/lib/wecom` (WeComReport type) | 移除 WeComReport 引用 |

**Step 1: 查找所有断裂引用**

```bash
$SSH "cd $PROJECT && grep -rn 'from.*@/lib/ace\|from.*@/lib/executive\|from.*@/lib/ontology\|from.*@/lib/workflow\|from.*@/lib/intelligence\|from.*@/lib/virtual-employee\|from.*@/lib/wecom\|from.*@/lib/welding\|from.*@/lib/learning\|from.*@/lib/mcp\|from.*@/lib/bom\|from.*@/lib/documents\|from.*@/lib/quality\|from.*@/lib/tracking\|from.*@/lib/rules\|from.*@/lib/ml\|from.*@/lib/graphrag\|from.*@/lib/server-monitor\|from.*@/lib/storage\|from.*@/lib/entity-resolution' src/ --include='*.ts' --include='*.tsx' 2>/dev/null"
```

**Step 2: 逐文件修复断裂引用**

对于每个匹配的文件：
1. 移除 import 语句
2. 移除使用该 import 的代码块
3. 如果是可选功能（如向 knowledge base 写入），直接删除调用
4. 如果是必需功能（如函数参数），提供空实现或移除该功能

**Step 3: 验证无断裂引用残留**

```bash
$SSH "cd $PROJECT && grep -rn 'from.*@/lib/ace\|from.*@/lib/executive\|from.*@/lib/ontology\|from.*@/lib/workflow\|from.*@/lib/intelligence\|from.*@/lib/virtual-employee\|from.*@/lib/wecom\|from.*@/lib/welding\|from.*@/lib/learning\|from.*@/lib/mcp\|from.*@/lib/bom\|from.*@/lib/documents\|from.*@/lib/quality\|from.*@/lib/tracking\|from.*@/lib/rules\|from.*@/lib/ml\|from.*@/lib/graphrag\|from.*@/lib/server-monitor\|from.*@/lib/storage\|from.*@/lib/entity-resolution' src/ --include='*.ts' --include='*.tsx' 2>/dev/null"
```

Expected: 0 matches

**Step 4: Commit**

```bash
$SSH "cd $PROJECT && git add -A && git commit -m 'fix: resolve all broken imports after module removal'"
```

---

## Phase 5: 删除多余脚本和服务定义

### Task 5.1: 清理 scripts/ 和 services/ 目录

**Step 1: 删除不再需要的脚本**

```bash
$SSH "cd $PROJECT && rm -f \
  scripts/daily-intelligence-task.ts \
  scripts/test-digest-generation.ts \
  scripts/test-intelligence-collection.ts"
```

**Step 2: 删除不再需要的微服务定义（如确认不用）**

```bash
$SSH "ls $PROJECT/services/ 2>/dev/null"
```

评估每个 service，如果是已删除功能的专用服务则删除。

**Step 3: Commit**

```bash
$SSH "cd $PROJECT && git add -A && git commit -m 'chore: remove unused scripts and service definitions'"
```

---

## Phase 6: Prisma Schema 精简

### Task 6.1: 安全删除 standalone 模型（Category A）

这些模型没有被任何保留模型引用，可以安全删除。

**Models to delete (standalone, zero risk):**

```
ACEPrediction, TriggerRule, PushRecord, ExecutionRequest
ACEConfig, LearningFeedback, PredictionModelPerformance, PushEffectiveness
OntologyEntityType, OntologyRelationType, OntologyRelationConstraint
OntologyInferenceRule, OntologyInferenceResult, OntologyVersion
CodeGenSession, CodeGenMessage, GeneratedFile, CodeTemplate, HardwareTemplate, CodeFeedback
ClaudeCodeSession, ClaudeCodeMessage
WeComMessage, WeComSyncLog
WeldingApplication, WeldingFeature, WeldingParameterPattern, WeldingSession, WeldingWaveform
AttackAttempt, BannedIP, HoneypotLog, AttackTraceReport, ThreatAlert
AutoBanConfig, HoneypotConfig, AlertConfig
UserContext (ACE 的)
CollectedArticle, TechIntelArticle, TechIntelSource, TechKnowledge
WorkflowDefinition, WorkflowInstance (确认无 cascade 引用)
TodoTask
MCPAccessLog
```

**Step 1: 编辑 schema.prisma，移除以上模型**

使用 SSH + sed 或手动编辑移除每个 model 块。每个 model 块从 `model XXX {` 到对应的 `}` 结束。

**Step 2: 清理 User 模型中指向已删除模型的关系字段**

在 `User` 模型中，移除这些 relation 字段:
- `todoTasks TodoTask[]`
- `Experiment Experiment[]` (如果删除 Experiment)
- `Opportunity Opportunity[]` (如果删除 Opportunity)
- `QCReport QCReport[]` (如果删除 QCReport)
- `contracts Contract[]` (如果删除 Contract)
- `learningRules LearningRule[]` (如果删除 LearningRule)
- `knowledgeItems KnowledgeItem[]` (如果删除 KnowledgeItem)
- `knowledgeNodes KnowledgeNode[]` (如果删除 KnowledgeNode)
- `behaviorLogs UserBehaviorLog[]` (如果删除 UserBehaviorLog)
- `codeGenSessions CodeGenSession[]`

**Step 3: 处理 Mail 模型中的可选关系**

Mail 有 `customerId` 和 `projectId` 可选 FK。由于 Customer 和 Project 是 hub 模型（被很多其他模型引用），**保留 Customer 和 Project 模型**，不删除它们。这避免了大量级联修改。

> **注意**: Customer 和 Project 模型保留在 schema 中作为邮件-实体关联使用。它们的数据来源是邮件 AI 提取，不再从 CRM 模块管理。前端页面已删除，只保留 DB 模型。

**Step 4: 验证 schema 有效**

```bash
$SSH "cd $PROJECT && npx prisma validate"
```

Expected: "The schema is valid."

**Step 5: 生成迁移**

```bash
$SSH "cd $PROJECT && npx prisma migrate dev --name simplification-remove-unused-models"
```

Expected: Migration created and applied

**Step 6: 重新生成 Prisma Client**

```bash
$SSH "cd $PROJECT && npx prisma generate"
```

**Step 7: Commit**

```bash
$SSH "cd $PROJECT && git add -A && git commit -m 'refactor: remove ~40 unused Prisma models from schema'"
```

---

## Phase 7: 重写 ERP Adapter

### Task 7.1: 删除旧适配器，创建 SinnexAdapter

**Step 1: 删除旧适配器**

```bash
$SSH "rm -f $PROJECT/src/lib/erp/adapters/kingdee.ts $PROJECT/src/lib/erp/adapters/mock.ts"
```

**Step 2: 创建 SinnexAdapter**

创建 `src/lib/erp/adapters/sinnex.ts`:

```typescript
import { ERPAdapter, ERPCustomer, ERPSupplier, ERPSalesOrder, ERPPurchaseOrder, ERPInventory, ERPFinanceSummary } from "../types";

export class SinnexAdapter implements ERPAdapter {
  private baseUrl: string;
  private apiKey: string;

  constructor() {
    this.baseUrl = process.env.ERP_BASE_URL || "http://127.0.0.1:3002";
    this.apiKey = process.env.ERP_API_KEY || "";
  }

  private async fetch<T>(path: string, params?: Record<string, string>): Promise<T> {
    const url = new URL(path, this.baseUrl);
    if (params) {
      Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));
    }
    const res = await fetch(url.toString(), {
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
    });
    if (!res.ok) {
      throw new Error(`ERP API error: ${res.status} ${res.statusText}`);
    }
    return res.json();
  }

  async getCustomers(): Promise<ERPCustomer[]> {
    return this.fetch("/api/customers");
  }

  async getSuppliers(): Promise<ERPSupplier[]> {
    return this.fetch("/api/suppliers");
  }

  async getSalesOrders(): Promise<ERPSalesOrder[]> {
    return this.fetch("/api/sales-orders");
  }

  async getPurchaseOrders(): Promise<ERPPurchaseOrder[]> {
    return this.fetch("/api/purchase-orders");
  }

  async getInventory(): Promise<ERPInventory[]> {
    return this.fetch("/api/inventory");
  }

  async getFinanceSummary(): Promise<ERPFinanceSummary> {
    return this.fetch("/api/finance/reports");
  }

  // 新增方法 - 自研 ERP 独有
  async getWorkOrders() {
    return this.fetch("/api/work-orders");
  }

  async getQualityStats() {
    return this.fetch("/api/reports/ultrasonic-kpis");
  }

  async getProjectStatus() {
    return this.fetch("/api/projects");
  }

  async getReceivables() {
    return this.fetch("/api/finance/receivables");
  }

  async getPayables() {
    return this.fetch("/api/finance/payables");
  }
}
```

**Step 3: 更新 index.ts factory**

修改 `src/lib/erp/index.ts`，将默认适配器改为 `SinnexAdapter`:

```typescript
import { SinnexAdapter } from "./adapters/sinnex";
import type { ERPAdapter } from "./types";

export function getERPAdapter(): ERPAdapter {
  return new SinnexAdapter();
}

export { getERPDashboardData } from "./dashboard"; // 如果存在
```

**Step 4: 更新 .env 配置**

```bash
$SSH "cd $PROJECT && sed -i 's/^ERP_TYPE=.*/ERP_TYPE=sinnex/' .env && grep -q 'ERP_BASE_URL' .env || echo 'ERP_BASE_URL=http://127.0.0.1:3002' >> .env"
```

**Step 5: Commit**

```bash
$SSH "cd $PROJECT && git add -A && git commit -m 'feat: replace Kingdee adapter with SinnexAdapter for self-built ERP'"
```

---

## Phase 8: 增强 AI Chat — ERP 查询工具

### Task 8.1: 添加 ERP 查询工具函数

**File:** `src/lib/assistant/assistant-tools.ts` (或重建)

**Step 1: 查看现有 Chat API 和工具注册方式**

```bash
$SSH "cat $PROJECT/src/app/api/chat/stream/route.ts | head -50"
$SSH "cat $PROJECT/src/lib/assistant/assistant-tools.ts 2>/dev/null | head -50"
```

**Step 2: 添加 ERP 查询工具函数定义**

在 assistant-tools.ts 中注册以下工具:

```typescript
import { getERPAdapter } from "@/lib/erp";

export const erpTools = [
  {
    name: "query_erp_customers",
    description: "查询 ERP 系统中的客户列表",
    parameters: { type: "object", properties: { search: { type: "string", description: "搜索关键词" } } },
    handler: async (params: { search?: string }) => {
      const adapter = getERPAdapter();
      const customers = await adapter.getCustomers();
      if (params.search) {
        return customers.filter(c => JSON.stringify(c).includes(params.search!));
      }
      return customers;
    },
  },
  {
    name: "query_erp_sales_orders",
    description: "查询 ERP 系统中的销售订单",
    parameters: { type: "object", properties: { status: { type: "string" }, period: { type: "string" } } },
    handler: async (params: { status?: string; period?: string }) => {
      const adapter = getERPAdapter();
      return adapter.getSalesOrders();
    },
  },
  {
    name: "query_erp_inventory",
    description: "查询 ERP 系统中的库存数据",
    parameters: { type: "object", properties: { material: { type: "string" } } },
    handler: async () => {
      const adapter = getERPAdapter();
      return adapter.getInventory();
    },
  },
  {
    name: "query_erp_finance",
    description: "查询 ERP 系统中的财务汇总数据（应收/应付/利润）",
    parameters: { type: "object", properties: { type: { type: "string", enum: ["summary", "receivables", "payables"] } } },
    handler: async (params: { type?: string }) => {
      const adapter = getERPAdapter() as any;
      switch (params.type) {
        case "receivables": return adapter.getReceivables();
        case "payables": return adapter.getPayables();
        default: return adapter.getFinanceSummary();
      }
    },
  },
  {
    name: "query_erp_production",
    description: "查询 ERP 系统中的生产工单状态",
    parameters: { type: "object", properties: { status: { type: "string" } } },
    handler: async () => {
      const adapter = getERPAdapter() as any;
      return adapter.getWorkOrders();
    },
  },
  {
    name: "query_mail_analysis",
    description: "查询邮件分析结果，包括风险邮件、重要邮件",
    parameters: { type: "object", properties: { importance: { type: "string" }, recent: { type: "boolean" } } },
    handler: async (params: { importance?: string; recent?: boolean }) => {
      const { prisma } = await import("@/lib/prisma");
      return prisma.mail.findMany({
        where: {
          ...(params.importance && { analysis: { importance: params.importance } }),
        },
        include: { analysis: true },
        orderBy: { receivedAt: "desc" },
        take: params.recent ? 20 : 50,
      });
    },
  },
  {
    name: "query_risk_alerts",
    description: "查询当前风险预警",
    parameters: { type: "object", properties: { level: { type: "string", enum: ["high", "medium", "low"] } } },
    handler: async (params: { level?: string }) => {
      const { prisma } = await import("@/lib/prisma");
      return prisma.riskAlert.findMany({
        where: {
          ...(params.level && { level: params.level }),
          resolved: false,
        },
        orderBy: { createdAt: "desc" },
        take: 20,
      });
    },
  },
];
```

**Step 3: 在 Chat stream 路由中注册这些工具**

修改 `src/app/api/chat/stream/route.ts`，将 `erpTools` 加入 LLM function calling 的工具列表中。

**Step 4: Commit**

```bash
$SSH "cd $PROJECT && git add -A && git commit -m 'feat: add ERP query tools to AI Chat for natural language data access'"
```

---

## Phase 9: 更新 PM2 生态配置

### Task 9.1: 修改 ecosystem.config.js

**Step 1: 查看当前配置**

```bash
$SSH "cat $PROJECT/ecosystem.config.js"
```

**Step 2: 修改配置**

- `assistant` 实例数从 4 改为 2
- 移除 `cron-auto-crawl` 进程定义
- 移除 `cron-mail-analyzer` 进程定义（如存在）
- 保留 `queue-worker`, `cron-mail-sync`, `cron-daily-digest`

**Step 3: Commit**

```bash
$SSH "cd $PROJECT && git add -A && git commit -m 'chore: simplify PM2 ecosystem - reduce instances, remove crawl cron'"
```

---

## Phase 10: 构建与验证

### Task 10.1: TypeScript 类型检查

**Step 1: 运行类型检查**

```bash
$SSH "cd $PROJECT && npx tsc --noEmit 2>&1 | head -100"
```

Expected: 无类型错误。如有错误，逐一修复断裂引用。

**Step 2: 如有 TS 错误，修复并重复检查**

常见错误类型：
- `Cannot find module '@/lib/xxx'` → 移除 import 和使用
- `Property 'xxx' does not exist on type 'PrismaClient'` → 移除已删除模型的 prisma 调用
- `Type 'XXX' is not assignable` → 更新类型定义

**Step 3: 循环修复直到 tsc 通过**

### Task 10.2: Next.js 生产构建

**Step 1: 运行 build**

```bash
$SSH "cd $PROJECT && npm run build 2>&1 | tail -50"
```

Expected: Build 成功，无错误

**Step 2: 如有 build 错误，修复并重复**

**Step 3: Commit all fixes**

```bash
$SSH "cd $PROJECT && git add -A && git commit -m 'fix: resolve all build errors after simplification'"
```

---

## Phase 11: 部署

### Task 11.1: 重启 PM2 服务

**Step 1: 停止旧进程**

```bash
$SSH "pm2 stop assistant cron-auto-crawl cron-mail-analyzer 2>/dev/null; pm2 delete cron-auto-crawl cron-mail-analyzer 2>/dev/null"
```

**Step 2: 重载所有进程**

```bash
$SSH "cd $PROJECT && pm2 reload ecosystem.config.js"
```

**Step 3: 验证进程状态**

```bash
$SSH "pm2 list"
```

Expected: assistant (x2), queue-worker (x2), cron-mail-sync (x1), cron-daily-digest (x1) 全部 online

### Task 11.2: 验证功能

**Step 1: 检查应用健康**

```bash
$SSH "curl -s http://localhost:3000/api/health | head -20"
```

Expected: 200 OK

**Step 2: 检查邮件功能**

```bash
$SSH "curl -s http://localhost:3000/api/mail | head -5"
```

**Step 3: 检查 ERP 集成**

```bash
$SSH "curl -s http://localhost:3000/api/erp/dashboard | head -20"
```

**Step 4: 检查内存使用**

```bash
$SSH "pm2 monit" # 或
$SSH "pm2 describe 10 | grep -E 'heap|mem'"
```

Expected: Heap 使用率显著低于 90%

**Step 5: 最终 Commit**

```bash
$SSH "cd $PROJECT && git add -A && git commit -m 'chore: assistant simplification complete - ERP + Mail + Decision focus'"
```

---

## 完成总结

精简后的 assistant 系统包含：

| 功能 | 状态 |
|------|------|
| 完整邮件客户端 (IMAP/SMTP + AI 分析) | 保留 |
| ERP 数据获取 (SinnexAdapter → 自研 ERP) | 重写 |
| CEO 仪表盘 + 每日简报 | 保留 |
| BI 报表 + 风险检测 + 趋势分析 | 保留 |
| AI Chat (ERP + 邮件自然语言查询) | 增强 |
| 用户管理 + LLM 配置 | 保留 |
| 焊接/合同/CRM/工作流/情报/ACE 等 | 已移除 |

预期效果：前端路由 40→15，API 路由 80→30，内存 900MB→400MB，Heap 90%→~55%
