# i18n + Cross-Platform Packaging Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Chinese/English bilingual support via Qt QTranslator and package the app for macOS (.app) and Windows (.exe).

**Architecture:** Wrap all GUI strings in `self.tr()`, extract with `pyside6-lupdate`, translate to Chinese in `.ts` XML, compile to `.qm` with `pyside6-lrelease`. Load translations at startup based on system locale. Each panel gets a `retranslateUi()` method for live language switching. Build script automates translation compilation + PyInstaller packaging.

**Tech Stack:** PySide6 QTranslator / pyside6-lupdate / pyside6-lrelease / PyInstaller

---

## Task 1: Create translations directory and infrastructure

**Files:**
- Create: `translations/` directory
- Modify: `.gitignore`

**Step 1: Create translations directory**

Run: `mkdir -p translations`

**Step 2: Add .qm files to .gitignore**

Add to `.gitignore`:
```
translations/*.qm
```

**Step 3: Commit**

```bash
git add translations/ .gitignore
git commit -m "chore: add translations directory, gitignore .qm files"
```

---

## Task 2: Add self.tr() to MainWindow + retranslateUi

**Files:**
- Modify: `ultrasonic_weld_master/gui/main_window.py`

**Step 1: Rewrite main_window.py with self.tr() on every user-visible string**

Key changes:
- All menu/nav/status strings use `self.tr()`
- Store widget references for every translatable element
- Add `retranslateUi()` method
- Add `on_language_change` callback + `_on_language_changed(lang_code)` handler
- Add `_translator` and `_qt_translator` QTranslator attributes
- Add `_translations_path()` helper that works in dev and PyInstaller frozen mode
- `_on_language_changed`: remove old translator, load new .qm, install, call retranslateUi on self + all panels

Translatable strings (~23):
- Window title, Menu labels (File, View, Help), Menu actions (New Calculation, Quit, Toggle Theme, About)
- Nav items (New Calculation, Results, Reports, History, Settings)
- Status bar (Ready, Calculation complete: %s)
- About dialog title + body

Nav items: create with empty text in `_setup_ui`, set text in `retranslateUi()`.

**Step 2: Verify**

Run: `pytest tests/ -v --tb=short` → 93 passed
Run: `python -c "from ultrasonic_weld_master.gui.main_window import MainWindow; print('OK')"` → OK

**Step 3: Commit**

```bash
git add ultrasonic_weld_master/gui/main_window.py
git commit -m "feat(i18n): add self.tr() and retranslateUi to MainWindow"
```

---

## Task 3: Add self.tr() to InputWizardPanel + retranslateUi

**Files:**
- Modify: `ultrasonic_weld_master/gui/panels/input_wizard.py`

**Step 1: Rewrite input_wizard.py with self.tr()**

Key changes:
- `StepIndicator`: add `set_labels(labels)` method so parent can pass translated labels
- All QGroupBox titles, form labels, button text, application names, error messages use `self.tr()`
- Store QLabel refs for form rows (use `form.addRow(self._label_xxx, widget)` pattern)
- Add `retranslateUi()` that re-sets all text including StepIndicator labels

Translatable strings (~60):
- Step labels: Application, Materials, Tooling, Constraints
- Title: NEW CALCULATION
- Group titles: APPLICATION TYPE, UPPER MATERIAL (FOIL STACK), LOWER MATERIAL (TAB / BUSBAR), WELD GEOMETRY, EQUIPMENT CONSTRAINTS, INPUT SUMMARY
- Form labels: Application:, Material:, Foil Thickness:, Number of Layers:, Thickness:, Weld Width:, Weld Length:, Contact Area:, Frequency:, Max Power:
- Buttons: Back, Next Step, Calculate Parameters
- App names: Li-Battery Tab Welding, Li-Battery Busbar Welding, Li-Battery Collector, General Metal Welding
- Errors: No Engine, Engine not connected. Cannot calculate., Calculation Error
- Description text

**Step 2: Verify**

Run: `python -c "from ultrasonic_weld_master.gui.panels.input_wizard import InputWizardPanel; print('OK')"` → OK
Run: `pytest tests/ -v --tb=short` → all pass

**Step 3: Commit**

```bash
git add ultrasonic_weld_master/gui/panels/input_wizard.py
git commit -m "feat(i18n): add self.tr() and retranslateUi to InputWizardPanel"
```

---

## Task 4: Add self.tr() to ResultPanel + retranslateUi

**Files:**
- Modify: `ultrasonic_weld_master/gui/panels/result_panel.py`

**Step 1: Rewrite result_panel.py with self.tr()**

Translatable strings (~20):
- Title: CALCULATION RESULTS
- Empty label: No results yet. Run a calculation from the wizard.
- Group titles: PARAMETERS, RISK ASSESSMENT, ALL PARAMETERS, VALIDATION, RECOMMENDATIONS
- Table headers: PARAMETER, VALUE, SAFE MIN, SAFE MAX
- Status text: No validation data, No recommendations

Store QGroupBox and QLabel references. Table headers reset via `setHorizontalHeaderLabels()` in `retranslateUi()`.

**Step 2: Verify and commit**

Run: `python -c "from ultrasonic_weld_master.gui.panels.result_panel import ResultPanel; print('OK')"` → OK
Run: `pytest tests/ -v --tb=short` → all pass

```bash
git add ultrasonic_weld_master/gui/panels/result_panel.py
git commit -m "feat(i18n): add self.tr() and retranslateUi to ResultPanel"
```

---

## Task 5: Add self.tr() to ReportPanel, HistoryPanel, SettingsPanel

**Files:**
- Modify: `ultrasonic_weld_master/gui/panels/report_panel.py`
- Modify: `ultrasonic_weld_master/gui/panels/history_panel.py`
- Modify: `ultrasonic_weld_master/gui/panels/settings_panel.py`

**Step 1: Rewrite report_panel.py with self.tr()**

Translatable strings (~18):
- Title: REPORT GENERATION, Empty: No recipe loaded. Run a calculation first.
- Groups: RECIPE INFO, PARAMETER PREVIEW, RISK ASSESSMENT, VALIDATION, EXPORT
- Table headers: PARAMETER, VALUE, SAFE RANGE
- Buttons: Export JSON, Export Excel, Export PDF, Export All
- Dialog titles: Export, Export All

**Step 2: Rewrite history_panel.py with self.tr()**

Translatable strings (~12):
- Title: CALCULATION HISTORY, Button: Refresh
- Headers: ID, APPLICATION, MATERIALS, DATE, STATUS
- Status messages: No history loaded..., Engine not connected., No calculations found., Loaded %d records., Error: %s

**Step 3: Rewrite settings_panel.py with self.tr() + add Language dropdown**

Translatable strings (~15):
- Title: SETTINGS
- Groups: APPEARANCE, LANGUAGE (new), DEFAULT VALUES, FILE PATHS
- Labels: Theme:, Language: (new), Default Frequency:, Default Max Power:, Report Output:
- Button: Browse..., Select Report Directory

New: Add `on_language_change` callback parameter. Add LANGUAGE group with QComboBox `["中文", "English"]`. Handler maps index 0 → `"zh_CN"`, index 1 → `"en"` and calls callback.

**Step 4: Verify**

Run: `python -c "from ultrasonic_weld_master.gui.panels.report_panel import ReportPanel; from ultrasonic_weld_master.gui.panels.history_panel import HistoryPanel; from ultrasonic_weld_master.gui.panels.settings_panel import SettingsPanel; print('OK')"` → OK
Run: `pytest tests/ -v --tb=short` → all pass

**Step 5: Commit**

```bash
git add ultrasonic_weld_master/gui/panels/report_panel.py ultrasonic_weld_master/gui/panels/history_panel.py ultrasonic_weld_master/gui/panels/settings_panel.py
git commit -m "feat(i18n): add self.tr() and retranslateUi to Report, History, Settings panels"
```

---

## Task 6: Extract strings and create Chinese translation

**Files:**
- Create: `translations/app_zh_CN.ts`
- Create: `translations/app_en.ts`

**Step 1: Run pyside6-lupdate to extract all strings**

```bash
pyside6-lupdate -extensions py \
  ultrasonic_weld_master/gui/main_window.py \
  ultrasonic_weld_master/gui/panels/input_wizard.py \
  ultrasonic_weld_master/gui/panels/result_panel.py \
  ultrasonic_weld_master/gui/panels/report_panel.py \
  ultrasonic_weld_master/gui/panels/history_panel.py \
  ultrasonic_weld_master/gui/panels/settings_panel.py \
  -ts translations/app_zh_CN.ts

pyside6-lupdate -extensions py \
  ultrasonic_weld_master/gui/main_window.py \
  ultrasonic_weld_master/gui/panels/input_wizard.py \
  ultrasonic_weld_master/gui/panels/result_panel.py \
  ultrasonic_weld_master/gui/panels/report_panel.py \
  ultrasonic_weld_master/gui/panels/history_panel.py \
  ultrasonic_weld_master/gui/panels/settings_panel.py \
  -ts translations/app_en.ts
```

**Step 2: Fill in Chinese translations in app_zh_CN.ts**

Edit the XML: for each `<message>`, fill `<translation>` and remove `type="unfinished"`.

Key translations table:

| English | Chinese |
|---------|---------|
| File | 文件 |
| New Calculation | 新建计算 |
| Quit | 退出 |
| View | 视图 |
| Toggle Dark/Light Theme | 切换深色/浅色主题 |
| Help | 帮助 |
| About | 关于 |
| Results | 计算结果 |
| Reports | 报告 |
| History | 历史记录 |
| Settings | 设置 |
| Ready | 就绪 |
| Calculation complete: %s | 计算完成: %s |
| NEW CALCULATION | 新建计算 |
| APPLICATION TYPE | 应用类型 |
| Application: | 应用类型: |
| Li-Battery Tab Welding | 锂电池极耳焊接 |
| Li-Battery Busbar Welding | 锂电池汇流排焊接 |
| Li-Battery Collector | 锂电池集流体 |
| General Metal Welding | 通用金属焊接 |
| UPPER MATERIAL (FOIL STACK) | 上层材料 (箔片叠层) |
| LOWER MATERIAL (TAB / BUSBAR) | 下层材料 (极耳 / 汇流排) |
| Material: | 材料: |
| Foil Thickness: | 箔片厚度: |
| Number of Layers: | 层数: |
| Thickness: | 厚度: |
| WELD GEOMETRY | 焊接几何参数 |
| Weld Width: | 焊接宽度: |
| Weld Length: | 焊接长度: |
| Contact Area: | 接触面积: |
| EQUIPMENT CONSTRAINTS | 设备约束 |
| Frequency: | 频率: |
| Max Power: | 最大功率: |
| INPUT SUMMARY | 输入汇总 |
| Back | 返回 |
| Next Step | 下一步 |
| Calculate Parameters | 计算参数 |
| No Engine | 无引擎 |
| Engine not connected. Cannot calculate. | 引擎未连接，无法计算。 |
| Calculation Error | 计算错误 |
| CALCULATION RESULTS | 计算结果 |
| No results yet. Run a calculation from the wizard. | 暂无结果，请先从向导运行计算。 |
| PARAMETERS | 参数 |
| RISK ASSESSMENT | 风险评估 |
| ALL PARAMETERS | 全部参数 |
| PARAMETER | 参数名 |
| VALUE | 数值 |
| SAFE MIN | 安全下限 |
| SAFE MAX | 安全上限 |
| VALIDATION | 验证 |
| RECOMMENDATIONS | 建议 |
| No validation data | 无验证数据 |
| No recommendations | 无建议 |
| REPORT GENERATION | 报告生成 |
| No recipe loaded. Run a calculation first. | 未加载配方，请先运行计算。 |
| RECIPE INFO | 配方信息 |
| PARAMETER PREVIEW | 参数预览 |
| SAFE RANGE | 安全范围 |
| EXPORT | 导出 |
| Export JSON | 导出 JSON |
| Export Excel | 导出 Excel |
| Export PDF | 导出 PDF |
| Export All | 全部导出 |
| CALCULATION HISTORY | 计算历史 |
| Refresh | 刷新 |
| ID | ID |
| APPLICATION | 应用类型 |
| MATERIALS | 材料 |
| DATE | 日期 |
| STATUS | 状态 |
| No history loaded. Click Refresh or run a calculation. | 无历史记录，请点击刷新或运行计算。 |
| Engine not connected. | 引擎未连接。 |
| No calculations found. | 未找到计算记录。 |
| Loaded %d records. | 已加载 %d 条记录。 |
| Error: %s | 错误: %s |
| SETTINGS | 设置 |
| APPEARANCE | 外观 |
| Theme: | 主题: |
| LANGUAGE | 语言 |
| Language: | 语言: |
| DEFAULT VALUES | 默认值 |
| Default Frequency: | 默认频率: |
| Default Max Power: | 默认最大功率: |
| FILE PATHS | 文件路径 |
| Report Output: | 报告输出目录: |
| Browse... | 浏览... |
| Select Report Directory | 选择报告目录 |
| About UltrasonicWeldMaster | 关于 UltrasonicWeldMaster |
| Application | 应用 |
| Materials | 材料 |
| Tooling | 工装 |
| Constraints | 约束 |

For `app_en.ts`: set each `<translation>` identical to `<source>` (identity).

**Step 3: Compile .ts to .qm**

```bash
pyside6-lrelease translations/app_zh_CN.ts -qm translations/app_zh_CN.qm
pyside6-lrelease translations/app_en.ts -qm translations/app_en.qm
```

**Step 4: Commit .ts files**

```bash
git add translations/app_zh_CN.ts translations/app_en.ts
git commit -m "feat(i18n): add Chinese and English translation files"
```

---

## Task 7: Wire up translation loading in main.py

**Files:**
- Modify: `main.py`

**Step 1: Add QTranslator loading based on system locale**

Add after `QApplication()` creation:
- Determine translations path (dev: relative to `__file__`, frozen: `sys._MEIPASS`)
- Detect system locale via `QLocale.system().language()`
- If English → `lang_code = "en"`, else → `lang_code = "zh_CN"` (Chinese default)
- Create QTranslator, load `"app_%s" % lang_code` from tr_path
- Install translator on app
- Also load Qt built-in translations

**Step 2: Verify**

Run: `python -c "import main; print('OK')"` → OK
Run: `pytest tests/ -v --tb=short` → all pass

**Step 3: Commit**

```bash
git add main.py
git commit -m "feat(i18n): load translations at startup based on system locale"
```

---

## Task 8: Update .spec file and create build script

**Files:**
- Modify: `UltrasonicWeldMaster.spec`
- Create: `scripts/build.py`

**Step 1: Update UltrasonicWeldMaster.spec**

Changes:
- Import `sys` and `glob` at top
- Add `qm_files = [(f, 'translations') for f in glob.glob('translations/*.qm')]` and append to datas
- Add 3 widget hidden imports (gauge_widget, risk_indicator, status_led)
- Wrap `BUNDLE` in `if sys.platform == 'darwin':` guard

**Step 2: Create scripts/build.py**

Build script that:
1. Finds all GUI .py source files
2. Runs `pyside6-lupdate` to update .ts files
3. Runs `pyside6-lrelease` to compile .ts → .qm
4. Runs `pyinstaller UltrasonicWeldMaster.spec --noconfirm`

Supports flags: `--skip-translations`, `--skip-build`

**Step 3: Verify build script**

```bash
python scripts/build.py --skip-build
```

Expected: .ts files updated, .qm files generated

**Step 4: Commit**

```bash
git add UltrasonicWeldMaster.spec scripts/build.py
git commit -m "feat: update spec for i18n + cross-platform, add build script"
```

---

## Task 9: Run PyInstaller build and verify

**Step 1: Full build**

```bash
python scripts/build.py
```

Expected: Build completes, `dist/UltrasonicWeldMaster.app` (macOS) or `dist/UltrasonicWeldMaster/` (Windows).

**Step 2: Launch and verify**

macOS: `open dist/UltrasonicWeldMaster.app`

Verify:
- App launches with correct language based on system locale
- Navigate to Settings → Language → switch → all panels update immediately
- All 5 panels display translated text

**Step 3: Final commit if needed**

```bash
git add -A
git commit -m "feat: complete i18n (zh/en) and cross-platform packaging"
```

---

## Task 10: Run all tests and final verification

**Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: 93+ passed, 0 failed

**Step 2: Verify all GUI imports**

```bash
python -c "
from ultrasonic_weld_master.gui.main_window import MainWindow
from ultrasonic_weld_master.gui.panels.input_wizard import InputWizardPanel
from ultrasonic_weld_master.gui.panels.result_panel import ResultPanel
from ultrasonic_weld_master.gui.panels.report_panel import ReportPanel
from ultrasonic_weld_master.gui.panels.history_panel import HistoryPanel
from ultrasonic_weld_master.gui.panels.settings_panel import SettingsPanel
print('All GUI modules OK')
"
```

**Step 3: Verify .qm files**

Run: `ls -la translations/*.qm`
Expected: `app_zh_CN.qm` and `app_en.qm` present
