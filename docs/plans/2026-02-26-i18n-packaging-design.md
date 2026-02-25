# UltrasonicWeldMaster: i18n + Cross-Platform Packaging Design

## Overview

Add Chinese/English bilingual support using Qt QTranslator, and package the application as a directly runnable program for both macOS (.app) and Windows (.exe).

## Requirements

- **Languages**: Chinese (zh_CN) and English (en)
- **Default**: Chinese preferred; auto-detect system locale at startup
- **Switching**: Settings panel dropdown + auto-detect; live switch without restart
- **Packaging**: macOS .app bundle + Windows .exe (folder mode)
- **Build**: Single script handles translation compilation + PyInstaller build

---

## Part 1: i18n Architecture (Qt QTranslator)

### Translation Mechanism

All ~170 user-visible GUI strings converted to `self.tr("English text")` calls. English is the source language in code; Chinese translations stored in `.ts` XML files.

For non-QWidget contexts (e.g. StepIndicator labels), use:
```python
QCoreApplication.translate("ClassName", "text")
```

### File Structure

```
translations/
  app_zh_CN.ts    # Chinese translation source (XML, editable)
  app_zh_CN.qm    # Compiled binary (generated, gitignored)
  app_en.ts       # English (identity translation, optional)
  app_en.qm       # Compiled binary (generated, gitignored)
```

### Toolchain

1. `pyside6-lupdate -extensions py ultrasonic_weld_master/gui/*.py ultrasonic_weld_master/gui/panels/*.py -ts translations/app_zh_CN.ts`
   — Extract all `self.tr()` strings into .ts XML
2. Manually fill in `<translation>` tags in .ts file for Chinese
3. `pyside6-lrelease translations/app_zh_CN.ts -qm translations/app_zh_CN.qm`
   — Compile to binary .qm

### Language Loading (main.py)

```python
from PySide6.QtCore import QTranslator, QLocale

translator = QTranslator()
locale = QLocale.system().language()
if locale == QLocale.Chinese:
    translator.load("app_zh_CN", "translations")
else:
    translator.load("app_en", "translations")
app.installTranslator(translator)
```

### Live Language Switching

Each panel implements `retranslateUi()` method that re-sets all text:
```python
def retranslateUi(self):
    self._title.setText(self.tr("NEW CALCULATION"))
    self._back_btn.setText(self.tr("Back"))
    # ... etc
```

MainWindow coordinates the switch:
```python
def _on_language_changed(self, lang_code: str):
    app = QApplication.instance()
    app.removeTranslator(self._translator)
    self._translator.load("app_%s" % lang_code, translations_path)
    app.installTranslator(self._translator)
    self.retranslateUi()
    for panel in self._panels:
        panel.retranslateUi()
```

### Files to Modify (7 files)

| File | Changes |
|------|---------|
| `main.py` | Load QTranslator at startup based on system locale |
| `main_window.py` | `self.tr()` for menus, nav, status bar; add `retranslateUi()` |
| `input_wizard.py` | `self.tr()` for all labels/buttons; `retranslateUi()` |
| `result_panel.py` | `self.tr()` for group titles, headers; `retranslateUi()` |
| `report_panel.py` | `self.tr()` for titles, buttons; `retranslateUi()` |
| `history_panel.py` | `self.tr()` for titles, headers, status; `retranslateUi()` |
| `settings_panel.py` | `self.tr()` for all labels; add Language dropdown; `retranslateUi()` |

### Files to Create

| File | Purpose |
|------|---------|
| `translations/app_zh_CN.ts` | Chinese translation source XML |
| `translations/app_en.ts` | English identity translations (optional) |

---

## Part 2: Cross-Platform Packaging

### macOS (.app bundle)

Existing `UltrasonicWeldMaster.spec` already configured for macOS. Changes:
- Add `translations/*.qm` to `datas`
- Add 3 widget modules to `hiddenimports`
- Keep `BUNDLE` section for .app generation

### Windows (.exe folder)

Same `.spec` file with platform detection:
- Windows: `COLLECT` produces folder with `.exe` + dependencies
- Skip `BUNDLE` (macOS-only)
- Output: `dist/UltrasonicWeldMaster/UltrasonicWeldMaster.exe`

### Updated .spec datas

```python
datas = [
    ('ultrasonic_weld_master/plugins/material_db/materials.yaml', ...),
    ('ultrasonic_weld_master/plugins/knowledge_base/rules', ...),
    ('config.yaml', '.'),
    ('translations/*.qm', 'translations'),  # NEW
]
```

### Updated hiddenimports

Add:
```python
'ultrasonic_weld_master.gui.widgets.gauge_widget',
'ultrasonic_weld_master.gui.widgets.risk_indicator',
'ultrasonic_weld_master.gui.widgets.status_led',
```

### Build Script (scripts/build.py)

Single command `python scripts/build.py` that:
1. Runs `pyside6-lupdate` to extract/update .ts files
2. Runs `pyside6-lrelease` to compile .ts → .qm
3. Runs `pyinstaller UltrasonicWeldMaster.spec`

Supports `--skip-translations` flag to skip steps 1-2 if .qm files are current.

---

## Translation String Count Estimate

| Panel | Strings |
|-------|---------|
| main_window.py | ~23 |
| input_wizard.py | ~60 |
| result_panel.py | ~20 |
| report_panel.py | ~18 |
| history_panel.py | ~12 |
| settings_panel.py | ~15 |
| widgets | ~6 |
| **Total** | **~155** |
