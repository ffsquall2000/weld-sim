# UltrasonicWeldMaster UI Redesign: Industrial Dashboard Style

## Overview

Full visual redesign of the PySide6 desktop GUI from generic prototype to professional dark industrial dashboard style (SCADA/oscilloscope aesthetic). Includes QPainter custom gauge widgets and bug fixes for broken features.

## Design Decisions

- **Style**: Dark industrial dashboard — oscilloscope orange accent on near-black background
- **Approach**: QSS deep rewrite + 3 custom QPainter widgets + layout restructure + bug fixes
- **Scope**: All 5 panels + themes + 3 custom widgets + 5 bug fixes

## Color Palette

| Role | Color | Hex |
|------|-------|-----|
| Background (window) | Near-black | `#0d1117` |
| Background (panel) | Dark grey | `#161b22` |
| Background (card) | Medium dark | `#21262d` |
| Accent / Data | Electronic orange | `#ff9800` |
| Accent hover | Bright orange | `#ffb74d` |
| Safe / Normal | Green | `#4caf50` |
| Warning | Amber | `#f0b429` |
| Danger | Red | `#f44336` |
| Text primary | Light grey | `#e6edf3` |
| Text secondary | Mid grey | `#8b949e` |
| Borders | Dark line | `#30363d` |

## Custom QPainter Widgets

### 1. GaugeWidget (half-arc gauge)
- 180° arc with tick marks and value pointer
- Green arc segment for safe window range
- Orange pointer for current value
- Digital readout below: value + unit in monospace font
- Used for: amplitude (μm), pressure (MPa), energy (J)

### 2. RiskIndicator (circular risk light)
- Circle with glow effect
- LOW = green glow, MEDIUM = orange glow, HIGH = red pulsing glow
- Risk type label below
- Used for: overweld_risk, underweld_risk, perforation_risk

### 3. StatusLED (small dot indicator)
- 8px circle on navigation items
- Grey = no data, Orange = has data, Green = exported
- Used on nav list items to show panel state

## Layout Structure

```
┌──────────────────────────────────────────────────┐
│  ⚡ ULTRASONIC WELD MASTER          v0.1.0       │  ← Title bar #0d1117
├──────┬───────────────────────────────────────────┤
│      │                                           │
│ ◉ New│   Content area varies by panel            │
│ ◎ Res│                                           │
│ ◎ Rep│   Result panel shows:                     │
│ ◎ His│   3x GaugeWidget in a row                 │
│ ◎ Set│   3x RiskIndicator below                  │
│      │   Parameter table + recommendations       │
│      │                                           │
├──────┴───────────────────────────────────────────┤
│  Ready │ Li-Battery │ Al→Cu │ 20kHz │ 3500W      │  ← Status bar
└──────────────────────────────────────────────────┘
```

## Panel Redesign Details

### InputWizardPanel
- Each step as a dark card (#21262d) with rounded corners (8px)
- Orange progress bar instead of default blue
- Form labels in secondary text color, inputs with dark bg + orange focus border
- Step indicators: numbered circles connected by line

### ResultPanel
- Top row: 3 GaugeWidgets (amplitude, pressure, energy)
- Middle: 3 RiskIndicators in a row
- Bottom: parameter table with alternating dark rows + recommendations

### ReportPanel
- Structured preview using QTableWidget instead of plain QTextEdit
- Export buttons as icon + text cards

### HistoryPanel
- Fix SQL bug (recipe_id → id column)
- Auto-refresh after calculation
- Dark table with orange header

### SettingsPanel
- Connect default frequency/power to wizard
- Persist theme preference to config

## Bug Fixes

1. **History SQL**: `recipe_id` column → correct column name from schema
2. **Settings→Wizard connection**: defaults propagate to InputWizardPanel
3. **Report preview**: QTextEdit → structured QTableWidget display
4. **Auto-refresh History**: emit signal after calculation completes
5. **Input validation**: red border on out-of-range values before Calculate

## Files to Create/Modify

### Create:
- `ultrasonic_weld_master/gui/widgets/gauge_widget.py` — GaugeWidget QPainter
- `ultrasonic_weld_master/gui/widgets/risk_indicator.py` — RiskIndicator QPainter
- `ultrasonic_weld_master/gui/widgets/status_led.py` — StatusLED QPainter

### Modify:
- `ultrasonic_weld_master/gui/themes.py` — complete dark theme rewrite
- `ultrasonic_weld_master/gui/main_window.py` — layout + status bar + LED indicators
- `ultrasonic_weld_master/gui/panels/input_wizard.py` — card layout + validation
- `ultrasonic_weld_master/gui/panels/result_panel.py` — gauge widgets + risk indicators
- `ultrasonic_weld_master/gui/panels/report_panel.py` — structured preview
- `ultrasonic_weld_master/gui/panels/history_panel.py` — fix SQL + auto-refresh
- `ultrasonic_weld_master/gui/panels/settings_panel.py` — connect to wizard + persist
