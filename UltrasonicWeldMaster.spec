# -*- mode: python ; coding: utf-8 -*-
import sys
from PyInstaller.utils.hooks import collect_all

datas = [
    ('ultrasonic_weld_master/plugins/material_db/materials.yaml',
     'ultrasonic_weld_master/plugins/material_db'),
    ('ultrasonic_weld_master/plugins/knowledge_base/rules',
     'ultrasonic_weld_master/plugins/knowledge_base/rules'),
    ('config.yaml', '.'),
    ('translations/*.qm', 'translations'),
]
binaries = []
hiddenimports = [
    'yaml',
    # Core
    'ultrasonic_weld_master.core.models',
    'ultrasonic_weld_master.core.engine',
    'ultrasonic_weld_master.core.event_bus',
    'ultrasonic_weld_master.core.logger',
    'ultrasonic_weld_master.core.config',
    'ultrasonic_weld_master.core.database',
    'ultrasonic_weld_master.core.plugin_api',
    'ultrasonic_weld_master.core.plugin_manager',
    # Plugins
    'ultrasonic_weld_master.plugins.material_db.plugin',
    'ultrasonic_weld_master.plugins.li_battery.plugin',
    'ultrasonic_weld_master.plugins.li_battery.calculator',
    'ultrasonic_weld_master.plugins.li_battery.physics',
    'ultrasonic_weld_master.plugins.li_battery.validators',
    'ultrasonic_weld_master.plugins.general_metal.plugin',
    'ultrasonic_weld_master.plugins.general_metal.calculator',
    'ultrasonic_weld_master.plugins.knowledge_base.plugin',
    'ultrasonic_weld_master.plugins.reporter.plugin',
    'ultrasonic_weld_master.plugins.reporter.json_exporter',
    'ultrasonic_weld_master.plugins.reporter.excel_generator',
    'ultrasonic_weld_master.plugins.reporter.pdf_generator',
    # GUI
    'ultrasonic_weld_master.gui.main_window',
    'ultrasonic_weld_master.gui.themes',
    'ultrasonic_weld_master.gui.panels.input_wizard',
    'ultrasonic_weld_master.gui.panels.result_panel',
    'ultrasonic_weld_master.gui.panels.report_panel',
    'ultrasonic_weld_master.gui.panels.history_panel',
    'ultrasonic_weld_master.gui.panels.settings_panel',
    'ultrasonic_weld_master.gui.widgets.gauge_widget',
    'ultrasonic_weld_master.gui.widgets.risk_indicator',
    'ultrasonic_weld_master.gui.widgets.status_led',
]
tmp_ret = collect_all('PySide6')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('openpyxl')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('reportlab')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='UltrasonicWeldMaster',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='UltrasonicWeldMaster',
)
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='UltrasonicWeldMaster.app',
        icon=None,
        bundle_identifier='com.ultrasonicweldmaster.app',
    )
