"""UltrasonicWeldMaster command-line interface."""
from __future__ import annotations

import argparse
import json
import os
import sys


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="ultrasonic-weld-master",
        description="Ultrasonic metal welding parameter auto-generation tool",
    )
    parser.add_argument("--version", action="version", version="UltrasonicWeldMaster v0.1.0")

    sub = parser.add_subparsers(dest="command")

    calc = sub.add_parser("calculate", help="Calculate welding parameters")
    calc.add_argument("--application", default="li_battery_tab",
                      choices=["li_battery_tab", "li_battery_busbar", "li_battery_collector", "general_metal"])
    calc.add_argument("--upper-material", default="Al")
    calc.add_argument("--upper-thickness", type=float, default=0.012)
    calc.add_argument("--upper-layers", type=int, default=40)
    calc.add_argument("--lower-material", default="Cu")
    calc.add_argument("--lower-thickness", type=float, default=0.3)
    calc.add_argument("--width", type=float, default=5.0)
    calc.add_argument("--length", type=float, default=25.0)
    calc.add_argument("--frequency", type=float, default=20.0)
    calc.add_argument("--max-power", type=float, default=3500)
    calc.add_argument("--output", help="Output directory for reports")
    calc.add_argument("--format", choices=["json", "excel", "pdf", "all"], default="json")

    return parser


def _do_calculate(args):
    from ultrasonic_weld_master.plugins.material_db.plugin import MaterialDBPlugin
    from ultrasonic_weld_master.plugins.li_battery.plugin import LiBatteryPlugin
    from ultrasonic_weld_master.plugins.general_metal.plugin import GeneralMetalPlugin
    from ultrasonic_weld_master.plugins.reporter.json_exporter import JsonExporter
    from ultrasonic_weld_master.plugins.reporter.excel_generator import ExcelGenerator
    from ultrasonic_weld_master.plugins.reporter.pdf_generator import PdfGenerator

    mat_db = MaterialDBPlugin()
    mat_db.activate({})

    inputs = {
        "application": args.application,
        "upper_material_type": args.upper_material,
        "upper_thickness_mm": args.upper_thickness,
        "upper_layers": args.upper_layers,
        "lower_material_type": args.lower_material,
        "lower_thickness_mm": args.lower_thickness,
        "weld_width_mm": args.width,
        "weld_length_mm": args.length,
        "frequency_khz": args.frequency,
        "max_power_w": args.max_power,
    }

    if args.application.startswith("li_battery"):
        plugin = LiBatteryPlugin()
        plugin.activate({"config": None, "event_bus": None, "logger": None, "material_db": mat_db})
    else:
        plugin = GeneralMetalPlugin()
        plugin.activate({"material_db": mat_db})

    recipe = plugin.calculate_parameters(inputs)
    validation = plugin.validate_parameters(recipe)

    # Print results
    print("=" * 60)
    print("  Welding Parameter Calculation Result")
    print("=" * 60)
    print("  Recipe ID:   %s" % recipe.recipe_id)
    print("  Application: %s" % recipe.application)
    print()
    print("  --- Parameters ---")
    for k, v in recipe.parameters.items():
        sw = recipe.safety_window.get(k)
        if sw and len(sw) >= 2:
            print("  %-18s %8s  [%s - %s]" % (k, v, sw[0], sw[1]))
        else:
            print("  %-18s %8s" % (k, v))
    print()
    print("  --- Risk Assessment ---")
    for k, v in recipe.risk_assessment.items():
        print("  %-20s %s" % (k, v.upper()))
    print()
    print("  --- Validation: %s ---" % validation.status.value.upper())
    for msg in validation.messages:
        print("  %s" % msg)
    if not validation.messages:
        print("  All checks passed.")
    print()
    print("  --- Recommendations ---")
    for r in recipe.recommendations:
        print("  - %s" % r)
    print("=" * 60)

    # Export if requested
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        fmt = args.format
        if fmt in ("json", "all"):
            path = JsonExporter().export(recipe, validation, args.output)
            print("  JSON report: %s" % path)
        if fmt in ("excel", "all"):
            path = ExcelGenerator().export(recipe, validation, args.output)
            print("  Excel report: %s" % path)
        if fmt in ("pdf", "all"):
            path = PdfGenerator().export(recipe, validation, args.output)
            print("  PDF report: %s" % path)

    return 0


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "calculate":
        return _do_calculate(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
