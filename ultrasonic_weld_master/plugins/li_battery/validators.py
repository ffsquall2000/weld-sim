"""Three-validator system for welding parameter verification."""
from __future__ import annotations

from typing import Any
from ultrasonic_weld_master.core.models import WeldRecipe, ValidationResult, ValidationStatus


class PhysicsValidator:
    AMPLITUDE_RANGE = (10, 70)
    PRESSURE_RANGE = (0.1, 30.0)  # interface pressure (MPa), not air cylinder pressure

    def validate(self, recipe: WeldRecipe) -> dict:
        messages = []
        status = "pass"
        p = recipe.parameters

        amp = p.get("amplitude_um", 0)
        if not (self.AMPLITUDE_RANGE[0] <= amp <= self.AMPLITUDE_RANGE[1]):
            messages.append("Amplitude %.1f um outside range %s" % (amp, self.AMPLITUDE_RANGE))
            status = "fail"
        elif amp > 55:
            messages.append("Amplitude %.1f um is high, risk of material damage" % amp)
            status = "warning"

        pres = p.get("pressure_mpa", 0)
        if not (self.PRESSURE_RANGE[0] <= pres <= self.PRESSURE_RANGE[1]):
            messages.append("Pressure %.3f MPa outside range %s" % (pres, self.PRESSURE_RANGE))
            status = "fail"

        area = recipe.inputs.get("weld_area_mm2", 1)
        energy = p.get("energy_j", 0)
        if area > 0:
            energy_density = energy / area
            if not (0.05 <= energy_density <= 15.0):
                messages.append("Energy density %.2f J/mm2 outside normal range" % energy_density)
                status = "warning" if status == "pass" else status

        return {"status": status, "messages": messages}


class SafetyValidator:
    def validate(self, recipe: WeldRecipe) -> dict:
        messages = []
        status = "pass"
        risks = recipe.risk_assessment

        high_risks = [k for k, v in risks.items() if isinstance(v, str) and v == "high"]
        if len(high_risks) >= 2:
            messages.append("Multiple high risks: %s" % high_risks)
            status = "fail"
        elif high_risks:
            messages.append("High risk detected: %s" % high_risks)
            status = "warning"

        if risks.get("overweld_risk") == "high":
            messages.append("Overweld risk high: possible separator damage in li-battery")
            status = "fail" if status != "fail" else status

        if risks.get("perforation_risk") == "high":
            messages.append("Perforation risk high: foil stack may be punctured")
            if status == "pass":
                status = "warning"

        return {"status": status, "messages": messages}


class ConsistencyValidator:
    def validate(self, recipe: WeldRecipe) -> dict:
        messages = []
        status = "pass"
        p = recipe.parameters

        energy = p.get("energy_j", 0)
        time_ms = p.get("time_ms", 1)
        if time_ms > 0:
            implied_power = energy / (time_ms / 1000)
            max_power = recipe.inputs.get("max_power_w", 5000)
            if max_power and implied_power > max_power * 1.05:
                messages.append("Implied power %.0fW exceeds max %sW" % (implied_power, max_power))
                status = "warning"

        area = recipe.inputs.get("weld_area_mm2", 0)
        if area > 0:
            p_n = p.get("pressure_n", 0)
            p_mpa = p.get("pressure_mpa", 0)
            expected_n = p_mpa * area
            if p_n > 0 and abs(p_n - expected_n) / max(p_n, 1) > 0.1:
                messages.append("Pressure inconsistency: %.0fN vs %.0fN" % (p_n, expected_n))
                status = "warning" if status == "pass" else status

        return {"status": status, "messages": messages}


def validate_recipe(recipe: WeldRecipe) -> ValidationResult:
    validators = {
        "physics": PhysicsValidator(),
        "safety": SafetyValidator(),
        "consistency": ConsistencyValidator(),
    }
    results = {}
    overall = ValidationStatus.PASS

    for name, validator in validators.items():
        result = validator.validate(recipe)
        results[name] = result
        if result["status"] == "fail":
            overall = ValidationStatus.FAIL
        elif result["status"] == "warning" and overall != ValidationStatus.FAIL:
            overall = ValidationStatus.WARNING

    all_messages = []
    for name, r in results.items():
        for msg in r["messages"]:
            all_messages.append("[%s] %s" % (name, msg))

    return ValidationResult(status=overall, validators=results, messages=all_messages)
