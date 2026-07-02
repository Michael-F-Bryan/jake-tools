from __future__ import annotations

from .models import AuditStage


def validate_stage(stage: AuditStage, records: list[object]) -> list[str]:
    errors: list[str] = []
    if not records:
        errors.append(f"{stage.value}: no records")
    return errors
