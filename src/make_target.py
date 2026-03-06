def bp_stage(sys_bp: float, dia_bp: float) -> str:
    # Based on AHA categories (simplified)
    if sys_bp > 180 or dia_bp > 120:
        return "Severe"
    if sys_bp >= 140 or dia_bp >= 90:
        return "Stage2"
    if (130 <= sys_bp <= 139) or (80 <= dia_bp <= 89):
        return "Stage1"
    if (120 <= sys_bp <= 129) and (dia_bp < 80):
        return "Elevated"
    return "Normal"
