def recommend(stage: str, patient: dict) -> list[str]:
    recs = []

    if stage in ["Normal", "Elevated"]:
        recs += [
            "Maintain a balanced diet and regular exercise.",
            "Monitor BP occasionally to catch early changes.",
        ]
    if stage == "Stage1":
        recs += [
            "Reduce sodium and ultra-processed foods.",
            "Increase physical activity and improve sleep.",
            "Schedule a clinical follow-up for confirmation."
        ]
    if stage == "Stage2":
        recs += [
            "Clinical follow-up soon is recommended.",
            "Regular home monitoring + follow medical advice.",
        ]
    if stage == "Severe":
        recs += [
            "This is in a danger zone. If symptoms exist, seek urgent medical care.",
            "Recheck BP after rest and contact a professional ASAP."
        ]

    if patient.get("smoke", 0) == 1:
        recs.append("Smoking increases cardiovascular risk — quitting is a big win.")
    if patient.get("active", 1) == 0:
        recs.append("Try building up to ~150 minutes/week of moderate activity.")

    return recs