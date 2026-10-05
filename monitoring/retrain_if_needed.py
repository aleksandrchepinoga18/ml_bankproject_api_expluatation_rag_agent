def retrain_if_needed():
    """Run drift checks, but do not auto-retrain on unverified drift signals."""
    print("Checking retraining signals...")

    from monitoring.check_data_drift import check_data_drift
    from monitoring.check_score_drift import check_score_drift

    feature_drift = check_data_drift()
    score_drift = check_score_drift()
    retrain_candidate = bool(feature_drift or score_drift)

    if retrain_candidate:
        print(
            "Drift candidate detected. Automatic retraining is disabled until "
            "the signal is manually verified."
        )
    else:
        print("Retraining is not required")

    return {
        "feature_drift": bool(feature_drift),
        "score_drift": bool(score_drift),
        "retrain_candidate": retrain_candidate,
        "auto_retrain_started": False,
    }


if __name__ == "__main__":
    retrain_if_needed()
