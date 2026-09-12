def take_action(risk_level):
    """
    Decides the security action based on the calculated risk level.
    """

    if risk_level == "LOW":
        action = "ALLOW"
        message = "Interaction appears safe."

    elif risk_level == "MEDIUM":
        action = "VERIFY"
        message = "Additional verification is required."

    elif risk_level == "HIGH":
        action = "ALERT"
        message = "Suspicious interaction detected. Security alert generated."

    elif risk_level == "CRITICAL":
        action = "BLOCK"
        message = "Critical threat detected. Interaction should be blocked."

    else:
        action = "REVIEW"
        message = "Unknown risk level. Manual review required."

    return {
        "action": action,
        "message": message
    }
