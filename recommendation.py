def build_reason(item):
    """
    Constructs a human-readable explanation of why a specific assessment is recommended.
    """
    return (
        f"Recommended because it evaluates "
        f"{', '.join(item.get('keys', []))} "
        f"and is suitable for "
        f"{', '.join(item.get('job_levels', []))}."
    )
