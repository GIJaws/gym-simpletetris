def format_value(v):
    if isinstance(v, (list, tuple)):
        return ", ".join([format_value(x) for x in v])
    return f"{v:.0f}" if isinstance(v, int) or v.is_integer() else f"{v:.6f}"


def iterate_nested_dict(d, prefix=""):
    """Helper function to iterate through nested dictionaries and handle nested keys."""
    for k, v in d.items():
        if isinstance(v, dict):
            # Recursively iterate through nested dictionaries
            yield from iterate_nested_dict(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v
