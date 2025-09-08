import re
from typing import Optional


def parse_value(input_str: str, value_key: str) -> Optional[str]:
    """
    Extracts the value of a given key in the input string.
    Supports key:value or key = value formats, with optional quotes.
    Returns the value with quotes and extra spaces removed.
    """
    # Regex pattern: key : or = value (quoted or unquoted)
    pattern = rf"{re.escape(value_key)}\s*[:=]\s*(\"[^\"]*\"|'[^']*'|[^,]+)"
    match = re.search(pattern, input_str)
    if match:
        value = match.group(1).strip()  # remove leading/trailing whitespace
        # Remove surrounding quotes if present
        value = re.sub(r"^['\"]|['\"]$", "", value)
        return value.strip()
    return None


