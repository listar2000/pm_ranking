import json
import logging
import sys

def parse_json_or_eval(val, expect_type=None):
    """
    Try to parse a value as JSON, then as a Python literal, and check type.
    If parsing fails or type does not match, return None.
    """
    try:
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
            except Exception:
                import ast
                parsed = ast.literal_eval(val)
            if expect_type and not isinstance(parsed, expect_type):
                raise ValueError(f"Expected {expect_type}, got {type(parsed)}")
            return parsed
        elif expect_type and not isinstance(val, expect_type):
            raise ValueError(f"Expected {expect_type}, got {type(val)}")
        else:
            return val
    except Exception:
        raise ValueError(f"Failed to parse value: {val}")


def get_logger(name: str = "pm_rank.data"):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger