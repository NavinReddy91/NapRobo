"""
Security Utilities
==================
Provides safe file operations, path validation, and input sanitization.
"""

import os
import json
import logging
from typing import Any, Dict, Optional


# Configure logging
logger = logging.getLogger(__name__)


def validate_path(filepath: str, base_dir: Optional[str] = None) -> str:
    """
    Validate and sanitize a file path to prevent path traversal attacks.

    Args:
        filepath: The path to validate
        base_dir: Optional base directory to restrict paths within

    Returns:
        The validated absolute path

    Raises:
        ValueError: If path is invalid or escapes base directory
    """
    # Resolve to absolute path
    abs_path = os.path.realpath(os.path.abspath(filepath))

    # If base_dir is specified, ensure path is within it
    if base_dir:
        base_abs = os.path.realpath(os.path.abspath(base_dir))
        if not abs_path.startswith(base_abs + os.sep) and abs_path != base_abs:
            raise ValueError(f"Path '{filepath}' escapes base directory '{base_dir}'")

    return abs_path


def safe_json_load(filepath: str, expected_keys: Optional[list] = None) -> Dict:
    """
    Safely load and validate a JSON file.

    Args:
        filepath: Path to JSON file
        expected_keys: Optional list of required keys to validate

    Returns:
        Parsed JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
        ValueError: If required keys are missing
    """
    validated_path = validate_path(filepath)

    with open(validated_path, "r") as f:
        data = json.load(f)

    # Validate structure if expected_keys provided
    if expected_keys:
        missing_keys = [k for k in expected_keys if k not in data]
        if missing_keys:
            raise ValueError(f"JSON file missing required keys: {missing_keys}")

    return data


def safe_json_save(filepath: str, data: Dict, secure: bool = False) -> None:
    """
    Safely save data to a JSON file with optional restricted permissions.

    Args:
        filepath: Path to save JSON file
        data: Data to save
        secure: If True, use restricted file permissions (0600)
    """
    validated_path = validate_path(filepath)

    # Create directory if needed
    os.makedirs(os.path.dirname(validated_path) or ".", exist_ok=True)

    if secure:
        # Use restricted permissions for sensitive files
        fd = os.open(validated_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
    else:
        with open(validated_path, "w") as f:
            json.dump(data, f, indent=2)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to remove dangerous characters.

    Args:
        filename: The filename to sanitize

    Returns:
        Sanitized filename safe for use
    """
    # Remove path separators and dangerous characters
    dangerous_chars = ["/", "\\", "..", ":", "*", "?", '"', "<", ">", "|"]
    sanitized = filename
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, "_")

    # Limit length
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[: 255 - len(ext)] + ext

    return sanitized


def log_safe_error(error: Exception, context: str = "") -> None:
    """
    Log an error safely without exposing sensitive information.

    Args:
        error: The exception to log
        context: Optional context about where the error occurred
    """
    # Log generic message to stdout
    error_msg = (
        f"Error in {context}: {error.__class__.__name__}"
        if context
        else f"Error: {error.__class__.__name__}"
    )
    logger.warning(error_msg)

    # Log full details to debug log only
    logger.debug(f"Full error details: {str(error)}", exc_info=True)
