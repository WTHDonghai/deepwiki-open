import logging
import os
import sys
from pathlib import Path

class IgnoreLogChangeDetectedFilter(logging.Filter):
    def filter(self, record: logging.LogRecord):
        return "Detected file change in" not in record.getMessage()

class SafeUnicodeFilter(logging.Filter):
    """Filter to handle Unicode encoding issues by replacing problematic characters."""
    def filter(self, record: logging.LogRecord):
        try:
            # Try to encode the message to detect encoding issues
            record.getMessage().encode('gbk')
            return True
        except UnicodeEncodeError:
            # Replace problematic Unicode characters with safe alternatives
            original_msg = record.getMessage()
            safe_msg = original_msg.replace('🔍', '[DEBUG]').replace('😭', '[ERROR]')
            # Update the record's message
            record.msg = safe_msg
            record.args = ()
            return True

def setup_logging(format: str = None):
    """
    Configure logging for the application.
    Reads LOG_LEVEL and LOG_FILE_PATH from environment (defaults: INFO, logs/application.log).
    Ensures log directory exists, and configures both file and console handlers.
    """
    # Determine log directory and default file path
    base_dir = Path(__file__).parent
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    default_log_file = log_dir / "application.log"

    # Get log level and file path from environment
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_file_path = Path(os.environ.get(
        "LOG_FILE_PATH", str(default_log_file)))

    # ensure log_file_path is within the project's logs directory to prevent path traversal
    log_dir_resolved = log_dir.resolve()
    resolved_path = log_file_path.resolve()
    if not str(resolved_path).startswith(str(log_dir_resolved) + os.sep):
        raise ValueError(
            f"LOG_FILE_PATH '{log_file_path}' is outside the trusted log directory '{log_dir_resolved}'"
        )
    # Ensure parent dirs exist for the log file
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    # Create console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Set UTF-8 encoding for console output on Windows
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass  # Fallback if reconfigure fails
    
    # Configure logging handlers and format
    logging.basicConfig(
        level=log_level,
        format = format or "%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s",
        handlers=[
            logging.FileHandler(resolved_path, encoding='utf-8'),
            console_handler
        ],
        force=True
    )
    
    # Add filters to all handlers
    for handler in logging.getLogger().handlers:
        handler.addFilter(IgnoreLogChangeDetectedFilter())
        handler.addFilter(SafeUnicodeFilter())

    # Initial debug message to confirm configuration
    logger = logging.getLogger(__name__)
    logger.debug(f"Log level set to {log_level_str}, log file: {resolved_path}")
