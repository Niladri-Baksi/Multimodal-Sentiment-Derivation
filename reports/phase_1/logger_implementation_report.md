# Phase 1 - Step 3: Shared Logging Utility

## Implementation

Implemented `configure_logging` in `src/sentiment_derivation/utils/logger.py` using only Python's standard `logging` library. Importing the module does not configure or modify the root logger.

The utility configures a named logger with a console `StreamHandler` and a consistent format containing timestamp, severity level, logger name, and message. It disables propagation for that named logger to prevent its console output from being duplicated by root handlers. Repeated calls reuse the utility-owned handler and update its level and formatter.

## Public function contract

| Function | Input | Output | Errors |
|---|---|---|---|
| `configure_logging(name: str, level: str \| int = "INFO") -> logging.Logger` | Logger name and one of DEBUG, INFO, WARNING, ERROR, CRITICAL (case-insensitive string or corresponding standard numeric constant) | Configured named logger | `ValueError` for unsupported levels |

The console handler defaults to standard error, as provided by `logging.StreamHandler`. The formatter is `%(asctime)s | %(levelname)s | %(name)s | %(message)s`, with timestamps formatted as `%Y-%m-%d %H:%M:%S`.

## Tests created

Created `tests/test_logger.py` with 10 pytest cases covering:

1. Logger creation and named logger properties.
2. Each of the five supported log levels.
3. Invalid log-level rejection.
4. Consistent formatter and timestamp format.
5. Repeated configuration without duplicate utility handlers.
6. Import/reload without root logger configuration.

Tests use unique logger names and clean up their handlers after each applicable test.

## Test execution results

Command run:

```text
python -m pytest tests/test_logger.py -v
```

| Result | Count / detail |
|---|---|
| Passed | 10 |
| Failed | 0 |
| Not executed | 0 |
| Warnings | 2: installed AnyIO pytest plugin deprecation warning; pytest cache warning because access to `.pytest_cache` was denied. |

## Limitations

- The utility provides console logging only; file logging and configurable format strings are not included.
- It does not configure the root logger. Applications should call `configure_logging` at an appropriate entry point and pass or retrieve named loggers for their modules.
- Logger configuration is process-local and does not establish a repository-wide logging policy until callers adopt it.
