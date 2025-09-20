import os

PRODUCTION_DEFAULTS = {
    "LOG_LEVEL": "WARNING",
    "MAX_FILE_SIZE_MB": 500,
    "TIMEOUT_SECONDS": 300,
    "CIRCUIT_BREAKER_ENABLED": True,
    "CACHE_ENABLED": True,
    "RATE_LIMIT_ENABLED": True,
}

# Apply production settings
for key, value in PRODUCTION_DEFAULTS.items():
    if key not in os.environ:
        os.environ[key] = str(value)
