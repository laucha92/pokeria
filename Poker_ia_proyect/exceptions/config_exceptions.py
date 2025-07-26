class ConfigException(Exception):
    """Base class for configuration exceptions."""
    pass

class ConfigLoadError(ConfigException):
    """Exception raised when the configuration file fails to load."""
    pass

class MissingConfigError(ConfigException):
    """Exception raised when a required configuration value is missing."""
    pass

class InvalidConfigError(ConfigException):
    """Exception raised when a configuration value is invalid."""
    pass