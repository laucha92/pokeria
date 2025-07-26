class AIException(Exception):
    """Base class for exceptions in the AI module."""
    pass

class ModelLoadError(AIException):
    """Exception raised when the AI model fails to load."""
    pass

class InvalidActionError(AIException):
    """Exception raised when the AI selects an invalid action."""
    pass