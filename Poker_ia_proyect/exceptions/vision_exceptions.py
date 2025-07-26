class VisionException(Exception):
    """Base class for exceptions in the vision module."""
    pass

class CameraError(VisionException):
    """Exception raised when there is an issue with the camera."""
    pass

class CardDetectionError(VisionException):
    """Exception raised when card detection fails."""
    pass

class ChipDetectionError(VisionException):
    """Exception raised when chip detection fails."""
    pass