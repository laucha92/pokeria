class GameException(Exception):
    """Base class for exceptions in the game logic."""
    pass

class InvalidMoveError(GameException):
    """Exception raised when a player attempts an invalid move."""
    pass

class InsufficientFundsError(GameException):
    """Exception raised when a player does not have enough funds to make a bet."""
    pass