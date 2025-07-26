from enum import Enum
from functools import total_ordering
from typing import Tuple, Optional, Union

class Suit(Enum):
    """
    Enumeración para representar los palos de una carta.
    """
    CLUB = "C"
    DIAMOND = "D"
    HEART = "H"
    SPADE = "S"

    def __int__(self):
        """
        Devuelve un valor entero para el palo.
        """
        if self == Suit.CLUB:
            return 1
        elif self == Suit.DIAMOND:
            return 2
        elif self == Suit.HEART:
            return 3
        else:  # Suit.SPADE
            return 4

    @classmethod
    def from_int(cls, value: int) -> Optional["Suit"]:
        """
        Devuelve el palo correspondiente al valor entero.
        """
        if value == 1:
            return cls.CLUB
        elif value == 2:
            return cls.DIAMOND
        elif value == 3:
            return cls.HEART
        elif value == 4:
            return cls.SPADE
        else:
            return None


class Rank(Enum):
    """
    Enumeración para representar los rangos de una carta.
    """
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "T"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"

    def __int__(self):
        """
        Devuelve un valor entero para el rango.
        """
        rank_values = {
            Rank.TWO: 2, Rank.THREE: 3, Rank.FOUR: 4, Rank.FIVE: 5, Rank.SIX: 6, Rank.SEVEN: 7,
            Rank.EIGHT: 8, Rank.NINE: 9, Rank.TEN: 10, Rank.JACK: 11, Rank.QUEEN: 12,
            Rank.KING: 13, Rank.ACE: 14
        }
        return rank_values[self]

    @classmethod
    def from_int(cls, value: int) -> Optional["Rank"]:
        """
        Devuelve el rango correspondiente al valor entero.
        """
        rank_values = {
            2: Rank.TWO, 3: Rank.THREE, 4: Rank.FOUR, 5: Rank.FIVE, 6: Rank.SIX, 7: Rank.SEVEN,
            8: Rank.EIGHT, 9: Rank.NINE, 10: Rank.TEN, 11: Rank.JACK, 12: Rank.QUEEN,
            13: Rank.KING, 14: Rank.ACE
        }
        return rank_values.get(value)

SUIT_SYMBOLS = {
    Suit.CLUB: "♣",
    Suit.DIAMOND: "♦",
    Suit.HEART: "♥",
    Suit.SPADE: "♠",
}

@total_ordering
class Card:
    """
    Clase que representa una carta de póker.

    Atributos:
        rank (Rank): El rango de la carta.
        suit (Suit): El palo de la carta.
    """

    def __init__(self, rank: Union[str, Rank], suit: Union[str, Suit]):
        """
        Inicializa una carta con un rango y un palo.

        Args:
            rank (Union[str, Rank]): El rango de la carta (e.g., "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A").
            suit (Union[str, Suit]): El palo de la carta (e.g., "C", "D", "H", "S").

        Raises:
            ValueError: Si el rango o el palo no son válidos.
        """
        if isinstance(rank, str):
            if rank not in Rank._value2member_map_:
                raise ValueError(f"Rango inválido: {rank}")
            rank = Rank(rank)
        if isinstance(suit, str):
            if suit not in Suit._value2member_map_:
                raise ValueError(f"Palo inválido: {suit}")
            suit = Suit(suit)
        self.rank = rank
        self.suit = suit

    def __str__(self, use_symbols: bool = False) -> str:
        """
        Devuelve una representación en cadena de la carta.

        Args:
            use_symbols (bool): Si se deben usar símbolos para los palos.

        Returns:
            str: Una cadena que representa la carta (e.g., "AC", "KH", "QD", "A♠", "K♥", "Q♦").
        """
        if use_symbols:
            return f"{self.rank.value}{SUIT_SYMBOLS[self.suit]}"
        else:
            return f"{self.rank.value}{self.suit.value}"

    def __repr__(self):
        """
        Devuelve una representación de la carta para debug y logging.
        """
        return f"Card(rank='{self.rank.value}', suit='{self.suit.value}')"

    def __eq__(self, other: object) -> bool:
        """
        Compara dos cartas para determinar si son iguales.

        Args:
            other (object): La otra carta a comparar.

        Returns:
            bool: True si las cartas son iguales, False de lo contrario.
        """
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self):
        """
        Devuelve un hash de la carta basado en su rango y palo.

        Returns:
            int: El hash de la carta.
        """
        return hash((self.rank, self.suit))

    def __lt__(self, other):
        """
        Compara dos cartas para determinar si una es menor que la otra.
        La comparación se basa en el valor del rango y, en caso de empate, en el valor del palo.
        """
        if isinstance(other, Card):
            if int(self.rank) != int(other.rank):
                return int(self.rank) < int(other.rank)
            else:
                return int(self.suit) < int(other.suit)
        return NotImplemented

    def to_tuple(self) -> Tuple[int, int]:
        """
        Devuelve la carta como una tupla de enteros (rango, palo).
        """
        return (int(self.rank), int(self.suit))

    @classmethod
    def from_tuple(cls, card_tuple: Tuple[int, int]) -> Optional["Card"]:
        """
        Crea una carta a partir de una tupla de enteros (rango, palo).
        """
        rank = Rank.from_int(card_tuple[0])
        suit = Suit.from_int(card_tuple[1])
        if rank and suit:
            return cls(rank, suit)
        else:
            return None