import random
import logging
from typing import List, Optional
from game.card import Card  # Importa la clase Card

logger = logging.getLogger(__name__)

class DeckEmptyError(Exception):
    """Excepción lanzada cuando se intenta repartir de una baraja vacía."""
    pass

class Deck:
    """
    Clase que representa una baraja de cartas estándar.

    Atributos:
        cards (List[Card]): Lista de objetos Card que representan las cartas en la baraja.
        initial_cards (List[Card]): Copia de las cartas iniciales para el método reset.
    """

    def __init__(self):
        """
        Inicializa una baraja de cartas estándar, creando todas las combinaciones de palos y rangos.
        """
        suits = ["C", "D", "H", "S"]  # Clubs, Diamonds, Hearts, Spades
        ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]  # 10 is T
        self.cards: List[Card] = [Card(rank, suit) for suit in suits for rank in ranks]
        self.initial_cards: List[Card] = self.cards[:]  # Guarda una copia para el reset
        self.shuffle()
        logger.info("Baraja creada e inicializada.")

    def shuffle(self):
        """
        Mezcla las cartas en la baraja aleatoriamente.
        """
        random.shuffle(self.cards)
        logger.debug("Baraja mezclada.")

    def deal(self, num_cards: int, from_top: bool = True) -> List[Card]:
        """
        Reparte un número específico de cartas desde la parte superior o inferior de la baraja.

        Args:
            num_cards (int): El número de cartas a repartir.
            from_top (bool): True para repartir desde la parte superior (default), False para repartir desde el fondo.

        Returns:
            List[Card]: Una lista de objetos Card que representan las cartas repartidas.

        Raises:
            DeckEmptyError: Si no hay suficientes cartas en la baraja para repartir.
        """
        if num_cards > len(self.cards):
            logger.warning(f"Se intentaron repartir {num_cards} cartas, pero solo quedan {len(self.cards)} en la baraja.")
            raise DeckEmptyError("No hay suficientes cartas en la baraja para repartir.")

        dealt_cards: List[Card] = []
        if from_top:
            for _ in range(num_cards):
                dealt_cards.append(self.cards.pop(0))
        else:
            for _ in range(num_cards):
                dealt_cards.append(self.cards.pop())  # Pop desde el final (fondo)
        logger.debug(f"Se repartieron {num_cards} cartas desde {'arriba' if from_top else 'abajo'}.")
        return dealt_cards

    def reset(self):
        """
        Reinicia la baraja a su estado inicial y la mezcla.
        """
        self.cards = self.initial_cards[:]  # Copia las cartas iniciales
        self.shuffle()
        logger.info("Baraja reiniciada y mezclada.")

    def remaining(self) -> int:
        """
        Devuelve el número de cartas restantes en la baraja.

        Returns:
            int: El número de cartas restantes.
        """
        return len(self.cards)

    def peek(self, position: int = 0) -> Optional[Card]:
        """
        Permite mirar una carta en una posición específica sin retirarla.

        Args:
            position (int): La posición de la carta a mirar (0 es la primera carta).

        Returns:
            Optional[Card]: La carta en la posición especificada, o None si la posición es inválida.
        """
        if 0 <= position < len(self.cards):
            return self.cards[position]
        else:
            logger.warning(f"Posición inválida para peek: {position}. La baraja tiene {len(self.cards)} cartas.")
            return None

    def cut(self, position: int):
        """
        Corta la baraja en una posición específica.

        Args:
            position (int): La posición donde se corta la baraja.
        """
        if 0 < position < len(self.cards):
            self.cards = self.cards[position:] + self.cards[:position]
            logger.info(f"Baraja cortada en la posición {position}.")
        else:
            logger.warning(f"Posición de corte inválida: {position}. Debe estar entre 1 y {len(self.cards) - 1}.")

    def __str__(self):
        """
        Devuelve una representación en cadena de la baraja, mostrando las cartas restantes.

        Returns:
            str: Una cadena que representa la baraja.
        """
        card_strings = [str(card) for card in self.cards]
        return f"Deck with {self.remaining()} cards: {', '.join(card_strings)}"