import logging
from typing import List, Optional, Callable
from enum import Enum
import copy
from game.card import Card  # Importar la clase Card
# from utils.serialization import serialize_player, deserialize_player  # Ejemplo de separación de serialización

logger = logging.getLogger(__name__)

# Definir excepciones personalizadas
class PlayerError(Exception):
    """Base class for exceptions in this module."""
    pass

class PlayerInactiveError(PlayerError):
    """Raised when trying to perform an action on an inactive player."""
    pass

class InvalidCardError(PlayerError):
    """Raised when an invalid card is used."""
    pass

class InvalidActionError(PlayerError):
    """Raised when an invalid action is attempted."""
    pass

# Definir Enum para la posición
class Position(Enum):
    DEALER = "dealer"
    BB = "bb"
    SB = "sb"

class Player:
    """
    Clase que representa a un jugador de póker.

    Atributos:
        name (str): El nombre del jugador.
        chips (int): La cantidad de fichas del jugador.
        hole_cards (List[Card]): Las cartas en la mano del jugador.
        amount_invested (int): La cantidad de fichas que el jugador ha invertido en la ronda actual.
        in_hand (bool): Indica si el jugador está en la mano.
        is_all_in (bool): Indica si el jugador está all-in.
        folded (bool): Indica si el jugador se ha retirado de la mano.
        position (Position, optional): La posición del jugador en la mesa (e.g., Position.DEALER).
    """
    def __init__(self, name: str, initial_chips: int, position: Optional[Position] = None):
        """
        Inicializa un jugador.

        Args:
            name (str): El nombre del jugador.
            initial_chips (int): La cantidad inicial de fichas del jugador.
            position (Position, optional): La posición del jugador en la mesa (e.g., Position.DEALER).
        """
        if initial_chips < 0:
            raise ValueError("La cantidad inicial de fichas debe ser no negativa.")

        self.name: str = name
        self._chips: int = initial_chips  # Atributo protegido
        self._hole_cards: List[Card] = []  # Atributo protegido
        self._amount_invested: int = 0  # Atributo protegido
        self.in_hand: bool = True
        self.is_all_in: bool = False
        self.folded: bool = False
        self.position: Optional[Position] = position
        self._callbacks = {}  # Diccionario para almacenar callbacks

    def register_callback(self, event: str, callback: Callable):
        """
        Registra una función callback para un evento específico.

        Args:
            event (str): El nombre del evento (e.g., "all_in", "fold", "chips_changed").
            callback (Callable): La función a llamar cuando ocurre el evento.
        """
        self._callbacks.setdefault(event, []).append(callback)

    def _trigger_callback(self, event: str, *args, **kwargs):
        """
        Llama a todas las funciones callback registradas para un evento.

        Args:
            event (str): El nombre del evento.
            *args: Argumentos posicionales para pasar a las funciones callback.
            **kwargs: Argumentos de palabra clave para pasar a las funciones callback.
        """
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error al llamar al callback para el evento {event}: {e}")

    def add_hole_card(self, card: Card):
        """
        Agrega una carta a la mano del jugador.

        Args:
            card (Card): La carta a agregar.

        Raises:
            PlayerInactiveError: Si el jugador no está activo en la mano.
            ValueError: Si el jugador ya tiene dos cartas en la mano.
            InvalidCardError: Si el parámetro no es una instancia de Card.

        Ejemplo:
            player.add_hole_card(Card("A", "S"))
        """
        if not self.is_active():
            raise PlayerInactiveError(f"{self.name} no está activo en la mano.")
        if not isinstance(card, Card):
            raise InvalidCardError("El parámetro debe ser una instancia de Card.")
        if len(self._hole_cards) >= 2:
            raise ValueError(f"{self.name} no puede tener más de dos cartas en la mano.")
        self._hole_cards.append(card)
        logger.debug(f"{self.name} recibió la carta {card}.")

    def add_chips(self, amount: int):
        """
        Agrega fichas al jugador.

        Args:
            amount (int): La cantidad de fichas a agregar.

        Raises:
            PlayerInactiveError: Si el jugador no está activo en la mano.
            ValueError: Si la cantidad a agregar debe ser positiva.
        """
        if not self.is_active():
            raise PlayerInactiveError(f"{self.name} no está activo en la mano.")
        if amount <= 0:
            raise ValueError("La cantidad a agregar debe ser positiva.")

        old_chips = self._chips
        self._chips += amount
        logger.info(f"{self.name} recibió {amount} fichas. Total: {self._chips}")
        self._trigger_callback("chips_changed", self, old_chips, self._chips)

    def pay(self, amount: int):
        """
        Paga una cantidad de fichas.

        Si el jugador no tiene suficientes fichas, paga todo lo que tiene y se va all-in.

        Args:
            amount (int): La cantidad de fichas a pagar.

        Raises:
            PlayerInactiveError: Si el jugador no está activo en la mano.
            ValueError: Si la cantidad a pagar debe ser no negativa.
        """
        if not self.is_active() and not self.is_all_in:
            raise PlayerInactiveError(f"{self.name} no está activo en la mano.")
        if amount < 0:
            raise ValueError("La cantidad a pagar debe ser no negativa.")
        if amount == 0:
            logger.warning(f"{self.name} intentó pagar 0 fichas.")
            return

        if amount > self._chips:
            logger.info(f"{self.name} going all-in with {self._chips} chips.")
            amount = self._chips
            self.is_all_in = True
            self._trigger_callback("all_in", self)

        old_chips = self._chips
        self._chips -= amount
        self._amount_invested += amount
        logger.debug(f"{self.name} pagó {amount} fichas. Restan: {self._chips}")
        self._trigger_callback("chips_changed", self, old_chips, self._chips)

        if self._chips == 0:
            self.is_all_in = True
            logger.info(f"{self.name} is now all-in.")
            self._trigger_callback("all_in", self)

    def receive(self, amount: int):
        """
        Recibe una cantidad de fichas.

        Args:
            amount (int): La cantidad de fichas a recibir.

        Raises:
            PlayerInactiveError: Si el jugador no está activo en la mano.
            ValueError: Si la cantidad a recibir debe ser positiva.
        """
        if not self.is_active():
            raise PlayerInactiveError(f"{self.name} no está activo en la mano.")
        if amount <= 0:
            raise ValueError("La cantidad a recibir debe ser positiva.")

        old_chips = self._chips
        self._chips += amount
        logger.info(f"{self.name} recibió {amount} fichas. Total: {self._chips}")
        self._trigger_callback("chips_changed", self, old_chips, self._chips)

    def clear_hole_cards(self):
        """
        Limpia las cartas de la mano del jugador.
        """
        self._hole_cards = []
        logger.debug(f"{self.name} limpió sus cartas de la mano.")

    def reset_for_round(self):
        """
        Reinicia el estado del jugador para una nueva ronda.
        """
        self.clear_hole_cards()
        self._amount_invested = 0
        self.update_state(in_hand=True, folded=False, is_all_in=False)

    def reset_chips(self, amount: int):
        """
        Reinicia la cantidad de fichas del jugador.

        Args:
            amount (int): La nueva cantidad de fichas del jugador.

        Raises:
            ValueError: Si la cantidad de fichas es negativa.
        """
        if amount < 0:
            raise ValueError("La cantidad de fichas debe ser no negativa.")

        old_chips = self._chips
        self._chips = amount
        logger.info(f"{self.name} reset chips to {amount}.")
        self._trigger_callback("chips_changed", self, old_chips, self._chips)

    def reset_player(self, initial_chips: int):
        """
        Reinicia completamente el estado del jugador.

        Args:
            initial_chips (int): La cantidad inicial de fichas del jugador.
        """
        if initial_chips < 0:
            raise ValueError("La cantidad inicial de fichas debe ser no negativa.")
        self.reset_for_round()
        self.reset_chips(initial_chips)
        logger.info(f"{self.name} reset player with {initial_chips} chips.")

    def fold(self):
        """
        Retira al jugador de la mano.
        """
        if not self.in_hand:
            logger.warning(f"{self.name} is already out of the hand.")
            return

        self.update_state(in_hand=False, folded=True)
        logger.info(f"{self.name} folds.")
        self._trigger_callback("fold", self)

    def show_hand(self, hidden: bool = False, show_one: bool = False) -> List[str]:
        """
        Devuelve una representación de la mano del jugador.

        Args:
            hidden (bool, optional): Indica si se deben ocultar las cartas. Por defecto es False.
            show_one (bool, optional): Indica si se debe mostrar solo una carta. Por defecto es False.

        Returns:
            List[str]: Una lista de strings que representan las cartas en la mano del jugador.
        """
        if hidden:
            return ["Hidden Card", "Hidden Card"]
        elif show_one:
            if self._hole_cards:
                return [str(self._hole_cards[0]), "Hidden Card"]
            else:
                return ["No Card", "Hidden Card"]
        else:
            return [str(card) for card in self._hole_cards]

    def is_active(self) -> bool:
        """
        Verifica si el jugador está activo en la mano.

        Un jugador está activo si está en la mano, no se ha retirado y no está all-in.

        Returns:
            bool: True si el jugador está activo, False de lo contrario.
        """
        return self.in_hand and not self.is_all_in and not self.folded

    def get_status(self, sensitive_info: bool = True) -> str:
        """
        Devuelve un resumen del estado del jugador.

        Args:
            sensitive_info (bool, optional): Indica si se debe incluir información sensible (e.g., hole cards). Por defecto es True.

        Returns:
            str: Un string que representa el estado del jugador.
        """
        status = f"{self.name}: Chips = {self.chips}, "
        if self.is_all_in:
            status += "All-in, "
        if self.folded:
            status += "Folded, "
        if not self.in_hand:
            status += "Out of hand, "
        status += f"Invested = {self.amount_invested}, "
        if sensitive_info:
            status += f"Hole Cards = {self.show_hand()}"
        else:
            status += "Hole Cards = [Hidden]"
        return status

    def set_position(self, position: Position):
        """
        Establece la posición del jugador en la mesa.

        Args:
            position (Position): La posición del jugador (e.g., Position.DEALER).
        """
        if not isinstance(position, Position):
            logger.warning(f"Posición inválida: {position}. Debe ser una instancia de Position Enum.")
            return
        self.position = position
        logger.info(f"{self.name} set position to {self.position}.")

    def can_raise(self, current_bet: int, min_raise: int) -> bool:
        """
        Verifica si el jugador puede subir la apuesta.

        Args:
            current_bet (int): La apuesta actual.
            min_raise (int): La subida mínima.

        Returns:
            bool: True si el jugador puede subir, False de lo contrario.
        """
        return self.is_active() and self.chips > current_bet + min_raise

    def can_call(self, amount: int) -> bool:
        """
        Verifica si el jugador puede igualar la apuesta.

        Args:
            amount (int): La cantidad a igualar.

        Returns:
            bool: True si el jugador puede igualar, False de lo contrario.
        """
        return self.is_active() and self.chips >= amount

    def update_state(self, in_hand: bool = None, folded: bool = None, is_all_in: bool = None):
        """
        Actualiza el estado del jugador de forma atómica.

        Args:
            in_hand (bool, optional): Indica si el jugador está en la mano.
            folded (bool, optional): Indica si el jugador se ha retirado de la mano.
            is_all_in (bool, optional): Indica si el jugador está all-in.
        """
        if in_hand is not None:
            self.in_hand = in_hand
        if folded is not None:
            self.folded = folded
        if is_all_in is not None:
            self.is_all_in = is_all_in
        logger.debug(f"{self.name} updated state: in_hand={self.in_hand}, folded={self.folded}, is_all_in={self.is_all_in}")

    @property
    def chips(self) -> int:
        """
        Devuelve la cantidad de fichas del jugador.

        Returns:
            int: La cantidad de fichas del jugador.
        """
        return self._chips

    @chips.setter
    def chips(self, amount: int):
        """
        Establece la cantidad de fichas del jugador.

        Args:
            amount (int): La nueva cantidad de fichas del jugador.

        Raises:
            ValueError: Si la cantidad de fichas es negativa.
        """
        if amount < 0:
            raise ValueError("La cantidad de fichas debe ser no negativa.")
        old_chips = self._chips
        self._chips = amount
        logger.debug(f"{self.name} chips set to {amount}.")
        self._trigger_callback("chips_changed", self, old_chips, self._chips)

    @property
    def hole_cards(self) -> List[Card]:
        """
        Devuelve las cartas en la mano del jugador.

        Returns:
            List[Card]: Las cartas en la mano del jugador.
        """
        return list(self._hole_cards)  # Devuelve una copia para evitar modificaciones externas

    @property
    def amount_invested(self) -> int:
        """
        Devuelve la cantidad de fichas que el jugador ha invertido en la ronda actual.

        Returns:
            int: La cantidad de fichas invertidas.
        """
        return self._amount_invested

    @property
    def is_folded(self) -> bool:
        """
        Devuelve si el jugador se ha retirado de la mano.

        Returns:
            bool: True si el jugador se ha retirado, False de lo contrario.
        """
        return self.folded

    def has_folded(self) -> bool:
        """
        Verifica si el jugador se ha retirado de la mano.

        Returns:
            bool: True si el jugador se ha retirado, False de lo contrario.
        """
        return self.folded

    def clone(self):
        """
        Crea una copia profunda del objeto Player.

        Returns:
            Player: Una copia profunda del objeto Player.
        """
        # Implementar clonación manual para mejorar performance si es relevante
        new_player = Player(self.name, self.chips, self.position)
        new_player.in_hand = self.in_hand
        new_player.is_all_in = self.is_all_in
        new_player.folded = self.folded
        new_player._amount_invested = self._amount_invested
        new_player._hole_cards = copy.copy(self._hole_cards)  # Copia superficial de la lista, pero las cartas son inmutables
        return new_player

    def __hash__(self):
        """
        Devuelve un hash del jugador basado en su nombre.

        Returns:
            int: El hash del jugador.
        """
        return hash(self.name)

    def __eq__(self, other):
        """
        Compara dos jugadores en base a atributos relevantes.

        Args:
            other (Player): El otro jugador a comparar.

        Returns:
            bool: True si los jugadores son iguales, False de lo contrario.
        """
        if isinstance(other, Player):
            return (self.name == other.name and
                    self.chips == other.chips and
                    self.in_hand == other.in_hand and
                    self.is_all_in == other.is_all_in and
                    self.folded == other.folded and
                    self.position == other.position)
        return False

    def get_sensitive_state(self, viewer="self"):
        """
        Devuelve un diccionario con el estado del jugador, controlando qué detalles se muestran.

        Args:
            viewer (str): El tipo de espectador ("self", "opponent", "spectator", "ai").

        Returns:
            dict: Un diccionario con el estado del jugador.
        """
        state = {
            "name": self.name,
            "chips": self.chips,
            "amount_invested": self.amount_invested,
            "in_hand": self.in_hand,
            "is_all_in": self.is_all_in,
            "folded": self.folded,
            "position": self.position.value if self.position else None
        }

        if viewer == "self" or viewer == "ai":
            state["hole_cards"] = [str(card) for card in self._hole_cards]
        else:
            state["hole_cards"] = ["Hidden" for _ in self._hole_cards]

        return state

    def __str__(self):
        """
        Devuelve una representación amigable del jugador para debug y logging.
        """
        return f"Player(name='{self.name}', chips={self.chips}, in_hand={self.in_hand}, is_all_in={self.is_all_in}, folded={self.folded}, position={self.position})"

    def __repr__(self):
        """
        Devuelve una representación del jugador para debug y logging.
        """
        return self.__str__()

    def to_dict(self, sensitive_info: bool = True) -> dict:
        """
        Devuelve un diccionario con el estado del jugador.

        Args:
            sensitive_info (bool, optional): Indica si se debe incluir información sensible (e.g., hole cards). Por defecto es True.
        """
        player_dict = {
            "name": self.name,
            "chips": self.chips,
            "amount_invested": self.amount_invested,
            "in_hand": self.in_hand,
            "is_all_in": self.is_all_in,
            "folded": self.folded,
            "position": self.position.value if self.position else None
        }
        if sensitive_info:
            player_dict["hole_cards"] = [str(card) for card in self._hole_cards]
        else:
            player_dict["hole_cards"] = ["Hidden" for _ in self._hole_cards]
        return player_dict

    @classmethod
    def from_dict(cls, player_dict: dict, card_reconstructor=None):
        """
        Crea un jugador a partir de un diccionario.

        Args:
            player_dict (dict): El diccionario con el estado del jugador.
            card_reconstructor: Una función que toma una representación de cadena de una carta y devuelve un objeto Card.

        Returns:
            Player: El jugador creado a partir del diccionario.
        """
        name = player_dict["name"]
        chips = player_dict["chips"]
        player = cls(name, chips)
        player._amount_invested = player_dict["amount_invested"]
        player.in_hand = player_dict["in_hand"]
        player.is_all_in = player_dict["is_all_in"]
        player.folded = player_dict["folded"]
        position_value = player_dict.get("position")
        player.position = Position(position_value) if position_value else None

        # Restaurar las cartas de la mano desde el diccionario si se proporciona un reconstructor
        if card_reconstructor and "hole_cards" in player_dict:
            try:
                player._hole_cards = [card_reconstructor(card_str) for card_str in player_dict["hole_cards"]]
            except Exception as e:
                logger.error(f"Error al reconstruir las cartas: {e}")
                player.clear_hole_cards()  # Limpiar las cartas si hay un error

        return player