from game.deck import Deck
from game.hand_evaluator import HandEvaluator
from game.player import Player  # Importar Player
from exceptions import game_exceptions  # Importar excepciones
import logging
from typing import Optional, Dict, Any

# Configuración del logger
logger = logging.getLogger(__name__)

ACTION_BET = "bet"
ACTION_CALL = "call"
ACTION_CHECK = "check"
ACTION_FOLD = "fold"
ACTION_ALL_IN = "all_in"
ACTION_RAISE = "raise" # Nueva acción para representar una subida

STAGE_PREFLOP = "preflop"
STAGE_FLOP = "flop"
STAGE_TURN = "turn"
STAGE_RIVER = "river"
STAGE_SHOWDOWN = "showdown"

class GameState:
    """
    Clase que representa el estado del juego de póker.
    """
    def __init__(self, player1: Player, player2: Player, small_blind: int, big_blind: int):
        """
        Inicializa el estado del juego.

        Args:
            player1 (Player): El primer jugador.
            player2: El segundo jugador.
            small_blind (int): La ciega pequeña.
            big_blind (int): La ciega grande.
        """
        self.player1: Player = player1
        self.player2: Player = player2
        self.small_blind: int = small_blind
        self.big_blind: int = big_blind
        self.deck: Deck = Deck()
        self.community_cards: list = []
        self.pot: int = 0
        self.current_bet: int = 0
        self.last_action: tuple = None  # Para rastrear la última acción realizada
        self.hand_evaluator: HandEvaluator = HandEvaluator()
        self.current_player: Player = player1  # El jugador1 empieza por defecto
        self.history: list = []  # Historial de acciones
        self.betting_round_actions: int = 0 # Contador de acciones en la ronda de apuestas
        self.min_raise: int = big_blind # Apuesta minima para subir
        self.last_raiser: Optional[Player] = None # Ultimo jugador que subio la apuesta
        self.blinds_posted: bool = False # Indica si las ciegas han sido pagadas
        self.current_stage: str = STAGE_PREFLOP # Etapa actual de la mano
        self.roles: Dict[Player, str] = {player1: "dealer", player2: "bb"} # Roles de los jugadores

    def start_new_round(self):
        """
        Prepara el estado del juego para una nueva ronda.
        """
        self.deck = Deck()
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.last_action = None
        self.player1.hole_cards = []
        self.player2.hole_cards = []
        self.player1.in_hand = True  # Ambos jugadores están en la mano al inicio
        self.player2.in_hand = True
        self.current_player = self.player1 # Restablecer al jugador 1 al inicio de la ronda
        self.history = [] # Limpiar el historial
        self.betting_round_actions = 0 # Restablecer el contador de acciones
        self.player1.amount_invested = 0 # Restablecer la inversion de los jugadores
        self.player2.amount_invested = 0
        self.min_raise = self.big_blind # Restablecer la apuesta minima
        self.last_raiser = None
        self.blinds_posted = False
        self.current_stage = STAGE_PREFLOP # Restablecer la etapa
        self.rotate_roles() # Rotar los roles

        # Forzar las ciegas
        self.force_blinds()

    def rotate_roles(self):
        """
        Rota los roles de los jugadores (dealer, small blind, big blind).
        """
        # Guarda el rol del dealer actual
        dealer = next(player for player, role in self.roles.items() if role == "dealer")

        # Rota los roles
        if dealer == self.player1:
            self.roles = {self.player2: "dealer", self.player1: "bb"}
        else:
            self.roles = {self.player1: "dealer", self.player2: "bb"}

    def force_blinds(self):
        """
        Forzar a los jugadores a pagar las ciegas.
        """
        # Determina quien es small blind y big blind
        sb = next(player for player, role in self.roles.items() if role == "dealer")
        bb = next(player for player, role in self.roles.items() if role == "bb")

        # Forzar el pago de las ciegas
        sb.pay(self.small_blind)
        bb.pay(self.big_blind)
        self.pot += self.small_blind + self.big_blind
        sb.amount_invested = self.small_blind # Registrar la inversion
        bb.amount_invested = self.big_blind
        self.betting_round_actions = 2 # Ambos jugadores realizaron una accion
        self.history.append((sb, ACTION_BET, self.small_blind, True)) # Registrar en el historial, True indica apuesta forzada
        self.history.append((bb, ACTION_BET, self.big_blind, True))
        self.last_raiser = bb # El jugador de la ciega grande es el ultimo en apostar
        self.blinds_posted = True

    def deal_hole_cards(self):
        """
        Reparte dos cartas a cada jugador.
        """
        self.player1.hole_cards = self.deck.deal(2)
        self.player2.hole_cards = self.deck.deal(2)

    def deal_flop(self):
        """
        Reparte tres cartas comunitarias (el flop).
        """
        self.burn_card()
        self.community_cards = self.deck.deal(3)
        self.current_stage = STAGE_FLOP # Actualizar la etapa
        self.reset_betting_round() # Resetea las variables de apuesta para la siguiente ronda

    def deal_turn(self):
        """
        Reparte una carta comunitaria (el turn).
        """
        self.burn_card()
        self.community_cards.extend(self.deck.deal(1))
        self.current_stage = STAGE_TURN # Actualizar la etapa
        self.reset_betting_round() # Resetea las variables de apuesta para la siguiente ronda

    def deal_river(self):
        """
        Reparte una carta comunitaria (el river).
        """
        self.burn_card()
        self.community_cards.extend(self.deck.deal(1))
        self.current_stage = STAGE_RIVER # Actualizar la etapa
        self.reset_betting_round() # Resetea las variables de apuesta para la siguiente ronda

    def burn_card(self):
        """
        Quema una carta del mazo.
        """
        self.deck.deal(1)

    def get_legal_actions(self, player: Player) -> list:
        """
        Devuelve una lista de acciones legales para el jugador actual.
        """
        actions = [ACTION_FOLD]  # Siempre se puede retirarse
        if player.chips > 0:
            if self.current_bet == 0:
                actions.append(ACTION_CHECK) # Solo se puede pasar si no hay apuesta
                actions.append(ACTION_BET)  # Siempre se puede apostar si tiene fichas
            else:
                actions.append(ACTION_CALL)  # Solo se puede igualar si hay una apuesta actual
                actions.append(ACTION_RAISE) # Se puede subir la apuesta
            actions.append(ACTION_ALL_IN)  # Siempre se puede ir all-in si tiene fichas
        return actions

    def can_bet(self, player: Player, amount: int) -> bool:
        """
        Verifica si el jugador puede realizar una apuesta de la cantidad especificada.
        """
        if amount is None:
            return False
        if amount > player.chips:
            return False

        if self.current_bet == 0:
            if amount < self.big_blind: # La apuesta minima es la ciega grande
                return False
        else:
            if amount < self.current_bet + self.min_raise: # Debe ser al menos la subida minima
                return False
        return True

    def can_check(self, player: Player) -> bool:
        """
        Verifica si el jugador puede pasar.
        """
        # No se puede pasar justo despues de una subida
        if self.last_raiser == player:
            return False
        return self.current_bet == 0

    def apply_action(self, player: Player, action: str, amount: int = None):
        """
        Aplica la acción al estado del juego.
        """
        if player != self.current_player:
            raise game_exceptions.InvalidAction("No es el turno del jugador.")

        if action == ACTION_BET or action == ACTION_RAISE:
            if not self.can_bet(player, amount):
                raise game_exceptions.InvalidAction("Apuesta inválida.")

            # Distinguir entre bet y raise
            is_raise = self.current_bet > 0
            if is_raise:
                action_taken = ACTION_RAISE
                self.min_raise = amount - self.current_bet # Actualizar la subida minima
            else:
                action_taken = ACTION_BET
                self.min_raise = amount # La primera apuesta es la subida minima

            player.pay(amount)
            self.pot += amount
            self.current_bet = amount
            self.last_action = (player, action_taken, amount)
            self.last_raiser = player # Actualizar el ultimo jugador que subio
        elif action == ACTION_CALL:
            amount_to_call = self.current_bet - player.amount_invested
            if amount_to_call < 0:
                raise game_exceptions.InvalidAction("Monto a igualar inválido.")
            if amount_to_call > player.chips:
                amount_to_call = player.chips  # Ir all-in si no tiene suficientes fichas
            player.pay(amount_to_call)
            self.pot += amount_to_call
            self.last_action = (player, action, amount_to_call)
        elif action == ACTION_CHECK:
            if not self.can_check(player):
                raise game_exceptions.InvalidAction("No se puede pasar cuando hay una apuesta actual o despues de una subida.")
            self.last_action = (player, action)
        elif action == ACTION_FOLD:
            player.in_hand = False
            self.last_action = (player, action)
        elif action == ACTION_ALL_IN:
            amount = player.chips
            player.pay(amount)
            self.pot += amount
            self.current_bet = max(self.current_bet, amount)  # La apuesta actual es el máximo entre la apuesta actual y el all-in
            self.last_action = (player, action, amount)
            self.last_raiser = player # All-in cuenta como ultima subida
        else:
            raise game_exceptions.InvalidAction(f"Acción desconocida: {action}")

        # Actualizar el historial y el estado del juego
        is_forced = False # Por defecto, la apuesta no es forzada
        self.history.append((player, action, amount, is_forced))
        self.betting_round_actions += 1

        # Actualizar la inversion del jugador (manejar el caso de check/fold)
        if amount is not None:
            player.amount_invested += amount

        self.next_player() # Cambiar al siguiente jugador

    def is_betting_round_over(self) -> bool:
        """
        Verifica si la ronda de apuestas ha terminado.
        """
        # Si solo queda un jugador en la mano
        if not (self.player1.in_hand and self.player2.in_hand):
            return True

        # Si ambos jugadores han invertido la misma cantidad
        if self.player1.amount_invested == self.player2.amount_invested:
            # Si alguien hizo all-in, la ronda termina
            if self.player1.chips == 0 or self.player2.chips == 0:
                return True
            # Si es el turno del apostador original, la ronda termina
            if self.current_player == self.get_first_better():
                return True

        return False

    def get_first_better(self) -> Optional[Player]:
        """
        Devuelve el jugador que apostó primero en la ronda de apuestas actual.
        """
        for player, action, amount, is_forced in self.history:
            if action == ACTION_BET and not is_forced: # Ignorar las apuestas forzadas
                return player
        if self.blinds_posted: # Si las ciegas fueron pagadas, el jugador de la ciega grande es el primero
            return self.player2
        return None

    def next_player(self):
        """
        Cambia al siguiente jugador.
        """
        if self.current_player == self.player1:
            self.current_player = self.player2
        else:
            self.current_player = self.player1

    def determine_winner(self) -> Player:
        """
        Determina el ganador de la ronda.
        """
        # Si un jugador se retiró, el otro gana
        if not self.player1.in_hand:
            return self.player2
        if not self.player2.in_hand:
            return self.player1

        # Evaluar las manos de los jugadores
        player1_hand = self.player1.hole_cards + self.community_cards
        player2_hand = self.player2.hole_cards + self.community_cards
        player1_rank = self.hand_evaluator.evaluate_hand(player1_hand)
        player2_rank = self.hand_evaluator.evaluate_hand(player2_hand)

        # Determinar el ganador basado en el ranking de las manos
        if player1_rank > player2_rank:
            return self.player1
        elif player2_rank > player1_rank:
            return self.player2
        else:
            # Si hay un empate, dividir el pozo
            return None

    def determine_final_winner(self) -> Player:
        """
        Determina el ganador final del juego.
        """
        if self.player1.chips > self.player2.chips:
            return self.player1
        elif self.player2.chips > self.player1.chips:
            return self.player2
        else:
            return None  # Empate

    def reset_round(self):
        """
        Reinicia el estado del juego para la siguiente ronda.
        """
        self.player1.reset_for_round()
        self.player2.reset_for_round()
        self.current_bet = 0
        self.pot = 0
        self.community_cards = []
        self.deck = Deck()  # Crear una nueva baraja
        self.last_action = None
        self.betting_round_actions = 0
        self.min_raise = self.big_blind
        self.last_raiser = None
        self.current_stage = STAGE_PREFLOP
        self.rotate_roles()

    def reset_betting_round(self):
        """
        Reinicia las variables de la ronda de apuestas para la siguiente calle.
        """
        self.current_bet = 0
        self.betting_round_actions = 0
        self.player1.amount_invested = 0
        self.player2.amount_invested = 0
        self.last_raiser = None

    def is_terminal(self) -> bool:
        """
        Verifica si el juego ha terminado.
        """
        return self.player1.chips <= 0 or self.player2.chips <= 0

    @property
    def min_bet(self) -> int:
        """
        Devuelve la apuesta mínima permitida.
        """
        if self.current_bet == 0:
            return self.big_blind
        else:
            return self.current_bet + self.min_raise

    def to_dict(self) -> Dict[str, Any]:
        """
        Devuelve un diccionario con el estado actual del juego.
        """
        return {
            "player1": self.player1.to_dict(),
            "player2": self.player2.to_dict(),
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "community_cards": [str(card) for card in self.community_cards],
            "pot": self.pot,
            "current_bet": self.current_bet,
            "last_action": self.last_action,
            "current_player": self.current_player.name,
            "history": [self._format_history_entry(entry) for entry in self.history],
            "deck_count": len(self.deck.cards),
            "min_raise": self.min_raise,
            "last_raiser": self.last_raiser.name if self.last_raiser else None,
            "current_stage": self.current_stage,
            "roles": {p.name: r for p, r in self.roles.items()}
        }

    def _format_history_entry(self, entry: tuple) -> str:
        """
        Formatea una entrada del historial para mejorar la legibilidad.
        """
        player, action, amount, is_forced = entry
        player_name = player.name if isinstance(player, Player) else player
        forced_str = " (Forced)" if is_forced else ""
        amount_str = f" ${amount}" if amount is not None else ""
        return f"{player_name} {action}{amount_str}{forced_str}"