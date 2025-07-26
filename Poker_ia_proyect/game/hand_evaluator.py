from typing import List, Tuple, Dict
from game.card import Card, Rank, Suit
from enum import Enum
import itertools
import cProfile
import pstats
import unittest
from collections import Counter

class HandRanking(Enum):
    """
    Enumeración para representar los posibles rankings de una mano de póker.
    """
    HIGH_CARD = 1
    ONE_PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9
    ROYAL_FLUSH = 10

class HandEvaluator:
    """
    Clase para evaluar una mano de póker y determinar su ranking.
    """

    def __init__(self):
        """
        Inicializa el HandEvaluator.
        """
        pass

    def evaluate_hand(self, hand: List[Card]) -> Tuple[HandRanking, List[Card], List[Card]]:
        """
        Evalúa una mano de póker y devuelve su ranking, las cartas que contribuyen a ese ranking y los kickers.

        Args:
            hand (List[Card]): Una lista de objetos Card que representan la mano (5 o 7 cartas).

        Returns:
            Tuple[HandRanking, List[Card], List[Card]]: El ranking de la mano, las cartas que lo forman y los kickers.
            Kickers son las cartas que NO forman parte de la mano ganadora.

        Raises:
            ValueError: Si la mano contiene cartas no válidas, duplicadas o un número incorrecto de cartas.

        Ejemplo:
            hand = [Card(Rank.ACE, Suit.HEARTS), Card(Rank.ACE, Suit.DIAMONDS), Card(Rank.KING, Suit.SPADES), Card(Rank.QUEEN, Suit.CLUBS), Card(Rank.JACK, Suit.HEARTS)]
            ranking, cards, kickers = evaluator.evaluate_hand(hand)
            # ranking == HandRanking.ONE_PAIR
            # cards == [Card(Rank.ACE, Suit.HEARTS), Card(Rank.ACE, Suit.DIAMONDS)]
            # kickers == [Card(Rank.KING, Suit.SPADES), Card(Rank.QUEEN, Suit.CLUBS), Card(Rank.JACK, Suit.HEARTS)]
        """
        try:
            if not all(isinstance(card, Card) for card in hand):
                raise ValueError("La mano debe contener solo objetos Card.")
            if len(set(hand)) != len(hand):
                raise ValueError("La mano no puede contener cartas duplicadas.")
            if len(hand) not in (5, 7):
                raise ValueError("La mano debe contener 5 o 7 cartas.")

            # Para manos de 7 cartas, encuentra la mejor mano de 5 cartas
            if len(hand) == 7:
                best_hand, kickers = self.find_best_5_card_hand(hand)
                remaining_kickers = sorted([card for card in hand if card not in best_hand], key=lambda card: card.rank.value, reverse=True)
            else:
                best_hand = hand
                remaining_kickers = [] # No hay kickers en manos de 5 cartas, ya que todas forman la mano

            ranking, cards = self.determine_hand_ranking(best_hand)
            # Corregir kickers: solo las cartas FUERA de la mano ganadora
            kickers = [card for card in remaining_kickers if card not in cards]
            return ranking, cards, kickers

        except ValueError as e:
            print(f"Error al evaluar la mano: {e}")
            return HandRanking.HIGH_CARD, [], []  # O lanza la excepción

    def determine_hand_ranking(self, hand: List[Card]) -> Tuple[HandRanking, List[Card]]:
        """
        Determina el ranking de una mano de 5 cartas.

        Args:
            hand (List[Card]): Una lista de 5 objetos Card.

        Returns:
            Tuple[HandRanking, List[Card]]: El ranking de la mano y las cartas que lo forman.
        """
        # Precalcular conteo de rangos para optimización
        rank_counts = Counter([card.rank for card in hand])

        if self.is_royal_flush(hand):
            return HandRanking.ROYAL_FLUSH, hand
        if self.is_straight_flush(hand):
            return HandRanking.STRAIGHT_FLUSH, hand
        if self.is_four_of_a_kind(hand, rank_counts):
            return HandRanking.FOUR_OF_A_KIND, self.get_cards_of_a_kind(hand, 4)
        if self.is_full_house(hand, rank_counts):
            return HandRanking.FULL_HOUSE, self.get_full_house_cards(hand)
        if self.is_flush(hand):
            return HandRanking.FLUSH, hand
        if self.is_straight(hand):
            return HandRanking.STRAIGHT, hand
        if self.is_three_of_a_kind(hand, rank_counts):
            return HandRanking.THREE_OF_A_KIND, self.get_cards_of_a_kind(hand, 3)
        if self.is_two_pair(hand, rank_counts):
            return HandRanking.TWO_PAIR, self.get_two_pair_cards(hand)
        if self.is_one_pair(hand, rank_counts):
            return HandRanking.ONE_PAIR, self.get_cards_of_a_kind(hand, 2)
        return HandRanking.HIGH_CARD, sorted(hand, key=lambda card: card.rank.value, reverse=True)[:1]

    def find_best_5_card_hand(self, hand: List[Card]) -> Tuple[List[Card], List[Card]]:
        """
        Encuentra la mejor mano de 5 cartas posible dentro de una mano de 7 cartas.

        Args:
            hand (List[Card]): Una lista de 7 objetos Card.

        Returns:
            Tuple[List[Card], List[Card]]: La mejor mano de 5 cartas y los kickers.
        """
        best_hand = []
        best_ranking = HandRanking.HIGH_CARD
        kickers = []

        # No ordenar, iterar y guardar la mejor
        for combo in itertools.combinations(hand, 5):
            ranking, cards = self.determine_hand_ranking(list(combo))

            if ranking == HandRanking.ROYAL_FLUSH: # Optimizacion: early exit
                return list(combo), []

            if ranking == HandRanking.STRAIGHT_FLUSH: # Optimizacion: early exit
                if best_ranking.value < ranking.value:
                    best_hand = list(combo)
                    best_ranking = ranking
                    kickers = sorted([card for card in hand if card not in best_hand], key=lambda card: card.rank.value, reverse=True)
                    continue

            if ranking.value > best_ranking.value:
                best_ranking = ranking
                best_hand = list(combo)
                kickers = sorted([card for card in hand if card not in best_hand], key=lambda card: card.rank.value, reverse=True)
            elif ranking.value == best_ranking.value:
                # Si el ranking es el mismo, compara las manos para desempatar
                comparison_result = self.compare_hands(list(combo), best_hand, ranking)
                if comparison_result == 1:
                    best_hand = list(combo)
                    kickers = sorted([card for card in hand if card not in best_hand], key=lambda card: card.rank.value, reverse=True)

        return best_hand, kickers

    def compare_hands(self, hand1: List[Card], hand2: List[Card], ranking: HandRanking) -> int:
        """
        Compara dos manos con el mismo ranking y devuelve:
         1 si hand1 es mejor
        -1 si hand2 es mejor
         0 si son iguales
        """
        if ranking in (HandRanking.ROYAL_FLUSH, HandRanking.STRAIGHT_FLUSH, HandRanking.STRAIGHT, HandRanking.FLUSH):
            ranks1 = sorted([card.rank.value for card in hand1], reverse=True)
            ranks2 = sorted([card.rank.value for card in hand2], reverse=True)
            if ranks1 > ranks2:
                return 1
            elif ranks1 < ranks2:
                return -1
            else:
                return 0
        elif ranking in (HandRanking.FOUR_OF_A_KIND, HandRanking.THREE_OF_A_KIND, HandRanking.ONE_PAIR):
            # Comparar la carta del grupo, luego los kickers
            rank1 = self.get_cards_of_a_kind(hand1, 4 if ranking == HandRanking.FOUR_OF_A_KIND else (3 if ranking == HandRanking.THREE_OF_A_KIND else 2))[0].rank
            rank2 = self.get_cards_of_a_kind(hand2, 4 if ranking == HandRanking.FOUR_OF_A_KIND else (3 if ranking == HandRanking.THREE_OF_A_KIND else 2))[0].rank
            if rank1.value > rank2.value:
                return 1
            elif rank1.value < rank2.value:
                return -1
            else:
                # Comparar kickers (todos)
                kickers1 = sorted([card.rank.value for card in hand1 if card.rank != rank1], reverse=True)
                kickers2 = sorted([card.rank.value for card in hand2 if card.rank != rank2], reverse=True)
                if kickers1 > kickers2:
                    return 1
                elif kickers1 < kickers2:
                    return -1
                else:
                    return 0
        elif ranking == HandRanking.FULL_HOUSE:
            # Comparar trio primero, luego pareja
            trio_rank1 = Counter([card.rank for card in hand1]).most_common(1)[0][0]
            trio_rank2 = Counter([card.rank for card in hand2]).most_common(1)[0][0]
            if trio_rank1.value > trio_rank2.value:
                return 1
            elif trio_rank1.value < trio_rank2.value:
                return -1
            else:
                pair_rank1 = [card.rank for card in hand1 if card.rank != trio_rank1][0]
                pair_rank2 = [card.rank for card in hand2 if card.rank != trio_rank2][0]
                if pair_rank1.value > pair_rank2.value:
                    return 1
                elif pair_rank1.value < pair_rank2.value:
                    return -1
                else:
                    return 0
        elif ranking == HandRanking.TWO_PAIR:
            # Comparar pares en orden, luego kicker
            pairs1 = sorted([rank.value for rank, count in Counter([card.rank for card in hand1]).items() if count == 2], reverse=True)
            pairs2 = sorted([rank.value for rank, count in Counter([card.rank for card in hand2]).items() if count == 2], reverse=True)
            if pairs1 > pairs2:
                return 1
            elif pairs1 < pairs2:
                return -1
            else:
                kicker1 = [card.rank.value for card in hand1 if card.rank.value not in pairs1][0]
                kicker2 = [card.rank.value for card in hand2 if card.rank.value not in pairs2][0]
                if kicker1 > kicker2:
                    return 1
                elif kicker1 < kicker2:
                    return -1
                else:
                    return 0
        elif ranking == HandRanking.HIGH_CARD:
            # Comparar kickers uno a uno
            ranks1 = sorted([card.rank.value for card in hand1], reverse=True)
            ranks2 = sorted([card.rank.value for card in hand2], reverse=True)
            for i in range(len(ranks1)):
                if ranks1[i] > ranks2[i]:
                    return 1
                elif ranks1[i] < ranks2[i]:
                    return -1
            return 0
        return 0

    def is_royal_flush(self, hand: List[Card]) -> bool:
        """
        Verifica si la mano es una escalera real.
        """
        if not self.is_straight_flush(hand):
            return False
        ranks = [card.rank for card in hand]
        return all(rank in [Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE] for rank in ranks)

    def is_straight_flush(self, hand: List[Card]) -> bool:
        """
        Verifica si la mano es una escalera de color.
        """
        return self.is_flush(hand) and self.is_straight(hand)

    def is_four_of_a_kind(self, hand: List[Card], rank_counts: Counter = None) -> bool:
        """
        Verifica si la mano tiene cuatro cartas del mismo rango.
        """
        if rank_counts is None:
            rank_counts = Counter([card.rank for card in hand])
        return 4 in rank_counts.values()

    def is_full_house(self, hand: List[Card], rank_counts: Counter = None) -> bool:
        """
        Verifica si la mano es un full house (tres cartas del mismo rango y un par).
        """
        if rank_counts is None:
            rank_counts = Counter([card.rank for card in hand])
        return 3 in rank_counts.values() and 2 in rank_counts.values()

    def is_flush(self, hand: List[Card]) -> bool:
        """
        Verifica si la mano es un flush (todas las cartas del mismo palo).
        """
        suits = [card.suit for card in hand]
        return len(set(suits)) == 1

    def is_straight(self, hand: List[Card]) -> bool:
        """
        Verifica si la mano es una escalera (cinco cartas en secuencia).
        """
        ranks = sorted([card.rank.value for card in hand])
        # Verifica si es una escalera normal
        if ranks[-1] - ranks[0] == 4 and len(set(ranks)) == 5:
            return True
        # Verifica si es una escalera de As a 5 (A, 2, 3, 4, 5)
        if ranks == [2, 3, 4, 5, 14]:
            return True

        #Verifica si es una escalera de As a 5 (A, 2, 3, 4, 5) en manos de 7 cartas
        if len(hand) == 7:
            ranks = sorted([card.rank.value for card in hand])
            if ranks == [2, 3, 4, 5, 14]:
                return True
        return False

    def is_three_of_a_kind(self, hand: List[Card], rank_counts: Counter = None) -> bool:
        """
        Verifica si la mano tiene tres cartas del mismo rango.
        """
        if rank_counts is None:
            rank_counts = Counter([card.rank for card in hand])
        return 3 in rank_counts.values()

    def is_two_pair(self, hand: List[Card], rank_counts: Counter = None) -> bool:
        """
        Verifica si la mano tiene dos pares diferentes.
        """
        if rank_counts is None:
            rank_counts = Counter([card.rank for card in hand])
        return list(rank_counts.values()).count(2) == 2

    def is_one_pair(self, hand: List[Card], rank_counts: Counter = None) -> bool:
        """
        Verifica si la mano tiene un par.
        """
        if rank_counts is None:
            rank_counts = Counter([card.rank for card in hand])
        return 2 in rank_counts.values()

    def has_of_a_kind(self, hand: List[Card], count: int) -> bool:
        """
        Verifica si la mano tiene 'count' cartas del mismo rango.
        """
        ranks = [card.rank for card in hand]
        rank_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        return count in rank_counts.values()

    def get_cards_of_a_kind(self, hand: List[Card], count: int) -> List[Card]:
         """
         Obtiene las cartas que forman un "of a kind" (par, trío, póker).
         """
         ranks = [card.rank for card in hand]
         rank_counts = {}
         for rank in ranks:
             rank_counts[rank] = rank_counts.get(rank, 0) + 1

         for rank, c in rank_counts.items():
             if c == count:
                 return [card for card in hand if card.rank == rank]
         return []

    def get_full_house_cards(self, hand: List[Card]) -> List[Card]:
        """
        Obtiene las cartas que forman el full house.
        """
        # Evitar variables temporales innecesarias
        counts = Counter([card.rank for card in hand])
        trio_rank = counts.most_common(1)[0][0]
        pair_rank = [rank for rank in counts if counts[rank] == 2][0]
        return [card for card in hand if card.rank in (trio_rank, pair_rank)]

    def get_two_pair_cards(self, hand: List[Card]) -> List[Card]:
        """
        Obtiene las cartas que forman los dos pares.
        """
        ranks = [card.rank for card in hand]
        rank_counts = {}
        pairs = []
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

        for rank, count in rank_counts.items():
            if count == 2:
                pairs.extend([card for card in hand if card.rank == rank])
        return pairs

# --- Pruebas Unitarias ---
class TestHandEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = HandEvaluator()

    def test_royal_flush(self):
        hand = [
            Card(Rank.TEN, Suit.HEARTS),
            Card(Rank.JACK, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.HEARTS),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.ACE, Suit.HEARTS),
        ]
        ranking, _, _ = self.evaluator.evaluate_hand(hand)
        self.assertEqual(ranking, HandRanking.ROYAL_FLUSH)

    def test_straight_flush(self):
        hand = [
            Card(Rank.NINE, Suit.HEARTS),
            Card(Rank.TEN, Suit.HEARTS),
            Card(Rank.JACK, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.HEARTS),
            Card(Rank.KING, Suit.HEARTS),
        ]
        ranking, _, _ = self.evaluator.evaluate_hand(hand)
        self.assertEqual(ranking, HandRanking.STRAIGHT_FLUSH)

    def test_one_pair(self):
        hand = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.ACE, Suit.DIAMONDS),
            Card(Rank.KING, Suit.SPADES),
            Card(Rank.QUEEN, Suit.CLUBS),
            Card(Rank.JACK, Suit.HEARTS),
        ]
        ranking, cards, kickers = self.evaluator.evaluate_hand(hand)
        self.assertEqual(ranking, HandRanking.ONE_PAIR)
        self.assertEqual(len(cards), 2)
        self.assertEqual(len(kickers), 3)  # Ahora devuelve solo los kickers
        self.assertEqual(kickers[0].rank, Rank.KING)
        self.assertEqual(kickers[1].rank, Rank.QUEEN)

    def test_two_pair(self):
        hand = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.ACE, Suit.DIAMONDS),
            Card(Rank.KING, Suit.SPADES),
            Card(Rank.KING, Suit.CLUBS),
            Card(Rank.JACK, Suit.HEARTS),
        ]
        ranking, cards, kickers = self.evaluator.evaluate_hand(hand)
        self.assertEqual(ranking, HandRanking.TWO_PAIR)
        self.assertEqual(len(cards), 4)
        self.assertEqual(len(kickers), 1)  # Ahora devuelve solo los kickers
        self.assertEqual(kickers[0].rank, Rank.JACK)

    def test_three_of_a_kind(self):
        hand = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.ACE, Suit.DIAMONDS),
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.KING, Suit.CLUBS),
            Card(Rank.JACK, Suit.HEARTS),
        ]
        ranking, cards, kickers = self.evaluator.evaluate_hand(hand)
        self.assertEqual(ranking, HandRanking.THREE_OF_A_KIND)
        self.assertEqual(len(cards), 3)
        self.assertEqual(len(kickers), 2)  # Ahora devuelve solo los kickers
        self.assertEqual(kickers[0].rank, Rank.KING)
        self.assertEqual(kickers[1].rank, Rank.JACK)

    def test_straight(self):
        hand = [
            Card(Rank.TWO, Suit.CLUBS),
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.HEARTS),
            Card(Rank.FIVE, Suit.SPADES),
            Card(Rank.SIX, Suit.CLUBS),
        ]
        ranking, cards, kickers = self.evaluator.evaluate_hand(hand)
        self.assertEqual(ranking, HandRanking.STRAIGHT)
        self.assertEqual(len(cards), 5)
        self.assertEqual(len(kickers), 0)  # Ahora devuelve solo los kickers

    def test_flush(self):
        hand = [
            Card(Rank.TWO, Suit.HEARTS),
            Card(Rank.FOUR, Suit.HEARTS),
            Card(Rank.SIX, Suit.HEARTS),
            Card(Rank.EIGHT, Suit.HEARTS),
            Card(Rank.TEN, Suit.HEARTS),
        ]
        ranking, cards, kickers = self.evaluator.evaluate_hand(hand)
        self.assertEqual(ranking, HandRanking.FLUSH)
        self.assertEqual(len(cards), 5)
        self.assertEqual(len(kickers), 0)  # Ahora devuelve solo los kickers

    def test_full_house(self):
        hand = [
            Card(Rank.TWO, Suit.HEARTS),
            Card(Rank.TWO, Suit.DIAMONDS),
            Card(Rank.TWO, Suit.SPADES),
            Card(Rank.EIGHT, Suit.HEARTS),
            Card(Rank.EIGHT, Suit.DIAMONDS),
        ]
        ranking, cards, kickers = self.evaluator.evaluate_hand(hand)
        self.assertEqual(ranking, HandRanking.FULL_HOUSE)
        self.assertEqual(len(cards), 5)
        self.assertEqual(len(kickers), 0)  # Ahora devuelve solo los kickers

    def test_four_of_a_kind(self):
        hand = [
            Card(Rank.TWO, Suit.HEARTS),
            Card(Rank.TWO, Suit.DIAMONDS),
            Card(Rank.TWO, Suit.SPADES),
            Card(Rank.TWO, Suit.CLUBS),
            Card(Rank.EIGHT, Suit.DIAMONDS),
        ]
        ranking, cards, kickers = self.evaluator.evaluate_hand(hand)
        self.assertEqual(ranking, HandRanking.FOUR_OF_A_KIND)
        self.assertEqual(len(cards), 4)
        self.assertEqual(len(kickers), 1)  # Ahora devuelve solo los kickers
        self.assertEqual(kickers[0].rank, Rank.EIGHT)

    def test_straight_flush_seven_cards(self):
        hand = [
            Card(Rank.NINE, Suit.HEARTS),
            Card(Rank.TEN, Suit.HEARTS),
            Card(Rank.JACK, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.HEARTS),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.TWO, Suit.DIAMONDS),
            Card(Rank.THREE, Suit.CLUBS),
        ]
        ranking, cards, kickers = self.evaluator.evaluate_hand(hand)
        self.assertEqual(ranking, HandRanking.STRAIGHT_FLUSH)
        self.assertEqual(len(cards), 5)
        self.assertEqual(len(kickers), 0)  # Ahora devuelve solo los kickers

    def test_high_card(self):
        hand = [
            Card(Rank.TWO, Suit.HEARTS),
            Card(Rank.FOUR, Suit.DIAMONDS),
            Card(Rank.SIX, Suit.SPADES),
            Card(Rank.EIGHT, Suit.CLUBS),
            Card(Rank.TEN, Suit.DIAMONDS),
        ]
        ranking, cards, kickers = self.evaluator.evaluate_hand(hand)
        self.assertEqual(ranking, HandRanking.HIGH_CARD)
        self.assertEqual(len(cards), 1)
        self.assertEqual(len(kickers), 4)  # Ahora devuelve solo los kickers
        self.assertEqual(kickers[0].rank, Rank.TEN)
        self.assertEqual(kickers[1].rank, Rank.EIGHT)

    def test_straight_low_seven_cards(self):
        hand = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.TWO, Suit.DIAMONDS),
            Card(Rank.THREE, Suit.SPADES),
            Card(Rank.FOUR, Suit.CLUBS),
            Card(Rank.FIVE, Suit.DIAMONDS),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.DIAMONDS),
        ]
        ranking, cards, kickers = self.evaluator.evaluate_hand(hand)
        self.assertEqual(ranking, HandRanking.STRAIGHT)
        self.assertEqual(len(cards), 5)
        self.assertEqual(len(kickers), 0)

    def test_straight_low_wheel(self):
        hand = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.TWO, Suit.DIAMONDS),
            Card(Rank.THREE, Suit.SPADES),
            Card(Rank.FOUR, Suit.CLUBS),
            Card(Rank.FIVE, Suit.DIAMONDS),
        ]
        ranking, cards, kickers = self.evaluator.evaluate_hand(hand)
        self.assertEqual(ranking, HandRanking.STRAIGHT)
        self.assertEqual(len(cards), 5)
        self.assertEqual(len(kickers), 0)

# --- Profiling ---
def run_evaluation(evaluator, hands):
    for hand in hands:
        evaluator.evaluate_hand(hand)

# Generar algunas manos de prueba
hands = [
    [Card(Rank.TEN, Suit.HEARTS), Card(Rank.JACK, Suit.HEARTS), Card(Rank.QUEEN, Suit.HEARTS), Card(Rank.KING, Suit.HEARTS), Card(Rank.ACE, Suit.HEARTS)],
    [Card(Rank.TWO, Suit.CLUBS), Card(Rank.THREE, Suit.DIAMONDS), Card(Rank.FOUR, Suit.HEARTS), Card(Rank.FIVE, Suit.SPADES), Card(Rank.SIX, Suit.CLUBS)],
    [Card(Rank.TWO, Suit.HEARTS), Card(Rank.FOUR, Suit.DIAMONDS), Card(Rank.SIX, Suit.SPADES), Card(Rank.EIGHT, Suit.CLUBS), Card(Rank.TEN, Suit.DIAMONDS)],
    [Card(Rank.TWO, Suit.HEARTS), Card(Rank.TWO, Suit.DIAMONDS), Card(Rank.KING, Suit.SPADES), Card(Rank.KING, Suit.CLUBS), Card(Rank.JACK, Suit.HEARTS)],
    [Card(Rank.TWO, Suit.HEARTS), Card(Rank.TWO, Suit.DIAMONDS), Card(Rank.TWO, Suit.SPADES), Card(Rank.KING, Suit.CLUBS), Card(Rank.JACK, Suit.HEARTS)],
    [Card(Rank.TWO, Suit.HEARTS), Card(Rank.TWO, Suit.DIAMONDS), Card(Rank.TWO, Suit.SPADES), Card(Rank.TWO, Suit.CLUBS), Card(Rank.JACK, Suit.HEARTS)],
    [Card(Rank.TWO, Suit.HEARTS), Card(Rank.TWO, Suit.DIAMONDS), Card(Rank.TWO, Suit.SPADES), Card(Rank.EIGHT, Suit.HEARTS), Card(Rank.EIGHT, Suit.DIAMONDS)],
    [Card(Rank.NINE, Suit.HEARTS), Card(Rank.TEN, Suit.HEARTS), Card(Rank.JACK, Suit.HEARTS), Card(Rank.QUEEN, Suit.HEARTS), Card(Rank.KING, Suit.HEARTS)],
    [Card(Rank.ACE, Suit.HEARTS), Card(Rank.ACE, Suit.DIAMONDS), Card(Rank.KING, Suit.SPADES), Card(Rank.QUEEN, Suit.CLUBS), Card(Rank.JACK, Suit.HEARTS)],
    [Card(Rank.NINE, Suit.HEARTS), Card(Rank.TEN, Suit.HEARTS), Card(Rank.JACK, Suit.HEARTS), Card(Rank.QUEEN, Suit.HEARTS), Card(Rank.KING, Suit.HEARTS), Card(Rank.TWO, Suit.DIAMONDS), Card(Rank.THREE, Suit.CLUBS)],
    [Card(Rank.ACE, Suit.HEARTS), Card(Rank.TWO, Suit.DIAMONDS), Card(Rank.THREE, Suit.SPADES), Card(Rank.FOUR, Suit.CLUBS), Card(Rank.FIVE, Suit.DIAMONDS), Card(Rank.KING, Suit.HEARTS), Card(Rank.QUEEN, Suit.DIAMONDS)],
    [Card(Rank.ACE, Suit.HEARTS), Card(Rank.TWO, Suit.DIAMONDS), Card(Rank.THREE, Suit.SPADES), Card(Rank.FOUR, Suit.CLUBS), Card(Rank.FIVE, Suit.DIAMONDS)],
]

evaluator = HandEvaluator()

# Ejecutar el profiling
cProfile.run('run_evaluation(evaluator, hands)', 'profile_output')

# Analizar los resultados
p = pstats.Stats('profile_output')
p.sort_stats('cumulative').print_stats(10) # Muestra las 10 funciones más costosas

# --- Ejecutar Pruebas Unitarias ---
if __name__ == '__main__':
    unittest.main()