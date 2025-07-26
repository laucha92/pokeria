def display_game_state(game_state, player_view=None):
    """
    Muestra el estado actual del juego.
    """
    print("\n--- Estado del Juego ---")
    print(f"Pozo: {game_state.pot}")
    print(f"Apuesta actual: {game_state.current_bet}")
    print(f"Cartas comunitarias: {game_state.community_cards}")

    if player_view:
        print(f"\n--- Vista de {player_view.name} ---")
        print(f"Tus cartas: {player_view.hole_cards}")
        print(f"Tus fichas: {player_view.chips}")

    print(f"\n--- Jugadores ---")
    print(f"{game_state.player1.name}: {game_state.player1.chips} fichas")
    print(f"{game_state.player2.name}: {game_state.player2.chips} fichas")

def display_recommendation(message):
    """
    Muestra una recomendación o mensaje al usuario.
    """
    print(f"\nRecomendación: {message}")