def get_user_action(legal_actions):
    """
    Obtiene la acción del usuario desde la entrada estándar.
    """
    print("\nAcciones legales:")
    for i, action in enumerate(legal_actions):
        print(f"{i + 1}. {action}")

    while True:
        try:
            choice = int(input("Elige una acción (ingresa el número): "))
            if 1 <= choice <= len(legal_actions):
                return legal_actions[choice - 1]
            else:
                print("Número de acción inválido. Intenta de nuevo.")
        except ValueError:
            print("Entrada inválida. Ingresa un número.")

def get_bet_amount(min_bet, max_bet):
    """
    Obtiene el monto de la apuesta del usuario desde la entrada estándar.
    """
    while True:
        try:
            amount = int(input(f"Ingresa el monto de la apuesta (mínimo {min_bet}, máximo {max_bet}): "))
            if min_bet <= amount <= max_bet:
                return amount
            else:
                print(f"Monto inválido. Ingresa un monto entre {min_bet} y {max_bet}.")
        except ValueError:
            print("Entrada inválida. Ingresa un número.")