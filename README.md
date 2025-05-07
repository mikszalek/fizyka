# fizyka
#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
import os
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# opcjonalnie kolorowanie terminala
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError:
    class Fore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = ''
    class Style:
        BRIGHT = RESET_ALL = ''

# --- stałe fizyczne ---
kB: float = 1.380649e-23       # J/K
NA: float = 6.02214076e23      # 1/mol
MOLAR_MASSES = {
    'N2': 28.0134e-3,          # kg/mol
    'O2': 31.998e-3            # kg/mol
}

# --- klasa obsługująca rozkład Maxwell–Boltzmanna ---
class MaxwellBoltzmann:
    def __init__(self, gas: Literal['N2','O2'], n_mol: float = 1.0):
        if gas not in MOLAR_MASSES:
            raise ValueError(f"Nieobsługiwany gaz: {gas}")
        self.gas = gas
        self.n_mol = n_mol
        self.m0 = MOLAR_MASSES[gas] / NA
        self.N = NA * n_mol

    def N_v(self, v: np.ndarray, T: float) -> np.ndarray:
        coef = 4*np.pi * self.N * (self.m0/(2*np.pi*kB*T))**1.5
        return coef * v**2 * np.exp(-self.m0*v**2/(2*kB*T))

    def most_probable_speed(self, T: float) -> float:
        return np.sqrt(2*kB*T/self.m0)

    def probability_interval(self, T: float, vmin: float, vmax: float) -> float:
        integral, _ = quad(lambda v: self.N_v(np.array([v]), T)[0], vmin, vmax)
        return integral / self.N

# --- funkcje wejścia z walidacją ---
def input_float(prompt: str, min_value: float = None, max_value: float = None) -> float:
    while True:
        s = input(Fore.CYAN + prompt + Style.RESET_ALL).strip().replace(',','.')
        try:
            x = float(s)
        except ValueError:
            print(Fore.RED + "  ➤ To nie jest liczba. Spróbuj ponownie." + Style.RESET_ALL)
            continue
        if min_value is not None and x < min_value:
            print(Fore.RED + f"  ➤ Wartość musi być ≥ {min_value}." + Style.RESET_ALL)
            continue
        if max_value is not None and x > max_value:
            print(Fore.RED + f"  ➤ Wartość musi być ≤ {max_value}." + Style.RESET_ALL)
            continue
        return x

def input_gas() -> Literal['N2','O2']:
    while True:
        s = input(Fore.CYAN + "Wybierz gaz (N2 lub O2): " + Style.RESET_ALL).strip().upper()
        if s in MOLAR_MASSES:
            return s  # type: ignore
        print(Fore.RED + "  ➤ Nieobsługiwany gaz. Podaj 'N2' lub 'O2'." + Style.RESET_ALL)

def clear_screen():
    os.system('cls' if os.name=='nt' else 'clear')

# --- menu ---
def main_menu():
    clear_screen()
    # Pierwszy nagłówek
    print(Fore.MAGENTA + Style.BRIGHT + """
╔════════════════════════════════════════════╗
║       🔬  SYMULATOR GAZU DOSKONAŁEGO       ║
║     Maxwell-Boltzmann Speed Explorer 🧪    ║
╚════════════════════════════════════════════╝
""" + Style.RESET_ALL)
    print(Fore.YELLOW + "1)" + Style.RESET_ALL, "Rysuj rozkład prędkości (1 mol)")
    print(Fore.YELLOW + "2)" + Style.RESET_ALL, "Prawdopodobieństwo v ∈ [v_min, v_max] (n moli N2)")
    print(Fore.YELLOW + "3)" + Style.RESET_ALL, "Prędkość najbardziej prawdopodobna (n moli)")
    print(Fore.YELLOW + "4)" + Style.RESET_ALL, "Koniec programu")
    print()

# --- główna pętla ---
def main():
    while True:
        main_menu()
        choice = input(Fore.CYAN + "Twój wybór [1-4]: " + Style.RESET_ALL).strip()
        if choice == '1':
            T = input_float("Temperatura T [K]: ", min_value=1e-6)
            gas = input_gas()
            mb = MaxwellBoltzmann(gas, n_mol=1.0)
            v_mp = mb.most_probable_speed(T)
            v = np.linspace(0, 3*v_mp, 500)
            y = mb.N_v(v, T)

            plt.figure()
            plt.plot(v, y, label="N_v(v)")
            plt.axvline(v_mp, linestyle='--', label=f"v_mp={v_mp:.1f} m/s")
            plt.fill_between(v, y, where=(v<=v_mp), alpha=0.3)
            plt.title(f"Maxwell–Boltzmann dla {gas}, T={T} K")
            plt.xlabel("v (m/s)")
            plt.ylabel("N_v")
            plt.legend()
            plt.grid(True)
            plt.show()

        elif choice == '2':
            T    = input_float("Temperatura T [K]: ", min_value=1e-6)
            n    = input_float("Liczba moli n: ", min_value=1e-6)
            vmin = input_float("v_min [m/s]: ", min_value=0.0)
            vmax = input_float("v_max [m/s]: ", min_value=0.0)
            if vmax <= vmin:
                print(Fore.RED + "  ➤ Błąd: v_max musi być > v_min." + Style.RESET_ALL)
                input("Naciśnij Enter, by wrócić do menu…")
                continue

            mb = MaxwellBoltzmann('N2', n)
            p = mb.probability_interval(T, vmin, vmax)

            # wykres z zaznaczonym przedziałem
            v = np.linspace(0, max(3*mb.most_probable_speed(T), vmax*1.1), 500)
            y = mb.N_v(v, T)
            plt.figure()
            plt.plot(v, y)
            plt.fill_between(v, y, where=( (v>=vmin)&(v<=vmax) ), alpha=0.3)
            plt.title(f"P(v∈[{vmin:.1f},{vmax:.1f}]) = {p:.5f}")
            plt.xlabel("v (m/s)")
            plt.ylabel("N_v")
            plt.grid(True)
            plt.show()

            print(Fore.GREEN + f"\nPrawdopodobieństwo: {p:.6f}" + Style.RESET_ALL)
            input("Naciśnij Enter, by kontynuować…")

        elif choice == '3':
            T   = input_float("Temperatura T [K]: ", min_value=1e-6)
            gas = input_gas()
            n   = input_float("Liczba moli n: ", min_value=1e-6)
            mb = MaxwellBoltzmann(gas, n)
            v_mp = mb.most_probable_speed(T)
            print(Fore.GREEN + f"\nPrędkość najbardziej prawdopodobna: {v_mp:.2f} m/s" + Style.RESET_ALL)
            input("Naciśnij Enter, by wrócić do menu…")

        elif choice == '4':
            print(Fore.MAGENTA + "👋 Do zobaczenia!" + Style.RESET_ALL)
            sys.exit(0)

        else:
            print(Fore.RED + "  ➤ Niepoprawny wybór. Wpisz 1–4." + Style.RESET_ALL)
            input("Naciśnij Enter, by spróbować ponownie…")

if __name__ == "__main__":
    main()

