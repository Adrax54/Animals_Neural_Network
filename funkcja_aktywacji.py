from warstwa import Warstwa
import numpy as np

class Aktywacja(Warstwa):
    def __init__(self, funkcja_aktywacji, funkcja_aktywacji_prim):
        self.funkcja_aktywacji = funkcja_aktywacji
        self.funkcja_aktywacji_prim = funkcja_aktywacji_prim

    # Zwraca aktywację warstwy
    def forward_propagation(self, dane_wejsciowe):
        self.wejscie = dane_wejsciowe
        self.wyjscie = self.funkcja_aktywacji(self.wejscie)
        return self.wyjscie

    # Zwraca błąd wejścia de/dx dla danego błędu wyjścia de/dy
    def backward_propagation(self, blad_wyjscia, wspolczynnik_uczenia):
        return self.funkcja_aktywacji_prim(self.wejscie) * blad_wyjscia




#Funkcja aktywacji tanh
def tanh(x):
    return np.tanh(x)
#Pochodna funkcji aktywacji tanh
def tanh_prim(x):
    return 1-np.tanh(x)**2
#funkcja aktywacji sigmoidalna
def sigmoidalna(x):
    return 1 / (1 + np.exp(-x))
#pochodna funkcji aktywacji sigmoidalnej
def sigmoidalna_prim(x):
    return (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))