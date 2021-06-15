from warstwa import Warstwa
import numpy as np

# inherit from base class Layer
class Nastepna_Warstwa(Warstwa):

    def __init__(self, wielkosc_wejscie, wielkosc_wyjscie):
        self.wagi = np.random.rand(wielkosc_wejscie, wielkosc_wyjscie)
        self.bias = np.random.rand(1, wielkosc_wyjscie)

    # zwraca wyjście dla danego wejścia
    def forward_propagation(self, dane_wejsciowe):
        self.wejscie = dane_wejsciowe
        self.wyjscie = np.dot(self.wejscie, self.wagi) + self.bias
        return self.wyjscie

    # oblicza de/dw, de/db dla danego błędu wyjścia de/dy. Zwraca błąd wejścia dE/dX.
    def backward_propagation(self, blad_wyjscia, wspolczynnik_uczenia):

        blad_wejscia = np.dot(blad_wyjscia, self.wagi.T)
        blad_wagi = np.dot(self.wejscie.T, blad_wyjscia)


        # Uaktualni parametry

        self.wagi -= wspolczynnik_uczenia * blad_wagi

        self.bias -= wspolczynnik_uczenia * blad_wyjscia
        return blad_wejscia


