


class Warstwa:
    def __init__(self):
        self.wejście = None
        self.wyjście = None

    # Oblicza wyjście Y warstwy dla wejścia x X
    def forward_propagation(self, wejscie):
        raise NotImplementedError

    # oblicza dE/dX dla danego  dE/dY (i aktualize parametry)
    def backward_propagation(self, blad_wyjscia, wspolczynnik_uczenia):
        raise NotImplementedError