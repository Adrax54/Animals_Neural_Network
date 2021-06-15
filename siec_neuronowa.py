class Siec_neuronowa:

    def __init__(self):
        self.wartswy = []
        self.blad = None
        self.blad_prim = None

    # Dodaj warstwę do sieci
    def dodaj_warstwe(self, warstwa):
        self.wartswy.append(warstwa)

    # wybór funkcji błędu
    def jaki_blad(self, blad, blad_prim):
        self.blad = blad
        self.blad_prim = blad_prim

    # Testowanie sieci, klasyfikacja wejścia do danego wyjścia
    def test(self, dane_wejsciowe):
        # Wymiar
        dane = len(dane_wejsciowe)
        wynik = []

        # Sieć odpowiada na wszyskie dane
        for i in range(dane):
            # forward propagation
            wyjscie = dane_wejsciowe[i]
            for warstwa in self.wartswy:
                wyjscie = warstwa.forward_propagation(wyjscie)
            wynik.append(wyjscie)

        return wynik

    # Uczenie sieci
    def proces_uczenia(self, x_train, y_train, epoki, wspolczynnik_uczenia):
        f = open("wyniki.txt", "w")
        # Wymiar
        dane = len(x_train)

        for i in range(epoki):
            err = 0

            for j in range(dane):
                # forward propagation
                wyjscie = x_train[j]
                for warstwa in self.wartswy:
                    wyjscie = warstwa.forward_propagation(wyjscie)

                # Obliczanie funkcji błędu
                err += self.blad(y_train[j], wyjscie)

                # backward propagation
                error = self.blad_prim(y_train[j], wyjscie)
                for warstwa in reversed(self.wartswy):
                    error = warstwa.backward_propagation(error, wspolczynnik_uczenia)

            # Obliczanie średniego błędu dla wszyskich danych
            err /= dane
            print('epoka %d/%d   błąd=%f' % (i + 1, epoki, err))
            f.write(('epoka %d/%d   błąd=%f \n' % (i + 1, epoki, err)))



