import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

from siec_neuronowa import Siec_neuronowa
from tworzenie_warstw import Nastepna_Warstwa
from funkcja_aktywacji import Aktywacja
from funkcja_aktywacji import tanh, tanh_prim, sigmoidalna_prim, sigmoidalna
from blad import mse, mse_prim


# dane uczące
zoo_dane = pd.read_csv("zoo.data.csv")
y_train = zoo_dane.iloc[:, 17]
x_train = zoo_dane.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]
y_train = y_train.to_numpy()
x_train = x_train.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

y_test = np_utils.to_categorical(y_test)
y_train = np_utils.to_categorical(y_train)



y_test = np.delete(y_test, 0, axis=1)
y_train = np.delete(y_train, 0, axis =1)
print(y_test)
print(y_test.shape)
print(x_train.shape)

print(x_train.shape[0])
x_train = x_train.reshape(80, 1, 16)



# TWORZENIE SIECI
siec = Siec_neuronowa()
siec.dodaj_warstwe(Nastepna_Warstwa(16, 10))      # Wejście 16 atrybutów wyjście 10 neuronów (Warstwa wejśćiowa)
siec.dodaj_warstwe(Aktywacja(sigmoidalna, sigmoidalna_prim))
siec.dodaj_warstwe(Nastepna_Warstwa(10, 7))       # Wejście 10 neuronów wyjście 8 neuronów (Warstwa ukryta)
siec.dodaj_warstwe(Aktywacja(sigmoidalna, sigmoidalna_prim))
siec.dodaj_warstwe(Nastepna_Warstwa(7, 7))        # Wejście 8 neuronów wyjście 8 wyjść (Warstwa wyjściowa)
siec.dodaj_warstwe(Aktywacja(sigmoidalna, sigmoidalna_prim))
siec.jaki_blad(mse, mse_prim)

#ROZPOCZecie uczenia
siec.proces_uczenia(x_train, y_train, epoki=1000, wspolczynnik_uczenia=1)

#TESTOWANIE MODELU
f = open("wyniki.txt", "a")
out = siec.test(x_test)
print("\n")
print("Odpowiedź sieci : ")
print(out, end="\n")

f.write("Odpowiedzi sieci \n")
f.write(str(out))
print(" \n Wartości prawdziwe : ")
print(y_test)
f.write("Wartości prawdziwe : \n")
errors = []
good = []
for i in range(len(y_test)):

    klasyfikacja = np.argmax(out[i]) + 1
    print("{}. Zaklasyfikowany typ przez sieć : {}".format(i+1, klasyfikacja))
    f.write("{}. Zaklasyfikowany typ przez sieć : {}\n".format(i+1, klasyfikacja))
    y_max = np.argmax(y_test[i]) + 1
    print("{}. Typ  poprawny : {}".format(i+1, y_max))
    f.write("Typ poprawny : {}\n".format(i+1, y_max))
    if klasyfikacja == y_max:
        good.append(klasyfikacja)
    else: errors.append(klasyfikacja)


suma = len(y_test)
procent_blad = (len(errors) / suma) * 100
procent_dobrze = (len(good) / suma) * 100
print('Błędnie rozpoznane typy zwierząt = {} / {} Procentowo : {}%'.format(len(errors), suma, procent_blad))
f.write('Błędnie rozpoznane typy zwierząt = {} / {} Procentowo : {}%\n'.format(len(errors), suma, procent_blad))
print('Poprawnie rozpoznane typy zwierząt = {} / {} Procentowo : {}%'.format(len(good), suma, procent_dobrze))
f.write('Poprawnie rozpoznane typy zwierząt = {} / {} Procentowo : {}%\n'.format(len(good), suma, procent_dobrze))
f.write(str(y_test[0]))
