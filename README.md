# Wielowarstwowa Sieć Neuronowa na Zbiorze Danych MNIST

## Opis
Ten projekt implementuje wielowarstwową sieć neuronową (MLP) do klasyfikacji ręcznie pisanych cyfr przy użyciu zbioru danych MNIST. Model został zaimplementowany wyłącznie przy użyciu biblioteki NumPy, bez użycia dedykowanych bibliotek do sieci neuronowych.

## Zbiór Danych
Zbiór danych MNIST składa się z 70 000 obrazów w skali szarości przedstawiających cyfry (0-9), każdy o wymiarach 28x28 pikseli. Jest to szeroko stosowany benchmark w dziedzinie uczenia maszynowego.

## Instalacja
Upewnij się, że masz zainstalowanego Pythona (>=3.7) i utwórz wirtualne środowisko:

```bash
python -m venv venv
source venv/bin/activate  # W systemie Windows użyj `venv\Scripts\activate`
```

Zainstaluj wymagane zależności:

```bash
pip install -r requirements.txt
```

## Pobieranie Danych
Dane można pobrać bezpośrednio ze strony [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/) i rozpakować w katalogu `data`.

## Architektura Modelu
Model składa się z następujących warstw:
- Warstwa wejściowa: 28x28 (spłaszczona do 784 neuronów)
- Warstwy ukryte: w pełni połączone warstwy z funkcjami aktywacji ReLU
- Warstwa wyjściowa: 10 neuronów (po jednym na każdą cyfrę) z aktywacją softmax

## Licencja
Ten projekt jest open-source i dostępny na licencji MIT.

