# Cifar-10

Baza de date contine 60.000 de imagini colorate impartite in 10 clase cu 6000 de imagini pe clasa

Sunt in total 50.000 de imagini de antrenare si 10.000 de testare.

Cele 10 clase sunt:
 - avion
 - automobil
 - pasare
 - pisica
 - caprioara
 - caine
 - broasca
 - cal
 - barca
 - camion
 
Codul se ruleaza din `app.py`.
Structura fisierelor este urmatoarea:
  - cifar-10-batches-py:
    - ...
  - app.py
  - lbp.py
  - load_data.py
  - test_data
  - train_data
  
`app.py` este script-ul principal. `lbp.py` face apel la `load_data.py` pentru a incarca datele
 din baza de date (din fisierul `cifar-10-batches-py`) si apoi pentru a genera features pe baza
  histogramelor generate de regiuni de imagine.

`train_data` si `test_data` sunt fisiere pickle ce contin DataFrames cu features pentru antrenare
 si testare. Exista pentru a crea o metoda de caching pentru date (histogramele lbp sunt foarte
  costisitoare ca timp de executie).
