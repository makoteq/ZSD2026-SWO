
# Milestone 0 – Pre-projekt

## Zakres prac

1. Identyfikacja i analiza dostępnych datasetów ruchu drogowego  
   (np. Kaggle lub inne publiczne zbiory) oraz licencji.
2. Wybór modelu do detekcji pojazdów (YOLOv26n).
3. Analiza licencji używanych narzędzi.
4. Wybór platformy sprzętowej (AI+ HAT, Google Coral lub inny).

---

# Milestone 1 — Pozyskanie danych i opracowanie algorytmów percepcji

**Cel:**  
Opracowanie podstawowego systemu detekcji i analizy ruchu pojazdów na podstawie danych wizyjnych oraz przygotowanie datasetu treningowego.

## Zakres prac

1. Wyznaczanie pasów ruchu (OpenCV albo model), realizowane w fazie setup.
2. Pozyskanie i wstępne parsowanie danych z RADARU.
3. Konfiguracja środowiska do transfer learning modelu YOLO i przygotowanie datasetu.
4. Implementacja algorytmów:
   - detekcji pojazdów
   - zapisu trajektorii ruchu
5. Konfiguracja środowiska drogowego w CARLA.
6. Przygotowanie pierwszych elementów zestawu scenariuszy zawierającego:
   - scenariusze niebezpieczne (wyprzedzanie, nadmierna prędkość, zmiana pasa)
   - scenariusze normalne
7. Konfiguracja platformy docelowej.
8. Stress testy platformy docelowej na przykładowym modelu detekcji.
9. Przygotowanie kodu badającego vitale programu.

## Metryki

1. Jesteśmy w stanie wyznaczyć linie pasów ruchu poprzez znaki poziome  
   (linia przerywana, ciągła, podwójna ciągła) oraz je rozróżniać.
2. Jesteśmy w stanie prowadzić ciągłą detekcję pojazdów oraz zapisywać ich trajektorie.
3. Wykrywamy **90% pojazdów** przejeżdżających (z wybranego datasetu).
4. System zapamiętuje wszystkie wykryte pojazdy i utrzymuje ciągłą trajektorię dla **co najmniej 75%** z nich.
5. Gotowy zestaw nagrań / danych.
6. Testy z ograniczeniami kodu jako sprawdzian przystosowania do platformy  
   (ograniczenie pamięci i mocy obliczeniowej).

---

# Milestone 2 — Predykcja zdarzeń niebezpiecznych i platforma sprzętowa

**Cel:**  
Opracowanie modułu predykcji zagrożeń.

## Zakres prac

1. Powiązanie danych z radaru z bounding boxami z YOLO.
2. Pomiar prędkości.
3. Transfer learning modelu.
4. Stress testy platformy docelowej na docelowym mechanizmie detekcji samochodów.
5. Wykrywanie wyprzedzania na podstawie trajektorii.
6. Generacja rozszerzonego zestawu scenariuszy zawierających:
   - profile prędkości
   - zdarzenia wyprzedzania
7. Integracja elementów mechanizmu predykcyjnego przewidującego ryzyko zdarzenia niebezpiecznego w horyzoncie czasowym **5–15 s**.
8. Ewaluacja skuteczności predykcji:
   - true alarm rate
   - false alarm rate
9. Wytypowanie potencjalnych scenariuszy wymagających poprawy algorytmu.

## Metryki

1. **False alarm rate:** max 40%
2. **True alarm rate:** min 60%
3. Działający system predykcji zagrożeń obejmujący:
   - prędkość
   - wyprzedzanie
   - zjeżdżanie z toru jazdy
4. Rozróżnianie typów pojazdów osobowych (np. sedan, pickup) głównie ze względu na masę.
5. Gotowy raport wydajności platformy docelowej.
6. Raport dotyczący obszarów algorytmu wymagających poprawy.

### Scenariusze w CARLA

- Przekroczenie prędkości — **3 przypadki**
- Wyprzedzanie — **3 przypadki**
- Anomalie — **5 przypadków**
- Kontrolne — **3 przypadki**

Dla każdego przypadku **3 różne warunki pogodowe**.

---

# Milestone 3 — Integracja systemu i mechanizm ostrzegania

**Cel:**  
Integracja komponentów i testy końcowe.

## Zakres prac

1. Integracja modułów predykcji i systemu embedded bez fizycznej kamery / radaru.
2. Udoskonalenie algorytmu detekcji do poziomu:
   - **false alarm rate ≤ 20%**
   - **true alarm rate ≥ 80%**
3. Wykazanie wykrywania sytuacji niebezpiecznej.
4. Wybór optymalnej platformy sprzętowej niezależnie od budżetu.
5. Testy końcowe:
   - scenariusze symulacyjne
   - pomiar czasu wyprzedzenia ostrzeżenia (**5–15 s**)

## Notatka

- Prędkość analizy klatki **nie ma znaczenia**, dopóki spełnia wymagania projektu.

## Metryki

1. **False alarm rate:** max 20%
2. **True alarm rate:** min 80%
3. Zestaw testów do walidacji systemu.
4. Działający model uruchomiony na platformie docelowej przy dostarczonych plikach z RADARU i kamery.
5. Lista najlepszych konfiguracji sprzętowych niezależnie od budżetu.
6. Demonstracja działającego systemu (nagranie) wraz z pokazanym zużyciem zasobów sprzętowych.
