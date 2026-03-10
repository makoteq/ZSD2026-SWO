# CARLA 0.9.16 Instalacja

## Instalacja

Pobierz CARLA 0.9.16 z GitHub:
[Wersja CARLA 0.9.16](https://github.com/carla-simulator/carla/releases/tag/0.9.16/)

Rozpakuj archiwum po pobraniu.

---

## Konfiguracja

Utwórz nowy projekt w `Carla_0.9.16/PythonAPI`.

Możesz użyć dowolnego IDE (np. PyCharm, VS Code) lub pracować bezpośrednio z terminala.

### Środowisko Pythona

Utwórz wirtualne środowisko z **Python 3.12**.

Przykład użycia PyCharm:

Wybierz odpowiednią wersję Pythona podczas tworzenia nowego projektu,
lub później przejdź do: <br>
File → Settings → Python → Interpreter → Add Interpreter → Add Local Interpreter.

Upewnij się, że środowisko jest aktywowane i że używasz właściwej wersji Pythona.

---

## Uruchamianie symulacji

W katalogu `CARLA_0.9.16` uruchom plik wykonywalny: `CarlaUE4.exe`.

### Zależności i przykładowe skrypty

Zainstaluj zależności i uruchamiaj pliki `.py` **podczas gdy symulator jest uruchomiony**:

```bash
pip install numpy pygame carla opencv-python
```

Wszystkie przykładowe pliki Pythona znajdziesz w `Carla_0.9.16/PythonAPI/examples`

---

## Wersje

* Python: 3.12
* CARLA: 0.9.16

---

## Uwagi

Wszystkie skrypty Pythona związane z CARLA z tego repozytorium powinny być umieszczone w:

`Carla_0.9.16/PythonAPI`

W przeciwnym razie mogą nie działać poprawnie.
