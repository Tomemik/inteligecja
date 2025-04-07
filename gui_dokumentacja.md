## Struktura GUI

Interfejs aplikacji jest podzielony na dwie główne sekcje: panel konfiguracyjny (po lewej) oraz panel wizualizacji (po prawej). Okno aplikacji ma domyślny rozmiar 1200x600 pikseli.

### Panel konfiguracyjny

Panel konfiguracyjny znajduje się po lewej stronie interfejsu i pozwala użytkownikowi ustawić parametry algorytmu przed jego uruchomieniem. Zawiera następujące elementy:

- **Attack type (Typ ataku)**:
  - Pole typu `QComboBox` (lista rozwijana) umożliwiająca wybór typu ataku do analizy.
  - Dostępne opcje: `smurf`, `neptune`, `normal`, `satan`, `ipsweep`, `portsweep`, `nmap`, `back`, `warezclient`, `teardrop`, `pod`, `guess_passwd`, `buffer_overflow`, `land`, `warezmaster`, `imap`, `rootkit`, `loadmodule`, `ftp_write`, `multihop`, `phf`, `perl`, `spy`.
  - Domyślna wartość: `smurf`.
  - Funkcja: Wybór typu ataku określa, dla jakiego rodzaju danych algorytm będzie uruchamiany.

- **Population Size (Rozmiar populacji)**:
  - Pole typu `QSpinBox` (licznik numeryczny) do ustawienia liczby osobników w populacji.
  - Zakres: od 5 do 5000.
  - Domyślna wartość: 100.
  - Funkcja: Określa, ile chromosomów (osobników) będzie brało udział w każdej generacji algorytmu.

- **Generations (Liczba generacji)**:
  - Pole typu `QSpinBox` określające liczbę iteracji (generacji) algorytmu.
  - Zakres: od 1 do 10000.
  - Domyślna wartość: 100.
  - Funkcja: Ustawia, przez ile cykli algorytm będzie działał, o ile nie zostanie wcześniej zatrzymany przez limit stagnacji.

- **Stagnation limit (Limit stagnacji)**:
  - Pole typu `QSpinBox` określające maksymalną liczbę generacji bez poprawy wyniku, po której algorytm się zatrzyma (wczesne zatrzymanie).
  - Zakres: od 0 do 50.
  - Domyślna wartość: 5.
  - Funkcja: Jeśli przez podaną liczbę generacji nie będzie poprawy w najlepszym wyniku, algorytm zakończy działanie.

- **Mutation Rate (%) (Współczynnik mutacji)**:
  - Pole typu `QDoubleSpinBox` (licznik numeryczny z wartościami dziesiętnymi) określające procentową szansę na mutację osobnika.
  - Zakres: od 0.0 do 100.0.
  - Krok: 0.1.
  - Domyślna wartość: 5.0.
  - Funkcja: Ustawia prawdopodobieństwo, z jakim chromosomy będą podlegać losowym zmianom (mutacjom) w każdej generacji.

- **Elitism (Elityzm)**:
  - Pole typu `QCheckBox` (pole wyboru) umożliwiające włączenie elityzmu.
  - Domyślnie: wyłączone (odznaczone).
  - Funkcja: Po włączeniu elityzmu najlepsze osobniki z każdej generacji są automatycznie przenoszone do następnej generacji, co zapobiega utracie najlepszych rozwiązań.

- **Selection method (Metoda selekcji)**:
  - Pole typu `QLabel` (etykieta informacyjna) pokazujące używaną metodę selekcji.
  - Wartość: `Tournament` (selekcja turniejowa).
  - Funkcja: Informuje użytkownika o metodzie selekcji stosowanej w algorytmie (nie można jej zmienić w GUI).

- **Crossover method (Metoda krzyżowania)**:
  - Pole typu `QLabel` pokazujące używaną metodę krzyżowania.
  - Wartość: `One-point crossover` (krzyżowanie jednopunktowe).
  - Funkcja: Informuje użytkownika o metodzie krzyżowania stosowanej w algorytmie (nie można jej zmienić w GUI).

- **Mutation method (Metoda mutacji)**:
  - Pole typu `QLabel` pokazujące używaną metodę mutacji.
  - Wartość: `Mutation of a random 1 feature` (mutacja jednej losowej cechy).
  - Funkcja: Informuje użytkownika o metodzie mutacji stosowanej w algorytmie (nie można jej zmienić w GUI).

- **Run (Uruchom)**:
  - Przycisk typu `QPushButton` uruchamiający algorytm z wybranymi parametrami.
  - Funkcja: Po kliknięciu rozpoczyna działanie algorytmu, a panel wizualizacji zaczyna wyświetlać wyniki w czasie rzeczywistym.

### Panel wizualizacji

Panel wizualizacji znajduje się po prawej stronie interfejsu i służy do wyświetlania informacji o stanie algorytmu oraz wykresu fitness. Zawiera następujące elementy:

- **Algorythm state (Stan algorytmu)**:
  - Pole typu `QLabel` pokazujące aktualny stan algorytmu.
  - Możliwe wartości: `Not running`, `Starting`, `Running`.
  - Domyślna wartość: `Not running`.
  - Funkcja: Informuje użytkownika, czy algorytm jest w trakcie działania, czy został zatrzymany.

- **Best chromosome fitness (Najlepszy fitness chromosomu)**:
  - Pole typu `QLabel` wyświetlające fitness najlepszego chromosomu w populacji.
  - Domyślna wartość: `N/A`.
  - Funkcja: Podczas działania algorytmu wyświetla wartość fitness najlepszego chromosomu z dokładnością do 4 miejsc po przecinku. Po zakończeniu algorytmu wartość jest resetowana do `N/A`.

- **Maximum fitness on test set (Maksymalny fitness na zbiorze testowym)**:
  - Pole typu `QLabel` wyświetlające maksymalny fitness uzyskany na zbiorze testowym.
  - Domyślna wartość: `N/A`.
  - Funkcja: Po zakończeniu algorytmu wyświetla maksymalną wartość fitness na zbiorze testowym z dokładnością do 4 miejsc po przecinku. Po ponownym uruchomieniu algorytmu wartość jest resetowana do `N/A`.

- **Wykres fitness**:
  - Wykres generowany za pomocą Matplotlib, osadzony w GUI przy użyciu `FigureCanvasQTAgg`.
  - Oś X: Numer iteracji (generacji), od 1 do liczby wykonanych iteracji.
  - Oś Y: Wartość fitness, w zakresie od 0.0 do 1.0 (automatycznie skalowana).
  - Tytuł: `Fitness history`.
  - Legenda: `Fitness` (linia niebieska, oznaczona jako 'b-').
  - Funkcja: Wykres aktualizuje się dynamicznie podczas działania algorytmu, pokazując zmiany wartości fitness w każdej generacji. Po zakończeniu algorytmu lub przed jego uruchomieniem wykres jest czyszczony.