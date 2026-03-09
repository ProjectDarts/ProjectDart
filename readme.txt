# ProjectDart

ProjectDart ist ein lokales Dart-System mit Kameraerkennung, Kalibrierung, Spieloberfläche und Datenbank. Ziel ist es, Treffer auf einem Steeldartboard automatisch zu erkennen, an die Spiel-Logik zu übergeben und den Spielfluss inklusive Spielerwechsel und Takeout-Erkennung zu unterstützen.

## Funktionsumfang

ProjectDart besteht aus vier Hauptbereichen:

* **Spieloberfläche und Spielsteuerung** mit `pygame`
* **Vision-System** zur Treffer- und Takeout-Erkennung mit OpenCV
* **Kalibrierungssystem** für die Kameras
* **Datenbank** für Spieler, Spiele und Würfe

Aktuell ist das System auf ein lokales Setup mit **3 Kameras** ausgelegt, die rund um das Board montiert sind.

---

## Projektstruktur

```text
ProjectDart/
│
├─ main.py
├─ calibrate.py
├─ throw.py
├─ vision.py
├─ vision_absdiff.py
├─ vision_vector.py
├─ vision_shape.py
├─ vision_takeout.py
├─ vision_debug.py
├─ vision_debug.ini
│
├─ cam0_config.json
├─ cam1_config.json
├─ cam2_config.json
│
├─ games/
│  └─ x01.py
│
├─ database/
│  └─ database.py
│
└─ README.md
```

Je nach Stand des Projekts können zusätzliche Dateien vorhanden sein.

---

## Voraussetzungen

## Software

* Python 3.10 oder neuer
* Windows empfohlen
* OpenCV mit DirectShow-Support
* pygame
* sqlite3 (bei Python standardmäßig enthalten)
* numpy

## Python-Pakete installieren

```bash
pip install opencv-python pygame numpy
```

---

## Hardware-Setup

Empfohlenes Setup:

* 1 Steeldartboard
* 3 Kameras
* Kameras ungefähr um **120° versetzt** montiert
* Kameras schauen **schräg** auf das Board
* möglichst konstante Beleuchtung
* möglichst feste Kamerapositionen

Wichtig:

* Das System arbeitet mit einer perspektivischen Entzerrung über 4 Kalibrierpunkte pro Kamera.
* Diese Kalibrierung definiert die Lage des Boards im Kamerabild.
* Der Warp dient in erster Linie der mathematischen Zuordnung und dem Debugging, nicht als harter Bildausschnitt.

---

## Schnellstart

## 1. Projekt starten

```bash
python main.py
```

## 2. Kameras kalibrieren

Im Lobby-Menü kann die Kalibrierung gestartet werden.

Für jede Kamera werden 4 Punkte auf der Spinne gesetzt:

1. **Oben**
2. **Rechts**
3. **Unten**
4. **Links**

Die Punkte werden in folgenden Dateien gespeichert:

* `cam0_config.json`
* `cam1_config.json`
* `cam2_config.json`

## 3. Spieler anlegen oder auswählen

Im Setup können Spielernamen eingegeben werden.

Die Datenbank schlägt vorhandene Namen automatisch vor:

* maximal 3 Vorschläge
* sortiert nach `when_last_played`
* vorhandene Namen werden weiterverwendet
* neue Namen werden automatisch angelegt

## 4. Spiel starten

Aktuell ist insbesondere `X01` integriert.

---

## Hauptmodule im Überblick

## `main.py`

Zentrale Steuerung des Programms.

Aufgaben:

* Initialisiert `pygame`
* verwaltet die UI-Zustände:

  * `LOBBY`
  * `GAME_SELECT`
  * `SETTINGS`
  * `GAME`
* startet das Vision-System in einem Hintergrund-Thread
* verarbeitet Treffer über eine Queue
* startet die Kalibrierung
* übergibt erkannte Würfe an die Spiellogik

## `calibrate.py`

Kalibrierung der Kameras.

Aufgaben:

* öffnet jede Kamera einzeln
* zeigt das Livebild
* erlaubt das Setzen von 4 Referenzpunkten
* speichert die Kalibrierung als JSON-Datei

## `vision.py`

Zentrale Verwaltung des Vision-Systems.

Aufgaben:

* lädt und initialisiert die Unterdetektoren
* verwaltet alle Kameras
* liest Frames ein
* wendet Homographie und Boardspace-Zuordnung an
* sammelt Kandidaten aus mehreren Erkennungsmethoden
* fusioniert Kameraergebnisse
* berechnet daraus die wahrscheinlichste Trefferposition
* erkennt `missed` (0 Punkte)
* erkennt Takeout-Events
* gibt Treffer oder `NEXT_PLAYER` an `main.py` weiter

Wichtig:

`vision.py` ist der Orchestrator. Die eigentliche Detailerkennung wird in Untermodulen ausgeführt.

## `vision_absdiff.py`

AbsDiff-basierte Treffererkennung.

Prinzip:

* Referenzbild des leeren Boards
* Differenz zum aktuellen Bild
* Thresholding
* Konturerkennung
* geometrische Analyse
* Schätzung der Pfeilspitze

Stärken:

* gut bei frisch eingeschlagenen Pfeilen
* relativ schnell
* robust bei stabilem Licht

Schwächen:

* empfindlich bei Bewegungen und Lichtschwankungen
* kann ohne Zusatzfilter Flight oder Pfeilmitte bevorzugen

## `vision_vector.py`

Linien- bzw. Vektor-basierte Erkennung.

Prinzip:

* Kantenbild mit Canny
* HoughLinesP zur Liniensuche
* plausible Dart-Linien extrahieren
* geeigneten Linienendpunkt als Spitzenkandidat bestimmen

Stärken:

* hilfreich bei langen, gut sichtbaren Pfeilschäften
* unabhängig von klassischem AbsDiff

Schwächen:

* kann auf Boardkanten, Spinne oder andere Linien reagieren
* braucht gute Filterung

## `vision_shape.py`

Formbasierte Erkennung.

Prinzip:

* segmentiert neue Objekte
* bewertet Geometrie wie Länge, Breite, Achse, Schlankheit und Konturform
* schätzt daraus einen Spitzenkandidaten

Stärken:

* sinnvoll als dritte, redundante Erkennungsmethode
* kann Kandidaten bestätigen oder verwerfen

Schwächen:

* braucht gutes Tuning für reale Hardware

## `vision_takeout.py`

Erkennung gezogener Pfeile.

Aufgaben:

* vergleicht das aktuelle Board mit einem sauberen Referenzbild
* erkennt, ob Pfeile entfernt wurden
* meldet an `vision.py`, wenn das Board wieder frei ist

Wichtig:

Dieses Modul dient **nicht** der Treffererkennung, sondern nur der Spielflusssteuerung.

## `vision_debug.py`

Entwicklungs- und Debug-Anzeige.

Aufgaben:

* zeigt optional das Full-Frame-Bild
* zeigt optional das gewarpte Boardbild
* blendet erkannte Spitzen, Linien und Kandidaten ein
* kann über `vision_debug.ini` aktiviert oder deaktiviert werden

## `games/x01.py`

Spiellogik für X01.

Unterstützt unter anderem:

* Startscore
* Single In / Double In
* Single Out / Double Out / Master Out
* Legs
* Sets
* Endlosmodus
* Undo
* Bust-Regeln

## `database/database.py`

Datenbankverwaltung über SQLite.

Verwendet einen persistenten Speicherort unter:

```text
%LOCALAPPDATA%/ProjectDart/projectdart.db
```

Tabellen:

* `players`
* `matches`
* `throws`

Zusätzlich wird `when_last_played` verwendet, um Spielervorschläge zu priorisieren.

---

## Vision-System – Datenfluss

Der aktuelle Datenfluss sieht grob so aus:

1. `vision.py` liest pro Kamera ein neues Frame.
2. Aus den Kalibrierpunkten wird eine Homographie genutzt.
3. Das Bild wird für Boardspace und Debug-Zwecke gewarpt.
4. Mehrere Detektoren analysieren das Bild parallel:

   * AbsDiff
   * Vector
   * Shape
5. Jeder Detektor liefert Kandidaten mit Position und Confidence.
6. `vision.py` wählt pro Kamera die besten Kandidaten.
7. Danach werden die Ergebnisse mehrerer Kameras fusioniert.
8. Aus der finalen Position wird der Score oder `missed` bestimmt.
9. Das Ergebnis wird an `main.py` zurückgegeben.

---

## Multicam-Fusion

Da 3 Kameras vorhanden sind, wird nicht blind einer einzelnen Kamera vertraut.

Typischer Ablauf:

* jede Kamera liefert eine vermutete Spitzenposition
* jede Position erhält eine Confidence
* `vision.py` bildet daraus die wahrscheinlichste Gesamtposition

Mögliche Verfahren:

* gewichteter Mittelwert
* gewichteter Median
* Konsistenzprüfung zwischen den besten zwei Kameras
* Outlier-Verwerfung

Das Ziel ist, dass Fehlkandidaten wie Flight oder Pfeilmitte durch die anderen Kameras korrigiert werden.

---

## Missed-Erkennung

Ein `missed` ist ein Wurf, dessen berechnete Spitze **außerhalb des gültigen Boardbereichs** liegt.

Wichtig:

* Die Kalibrierpunkte definieren die geometrische Lage des Boards.
* Das Gesamtkamerabild darf weiterhin vollständig analysiert werden.
* Der Warp dient der Zuordnung, nicht als physische Begrenzung des Sichtfelds.
* Liegt die Spitze außerhalb des `double_outer`, wird der Wurf als `0 Punkte` gemeldet.

---

## Debugging

Die Debug-Ausgabe wird über `vision_debug.ini` gesteuert.

Beispiel:

```ini
[vision]
debugging = 1
warp_size = 800
show_full = 1
show_warp = 1
```

Bedeutung:

* `debugging = 1` → Debug aktiv
* `debugging = 0` → Debug komplett aus
* `show_full = 1` → Full-Frame anzeigen
* `show_warp = 1` → Warp anzeigen

Sobald das System stabil läuft, kann Debugging komplett deaktiviert werden.

---

## Datenbankverhalten für Spieler

Beim Eingeben von Spielernamen gilt:

* beim Tippen werden Vorschläge angezeigt
* es werden maximal 3 Namen vorgeschlagen
* Reihenfolge nach `when_last_played DESC`
* existierende Namen werden wiederverwendet
* neue Namen werden automatisch angelegt
* beim Spielstart werden die verwendeten Spieler mit aktuellem Zeitstempel aktualisiert

---

## Bekannte Herausforderungen

Aktuell typische Problemfelder bei Kameradartsystemen:

* Flight wird als Spitze erkannt
* Pfeilmitte wird statt Spitze erkannt
* Boardschwingung direkt nach Einschlag
* Lichtänderungen / Reflexionen
* Unterschiede zwischen Kameras
* schlecht gefilterte Linien im Fullframe
* Takeout-Fehlinterpretationen

Darum ist der modulare Aufbau mit mehreren Detektoren wichtig.

---

## Entwicklungsphilosophie

Die Architektur ist bewusst modular aufgebaut:

* `vision.py` verwaltet nur
* Unterdetektoren erkennen
* `main.py` steuert den Spielfluss
* `x01.py` enthält die Spielregeln
* `vision_debug.py` bleibt dauerhaft optional zuschaltbar

Das Ziel ist, neue Erkennungsmethoden ergänzen zu können, ohne die gesamte Vision-Logik neu schreiben zu müssen.

---

## Empfohlene nächste Schritte

1. **AbsDiff weiter stabilisieren**

   * bessere Spitzenlokalisierung
   * robustere Filter gegen Flight

2. **Vector sauber integrieren**

   * Linienfilter verbessern
   * nur plausible Dartlinien akzeptieren

3. **Shape-Detector ergänzen oder feinjustieren**

   * Geometrie stärker auswerten
   * Kandidaten besser priorisieren

4. **Multicam-Fusion verbessern**

   * Outlier robuster verwerfen
   * Confidence normalisieren

5. **Takeout robuster machen**

   * lokale statt globale Prüfung
   * mehr Stabilität bei Lichtschwankungen

6. **Tests und Logging verbessern**

   * Vision-Parameter zentralisieren
   * reproduzierbare Debug-Ausgaben

---

## Start im Entwicklungsmodus

```bash
python main.py
```

## Kalibrierung direkt starten

```bash
python calibrate.py
```

---

## Hinweise zur EXE

Das Projekt ist bereits so vorbereitet, dass Pfade sowohl im Python-Betrieb als auch im EXE-Betrieb funktionieren.

Wichtige Punkte:

* Ressourcenpfade werden über Helper-Funktionen aufgelöst
* Datenbank wird nicht im temporären Bundle gespeichert, sondern dauerhaft in `LOCALAPPDATA`
* Kalibrierdateien liegen im Ausführungsordner

---

## Lizenz / Nutzung

Dieses Projekt ist aktuell ein individuelles, lokal entwickeltes Dart-Erkennungssystem.

Falls du das öffentlich veröffentlichen willst, solltest du noch ergänzen:

* Lizenz
* bekannte Einschränkungen
* unterstützte Hardware
* Installationsanleitung für Endnutzer

---

## Kurzfazit

ProjectDart ist ein lokales Mehrkamera-Dartsystem mit:

* Spieloberfläche
* Kalibrierung
* Vision-Subsystem
* Debug-Ansicht
* Datenbankgestützter Spielerverwaltung

Der wichtigste technische Schwerpunkt liegt aktuell auf einer sauberen, redundanten Pfeilerkennung mit mehreren Detektoren und robuster Mehrkamera-Fusion.
