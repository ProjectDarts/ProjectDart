🎯 ProjectDart

ProjectDart ist ein kamerabasiertes Dart-Erkennungssystem für Steeldartboards.
Das System erkennt Darttreffer automatisch über mehrere Kameras, berechnet die Trefferposition auf dem Board und übergibt das Ergebnis an eine Spiel-Engine.

Das Projekt kombiniert:

🖥 pygame UI

📷 OpenCV Vision-System

🎯 Dart-Regelengine (X01)

💾 SQLite Spieler- und Match-Datenbank

Das Ziel ist ein vollständig automatisches lokales Dartsystem ohne Sensorboard.

📷 Systemüberblick

ProjectDart nutzt 3 Kameras, die das Board aus unterschiedlichen Winkeln beobachten.

Die Treffererkennung erfolgt über mehrere parallele Vision-Detektoren:

Detector	Aufgabe
AbsDiff	erkennt neue Objekte über Bilddifferenz
Vector	erkennt Dartlinien über Hough-Linien
Shape	erkennt dartähnliche Formen
Takeout	erkennt entfernte Pfeile

Die Ergebnisse werden anschließend über eine Multicam-Fusion kombiniert.

🧠 Architektur
main.py
│
├── UI / State Machine (pygame)
├── Vision Thread
│
└── DartVisionSystem
      │
      ├── CameraHandler
      │
      ├── vision_absdiff.py
      ├── vision_vector.py
      ├── vision_shape.py
      └── vision_takeout.py
🗂 Projektstruktur
ProjectDart
│
├── main.py
├── calibrate.py
├── throw.py
│
├── vision.py
├── vision_absdiff.py
├── vision_vector.py
├── vision_shape.py
├── vision_takeout.py
├── vision_debug.py
├── vision_debug.ini
│
├── cam0_config.json
├── cam1_config.json
├── cam2_config.json
│
├── games
│   └── x01.py
│
├── database
│   └── database.py
│
└── README.md
🖥 Installation
1️⃣ Repository klonen
git clone https://github.com/USERNAME/projectdart.git
cd projectdart
2️⃣ Python Abhängigkeiten installieren
pip install opencv-python pygame numpy
3️⃣ Projekt starten
python main.py
📷 Kamerasetup

Empfohlenes Setup:

      Cam1
        \
         \
Cam0 ---- Board ---- Cam2
Anforderungen

3 Kameras

1080p empfohlen

stabile Beleuchtung

feste Kamerapositionen

Blickwinkel ca. 30-45° zum Board

🎯 Kalibrierung

Vor der ersten Nutzung müssen die Kameras kalibriert werden.

In der Lobby:

Kameras Kalibrieren

Für jede Kamera werden 4 Punkte auf der Spinne gesetzt:

oben

rechts

unten

links

Die Kalibrierung erzeugt:

cam0_config.json
cam1_config.json
cam2_config.json

Diese enthalten die Referenzpunkte für die Homography.

🧮 Vision Pipeline

Der Erkennungsablauf:

Camera Frame
     │
     ▼
Homography
(Boardspace)
     │
     ▼
Detectors
 ├─ AbsDiff
 ├─ Vector
 └─ Shape
     │
     ▼
Best Candidate per Camera
     │
     ▼
Multicam Fusion
     │
     ▼
Score Calculation
     │
     ▼
Game Engine
🔍 Vision Detectors
AbsDiff

Datei:

vision_absdiff.py

Funktionsweise:

reference_board - current_frame
        │
   threshold
        │
   contours
        │
   tip estimation

Gut geeignet für:

neue Pfeile

stabile Beleuchtung

Vector Detector

Datei:

vision_vector.py

Funktionsweise:

edges (Canny)
     │
HoughLinesP
     │
Line candidates
     │
Tip = endpoint towards board center

Gut für:

lange sichtbare Pfeilschäfte

Shape Detector

Datei:

vision_shape.py

Analysiert:

Konturlänge

Breite

Achse

Schlankheit

geometrische Eigenschaften

Dient als zusätzlicher Filter gegen Fehlkandidaten.

Takeout Detector

Datei:

vision_takeout.py

Erkennt, wenn Pfeile entfernt wurden.

Signalisiert:

NEXT_PLAYER
🎯 Multicam Fusion

Das System kombiniert Treffer aus mehreren Kameras.

Beispiel:

Cam0 -> (302,198)
Cam1 -> (298,203)
Cam2 -> (305,201)

↓ Weighted Fusion

Final -> (301,201)

Outlier werden verworfen.

🎮 Spielmodus

Aktuell implementiert:

X01

Features:

Single In

Double In

Single Out

Double Out

Master Out

Legs

Sets

Endlosmodus

Undo

💾 Datenbank

SQLite Datenbank unter:

%LOCALAPPDATA%/ProjectDart/projectdart.db

Tabellen:

players
id
name
created_at
when_last_played
matches
id
mode
start_score
winner_id
timestamp
throws
id
match_id
player_id
segment
multiplier
x_rel
y_rel
timestamp
👤 Spieler-Autocomplete

Beim Eingeben von Spielernamen:

Vorschläge erscheinen automatisch

maximal 3 Vorschläge

sortiert nach

when_last_played DESC

Verhalten:

Name existiert → verwenden
Name neu → automatisch anlegen
🧪 Debug Modus

Debugfenster können aktiviert werden über:

vision_debug.ini

Beispiel:

[vision]
debugging = 1
warp_size = 800
show_full = 1
show_warp = 1

Debug zeigt:

Fullframe

Warp-Board

erkannte Spitzen

Linien

Detektoren

⚠️ Bekannte Herausforderungen

Typische Probleme bei Vision-Dartsystemen:

Flight wird als Spitze erkannt

Dartmitte statt Spitze erkannt

Boardschwingung

Lichtänderungen

Reflexionen

Linienartefakte

Mehrere Detektoren und Multicam-Fusion reduzieren diese Effekte.

🚀 Roadmap

Geplante Verbesserungen:

stabilere Shape-Erkennung

bessere Multicam-Fusion

Confidence-Normalisierung

Heatmap Statistik

Match History

Web UI

ML-basierte Dart-Erkennung

📜 Lizenz

Noch keine Lizenz definiert.

Empfohlen:

MIT License

oder

GPLv3
👤 Autor

Projekt entwickelt von

ProjectDart
