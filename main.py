import pygame
import sys
import os
import threading
from queue import Queue
import json
import calibrate  # Importiert die externe Kalibrierungslogik aus calibrate.py


# --- EXE-PFAD-KORREKTUR (VERBESSERT) ---
def resource_path(relative_path):
    """
    Liefert den korrekten absoluten Pfad zu einer Ressource zurück.

    Hintergrund:
    - Im normalen Python-Betrieb liegen Dateien relativ zur .py-Datei.
    - Wenn das Projekt mit PyInstaller zu einer .exe gebaut wurde,
      werden Dateien in ein temporäres Verzeichnis entpackt.
    - PyInstaller stellt dafür sys._MEIPASS bereit.

    Parameter:
        relative_path (str): Relativer Dateipfad, z. B. "cam0_config.json"

    Rückgabe:
        str: Absoluter Pfad zur gewünschten Datei
    """
    if hasattr(sys, '_MEIPASS'):
        # Fall 1: Das Programm läuft als gebündelte EXE (PyInstaller)
        base_path = sys._MEIPASS
    else:
        # Fall 2: Normale Ausführung als Python-Skript
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)


# Erlaubt das Importieren von Modulen aus Unterordnern relativ zu main.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import der Spielmodi
from games.x01 import X01Game
from games.cricket import CricketGame

# Import der Datenbankverwaltung für Spieler, Suchvorschläge usw.
from database.database import DatabaseManager

# Import des Vision-Systems zur Kamera-/Board-Erkennung
from vision import DartVisionSystem


class MainManager:
    """
    Zentrale Steuerklasse der Anwendung.

    Aufgaben dieser Klasse:
    - Initialisiert pygame und alle UI-Ressourcen
    - Verwaltet den aktuellen Programmzustand (Lobby, Auswahl, Einstellungen, Spiel)
    - Startet und überwacht das Vision-System
    - Reagiert auf Maus- und Tastatur-Eingaben
    - Leitet Treffer aus dem Vision-System an das aktive Spiel weiter
    - Zeichnet je nach Zustand die passende Oberfläche
    """

    def __init__(self):
        """
        Konstruktor der Hauptklasse.

        Hier werden:
        - pygame gestartet
        - Fenster und Fonts erzeugt
        - Datenbank initialisiert
        - die Kommunikation mit dem Vision-Thread vorbereitet
        - Standardwerte für UI und Spielkonfiguration gesetzt
        """
        pygame.init()

        # Fenster in Full-HD-Auflösung erzeugen
        self.screen = pygame.display.set_mode((1920, 1080))
        pygame.display.set_caption("ProjectDart Pro - Version 1.9.5 (Missed Feld Fix)")

        # Clock begrenzt später die Framerate
        self.clock = pygame.time.Clock()

        # --- FONT INITIALISIERUNG ---
        # Verschiedene Schriftarten für Titel, Menüs, Statusanzeigen usw.
        self.font_title = pygame.font.SysFont("Segoe UI", 100, bold=True)
        self.font_menu = pygame.font.SysFont("Segoe UI", 36)
        self.font_menu_bold = pygame.font.SysFont("Segoe UI", 42, bold=True)
        self.font_kb = pygame.font.SysFont("Segoe UI", 30, bold=True)
        self.font_status = pygame.font.SysFont("Segoe UI", 24, italic=True)
        self.font_suggestion = pygame.font.SysFont("Segoe UI", 24)

        # Datenbankmanager für Spielerverwaltung
        self.db = DatabaseManager()

        # Queue für die Thread-sichere Kommunikation zwischen Kamera/Vision und Hauptlogik
        # Das Vision-System legt Treffer in diese Queue, main.py liest sie aus.
        self.hit_queue = Queue()

        # --- VISION INITIALISIERUNG ---
        # Vision-System wird zunächst auf None gesetzt und danach gestartet
        self.vision_system = None
        self.start_vision_thread()

        # Aktueller UI-Zustand:
        # "LOBBY"       -> Spielsetup / Spielernamen
        # "GAME_SELECT" -> Spielmodus wählen
        # "SETTINGS"    -> Spieloptionen einstellen
        # "GAME"        -> laufendes Spiel
        self.state = "LOBBY"

        # Hier wird später die aktuelle Spielinstanz gespeichert (X01 oder Cricket)
        self.game_instance = None

        # Standardmäßig ist X01 vorausgewählt
        self.selected_game_mode = "X01"

        # Setup-Variablen für die Spielerlobby
        self.selected_player_count = 0              # Anzahl ausgewählter Spieler
        self.selected_names = [""] * 8              # Bis zu 8 Spielernamen
        self.active_input_idx = 0                   # Welches Eingabefeld gerade aktiv ist
        self.kb_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"  # Virtuelle Tastatur

        # Vorschläge aus der Datenbank für Spielernamen
        self.name_suggestions = []

        # Spiel-Konfiguration mit Standardwerten
        self.config = {
            "start_score": 501,     # Startpunktzahl für X01
            "in_mode": "Single In", # Einstieg: Single In / Double In
            "out_mode": "Double Out",  # Finish-Regel
            "player_count": 0,      # Wird beim Spielstart gesetzt
            "endlos": False,        # Endlosmodus für X01
            "legs_to_win": 3,       # Leg-Anzahl
            "sets_to_win": 1,       # Aktuell vorbereitet, aber hier nicht aktiv genutzt
            "cut_throat": False     # Cricket-Modus: Normal oder Cut Throat
        }

        # In diesem Dictionary werden pro Frame klickbare Buttons abgelegt
        # Format: {"ACTION_NAME": pygame.Rect(...)}
        self.buttons = {}

    def start_vision_thread(self):
        """
        Initialisiert das Vision-System und startet es in einem separaten Thread.

        Warum ein eigener Thread?
        - Die Kamera-/Bildverarbeitung läuft kontinuierlich
        - Sie soll die UI und die Hauptschleife nicht blockieren
        - Treffer werden über hit_callback direkt in die Queue geschrieben
        """
        try:
            # Vision-System erzeugen.
            # hit_callback=self.hit_queue.put bedeutet:
            # Jeder erkannte Treffer wird direkt in die Queue gelegt.
            self.vision_system = DartVisionSystem(hit_callback=self.hit_queue.put)

            # Eigenen Thread starten, damit vision_system.run() parallel läuft
            self.vision_thread = threading.Thread(
                target=self.vision_system.run,
                daemon=True
            )
            self.vision_thread.start()

            print("[INFO] Kamera-Thread erfolgreich gestartet.")
        except Exception as e:
            print(f"[FEHLER] Vision System konnte nicht geladen werden: {e}")

    def check_calibration_status(self):
        """
        Prüft, wie viele Kameras bereits korrekt kalibriert sind.

        Vorgehen:
        - Es werden drei JSON-Dateien erwartet: cam0_config.json bis cam2_config.json
        - Jede Datei muss ein Feld "points" enthalten
        - "points" muss eine Liste mit genau 4 Punkten sein

        Rückgabe:
            int: Anzahl der erfolgreich kalibrierten Kameras (0 bis 3)
        """
        calibrated_cams = 0

        # Es werden genau 3 Kameras erwartet: 0, 1, 2
        for i in range(3):
            config_file = resource_path(f"cam{i}_config.json")

            if os.path.exists(config_file):
                try:
                    with open(config_file, "r") as f:
                        data = json.load(f)

                        # Eine Kamera gilt nur dann als kalibriert,
                        # wenn "points" existiert und genau 4 Einträge hat
                        if "points" in data and isinstance(data["points"], list) and len(data["points"]) == 4:
                            calibrated_cams += 1
                except Exception:
                    # Fehler beim Lesen/Parsen werden ignoriert,
                    # die Kamera zählt dann einfach nicht als kalibriert
                    pass

        return calibrated_cams

    def refresh_name_suggestions(self):
        """
        Aktualisiert die Vorschläge für Spielernamen.

        Es werden maximal 3 Vorschläge aus der Datenbank geladen,
        basierend auf dem gerade aktiven Eingabefeld.

        Verhalten:
        - Wenn kein Spieler ausgewählt ist -> keine Vorschläge
        - Wenn das aktive Namensfeld leer ist -> keine Vorschläge
        - Sonst Datenbanksuche mit Suchbegriff aus aktivem Feld
        """
        if self.selected_player_count <= 0:
            self.name_suggestions = []
            return

        current = self.selected_names[self.active_input_idx].strip()
        if not current:
            self.name_suggestions = []
            return

        self.name_suggestions = self.db.search_players(current, limit=3)

    def finalize_selected_players(self):
        """
        Verarbeitet alle aktuell eingegebenen Spielernamen.

        Aufgabe:
        - Vorhandene Spieler aus der Datenbank finden
        - Falls nicht vorhanden: Spieler automatisch anlegen
        - Den kanonischen Namen aus der Datenbank übernehmen
        - "last_played" für alle aktiven Spieler aktualisieren

        Rückgabe:
            list[str]: Finale Liste der Spielernamen in Eingabereihenfolge
        """
        final_names = []

        for i in range(self.selected_player_count):
            name = self.selected_names[i].strip()
            if not name:
                # Leere Eingaben werden übersprungen
                continue

            # Spieler holen oder neu erzeugen
            player = self.db.get_or_create_player(name)
            if player:
                # Den Namen aus der DB übernehmen,
                # z. B. wenn dort eine standardisierte Schreibweise existiert
                final_name = player["name"]
                self.selected_names[i] = final_name
                final_names.append(final_name)

        # "Zuletzt gespielt"-Zeitstempel für alle verwendeten Spieler aktualisieren
        self.db.touch_players_last_played(final_names)

        return final_names

    def draw_button(self, text, x, y, w, h, m_pos, action,
                    color=(50, 70, 120), active=True, font_type="menu"):
        """
        Zeichnet einen Button und registriert ihn optional als klickbar.

        Parameter:
            text (str): Beschriftung des Buttons
            x, y (int): Position links oben
            w, h (int): Breite und Höhe
            m_pos (tuple): Aktuelle Mausposition
            action (str): Aktionsname für Klickverarbeitung
            color (tuple): Grundfarbe
            active (bool): Ob der Button aktiv/klickbar sein soll
            font_type (str): "menu", "bold" oder "kb"

        Besonderheiten:
        - Hover-Effekt bei aktiven Buttons
        - In self.buttons wird die Button-Action mit dem Rechteck gespeichert
        - action == "NONE" bedeutet: zeichnen, aber nicht klickbar
        """
        rect = pygame.Rect(x, y, w, h)
        curr_col = color

        # Hover-Effekt:
        # Wenn der Button aktiv ist, eine echte Action hat und die Maus darüber ist,
        # wird die Farbe leicht aufgehellt
        if active and action != "NONE" and rect.collidepoint(m_pos):
            curr_col = tuple(min(255, c + 35) for c in color)

        # Gefüllter Button
        pygame.draw.rect(self.screen, curr_col, rect, border_radius=12)

        # Weißer Rand
        pygame.draw.rect(self.screen, (255, 255, 255), rect, 2, border_radius=12)

        # Schriftart je nach Button-Typ wählen
        f = self.font_menu_bold if font_type == "bold" else (
            self.font_kb if font_type == "kb" else self.font_menu
        )

        # Text rendern
        txt_surf = f.render(
            text,
            True,
            (255, 255, 255) if active else (100, 100, 100)
        )

        # Text zentriert im Button platzieren
        self.screen.blit(
            txt_surf,
            (x + (w - txt_surf.get_width()) // 2,
             y + (h - txt_surf.get_height()) // 2)
        )

        # Nur aktive, echte Buttons werden im Klicksystem registriert
        if action != "NONE" and active:
            self.buttons[action] = rect

    def render_lobby(self, m_pos):
        """
        Zeichnet die Lobby / das Spiel-Setup.

        Inhalte:
        - Titel
        - Kamera-/Kalibrierungsstatus
        - Button zur Kalibrierung
        - Auswahl der Spieleranzahl
        - Virtuelle Tastatur
        - Eingabefelder für Spielernamen
        - Namensvorschläge aus der Datenbank
        - Weiter-Button zur Spielauswahl
        """
        # Titel anzeigen
        title = self.font_title.render("SPIEL SETUP", True, (0, 220, 255))
        self.screen.blit(title, (960 - title.get_width() // 2, 40))

        # --- KAMERA STATUS ANZEIGE (DATEIBASIERT) ---
        # Prüfen, wie viele Kameras kalibriert sind
        calibrated_cams = self.check_calibration_status()
        is_ready = (calibrated_cams == 3)

        # Text und Farbe je nach Bereitschaft setzen
        status_color = (0, 255, 100) if is_ready else (255, 50, 50)
        status_text = "SYSTEM BEREIT" if is_ready else f"KALIBRIERUNG: {calibrated_cams}/3 Kameras"

        status_surf = self.font_status.render(status_text, True, status_color)
        self.screen.blit(status_surf, (960 - status_surf.get_width() // 2, 160))

        # --- KALIBRIERUNGS-BUTTON ---
        # Immer anzeigen, damit jederzeit neu kalibriert werden kann
        self.draw_button(
            "Kameras Kalibrieren",
            1320, 800, 500, 100,
            m_pos,
            "START_CALIBRATION",
            color=(150, 50, 0)
        )

        # Spieleranzahl 1 bis 8 als Buttons darstellen
        for i in range(1, 9):
            x = 100 + ((i - 1) % 4) * 100
            y = 210 if i <= 4 else 310

            # Aktuell ausgewählte Spieleranzahl grün hervorheben
            col = (0, 180, 100) if self.selected_player_count == i else (50, 70, 120)

            self.draw_button(str(i), x, y, 85, 85, m_pos, f"SET_COUNT_{i}", color=col)

        # Bereich für virtuelle Tastatur-Hintergrund
        pygame.draw.rect(self.screen, (30, 40, 60), (950, 210, 870, 500), border_radius=20)

        # Alle Tastaturzeichen als Buttons zeichnen
        for i, char in enumerate(self.kb_chars):
            col, row = i % 9, i // 9
            self.draw_button(
                char,
                980 + col * 85,
                240 + row * 75,
                75,
                65,
                m_pos,
                f"KEY_{char}",
                color=(60, 65, 90),
                font_type="kb"
            )

        # Sondertasten der virtuellen Tastatur
        self.draw_button("<--", 1320, 540, 160, 65, m_pos, "KEY_BACK", color=(120, 40, 40))
        self.draw_button("ENTER", 1335, 615, 435, 65, m_pos, "KEY_ENTER", color=(0, 120, 0))

        # Namensfelder nur dann zeichnen, wenn mindestens 1 Spieler gewählt wurde
        if self.selected_player_count > 0:
            for i in range(self.selected_player_count):
                y_pos = 450 + i * 70
                box_rect = pygame.Rect(180, y_pos, 450, 55)

                # Aktives Eingabefeld farblich hervorheben
                is_active = (self.active_input_idx == i)
                pygame.draw.rect(
                    self.screen,
                    (0, 100, 200) if is_active else (30, 40, 60),
                    box_rect,
                    border_radius=10
                )

                # Bereits eingegebenen Namen anzeigen
                self.screen.blit(
                    self.font_menu.render(self.selected_names[i], True, (255, 255, 255)),
                    (195, y_pos + 5)
                )

                # Eingabefeld als klickbares Fokus-Element registrieren
                self.buttons[f"FOCUS_{i}"] = box_rect

            # Vorschläge für das aktuell aktive Namensfeld anzeigen
            if self.name_suggestions:
                base_x = 650
                base_y = 450 + self.active_input_idx * 70

                for i, suggestion in enumerate(self.name_suggestions[:3]):
                    rect = pygame.Rect(base_x, base_y + i * 45, 260, 36)
                    hovered = rect.collidepoint(m_pos)

                    # Hover-Effekt für Vorschlagsliste
                    color = (55, 85, 130) if hovered else (35, 50, 80)

                    pygame.draw.rect(self.screen, color, rect, border_radius=8)
                    pygame.draw.rect(self.screen, (180, 220, 255), rect, 1, border_radius=8)

                    text = self.font_suggestion.render(suggestion, True, (255, 255, 255))
                    self.screen.blit(text, (rect.x + 10, rect.y + 5))

                    # Vorschlag als klickbar registrieren
                    self.buttons[f"SUGGEST_{i}"] = rect

        # Weiter-Button ist nur aktiv, wenn:
        # - mindestens 1 Spieler gewählt wurde
        # - der erste Name nicht leer ist
        # - das Kamerasystem vollständig kalibriert ist
        ready = self.selected_player_count > 0 and self.selected_names[0].strip() != "" and is_ready

        self.draw_button(
            "WEITER ZUR AUSWAHL",
            1320, 920, 500, 100,
            m_pos,
            "GO_TO_SELECT",
            active=ready,
            color=(0, 150, 70)
        )

    def render_game_select(self, m_pos):
        """
        Zeichnet den Bildschirm zur Auswahl des Spielmodus.
        """
        title = self.font_title.render("SPIELAUSWAHL", True, (0, 220, 255))
        self.screen.blit(title, (960 - title.get_width() // 2, 100))

        # Auswahlbuttons für die beiden Spieltypen
        self.draw_button("X01 GAME", 760, 390, 400, 120, m_pos, "SELECT_X01", color=(0, 100, 200), font_type="bold")
        self.draw_button("CRICKET", 760, 560, 400, 120, m_pos, "SELECT_CRICKET", color=(0, 130, 110), font_type="bold")

        # Zurück in die Lobby
        self.draw_button("ZURÜCK", 100, 920, 400, 100, m_pos, "GO_TO_LOBBY", color=(100, 20, 20))

    def render_settings(self, m_pos):
        """
        Zeichnet den Einstellungsbildschirm.

        Je nach gewähltem Spielmodus werden unterschiedliche Optionen angezeigt:
        - X01: Startscore, In/Out-Regeln, Endlosmodus, Legs
        - Cricket: Normal / Cut Throat
        """
        title_text = "X01 EINSTELLUNGEN" if self.selected_game_mode == "X01" else "CRICKET EINSTELLUNGEN"
        title = self.font_title.render(title_text, True, (0, 220, 255))
        self.screen.blit(title, (960 - title.get_width() // 2, 50))

        if self.selected_game_mode == "X01":
            # Startscore ändern
            self.draw_button("<", 750, 250, 60, 75, m_pos, "DEC_SCORE")
            self.draw_button(f"SCORE: {self.config['start_score']}", 820, 250, 280, 75, m_pos, "NONE")
            self.draw_button(">", 1110, 250, 60, 75, m_pos, "INC_SCORE")

            # In/Out-Regeln umschalten
            self.draw_button(f"IN: {self.config['in_mode']}", 450, 400, 400, 80, m_pos, "TOGGLE_IN")
            self.draw_button(f"OUT: {self.config['out_mode']}", 1050, 400, 400, 80, m_pos, "TOGGLE_OUT")

            # Endlosmodus an/aus
            endlos_txt = "ENDLOS: AN" if self.config["endlos"] else "ENDLOS: AUS"
            self.draw_button(endlos_txt, 760, 550, 400, 80, m_pos, "TOGGLE_ENDLOS", color=(100, 80, 20))

            # Legs nur anzeigen, wenn nicht im Endlosmodus gespielt wird
            if not self.config["endlos"]:
                self.draw_button("<", 750, 650, 60, 75, m_pos, "DEC_LEGS")
                self.draw_button(f"LEGS: {self.config['legs_to_win']}", 820, 650, 280, 75, m_pos, "NONE")
                self.draw_button(">", 1110, 650, 60, 75, m_pos, "INC_LEGS")

        else:
            # Cricket-Modus umschalten
            mode_text = "MODUS: NORMAL" if not self.config["cut_throat"] else "MODUS: CUT THROAT"
            mode_color = (0, 130, 110) if not self.config["cut_throat"] else (130, 50, 50)

            self.draw_button(
                mode_text,
                700, 360, 520, 100,
                m_pos,
                "TOGGLE_CRICKET_MODE",
                color=mode_color,
                font_type="bold"
            )

            # Hinweistext für den Benutzer
            hint1 = self.font_menu.render("Wie bei X01 zuerst den Modus wählen.", True, (220, 220, 220))
            hint2 = self.font_menu.render("Danach unten rechts das Spiel starten.", True, (220, 220, 220))
            self.screen.blit(hint1, (700, 520))
            self.screen.blit(hint2, (700, 570))

        # Navigation
        self.draw_button("ZURÜCK", 100, 920, 400, 100, m_pos, "GO_TO_SELECT", color=(100, 20, 20))
        self.draw_button("SPIEL STARTEN", 1420, 920, 400, 100, m_pos, "START_GAME", color=(20, 140, 20))

    def handle_click(self, pos):
        """
        Verarbeitet Maus-Klicks auf registrierte Buttons.

        Ablauf:
        - Prüft alle im aktuellen Frame registrierten Buttons
        - Wenn die Klickposition in einem Button liegt, wird dessen Aktion ausgeführt
        - Die Logik für Zustandswechsel, Konfigurationsänderungen,
          Namenseingaben und Spielstart ist hier zentral gebündelt
        """
        for action, rect in list(self.buttons.items()):
            if rect.collidepoint(pos):

                # Spieleranzahl setzen
                if action.startswith("SET_COUNT_"):
                    self.selected_player_count = int(action.split("_")[-1])

                    # Falls der aktive Eingabefokus außerhalb der neuen Spieleranzahl liegt,
                    # auf das letzte gültige Feld zurücksetzen
                    if self.active_input_idx >= self.selected_player_count:
                        self.active_input_idx = max(0, self.selected_player_count - 1)

                    self.refresh_name_suggestions()

                # Fokus auf ein bestimmtes Namensfeld setzen
                elif action.startswith("FOCUS_"):
                    self.active_input_idx = int(action.split("_")[-1])
                    self.refresh_name_suggestions()

                # Einen Vorschlag aus der Datenbank übernehmen
                elif action.startswith("SUGGEST_"):
                    sug_idx = int(action.split("_")[-1])
                    if 0 <= sug_idx < len(self.name_suggestions):
                        self.selected_names[self.active_input_idx] = self.name_suggestions[sug_idx]
                        self.refresh_name_suggestions()

                # Weiter zur Spielauswahl
                elif action == "GO_TO_SELECT":
                    self.finalize_selected_players()
                    self.state = "GAME_SELECT"

                # Zurück in die Lobby
                elif action == "GO_TO_LOBBY":
                    self.state = "LOBBY"

                # Spieltyp X01 wählen
                elif action == "SELECT_X01":
                    self.selected_game_mode = "X01"
                    self.state = "SETTINGS"

                # Spieltyp Cricket wählen
                elif action == "SELECT_CRICKET":
                    self.selected_game_mode = "CRICKET"
                    self.state = "SETTINGS"

                # Startscore erhöhen
                elif action == "INC_SCORE":
                    self.config["start_score"] = min(901, self.config["start_score"] + 200)

                # Startscore verringern
                elif action == "DEC_SCORE":
                    self.config["start_score"] = max(101, self.config["start_score"] - 200)

                # In-Modus umschalten
                elif action == "TOGGLE_IN":
                    self.config["in_mode"] = "Double In" if self.config["in_mode"] == "Single In" else "Single In"

                # Out-Modus zyklisch durchschalten
                elif action == "TOGGLE_OUT":
                    modes = ["Single Out", "Double Out", "Master Out"]
                    idx = (modes.index(self.config["out_mode"]) + 1) % len(modes)
                    self.config["out_mode"] = modes[idx]

                # Endlosmodus an/aus
                elif action == "TOGGLE_ENDLOS":
                    self.config["endlos"] = not self.config["endlos"]

                # Legs erhöhen
                elif action == "INC_LEGS":
                    self.config["legs_to_win"] += 1

                # Legs verringern, aber nicht unter 1
                elif action == "DEC_LEGS":
                    self.config["legs_to_win"] = max(1, self.config["legs_to_win"] - 1)

                # Cricket-Modus umschalten
                elif action == "TOGGLE_CRICKET_MODE":
                    self.config["cut_throat"] = not self.config["cut_throat"]

                # Spiel starten
                elif action == "START_GAME":
                    # Spielernamen finalisieren und in DB sichern
                    names = self.finalize_selected_players()
                    self.config["player_count"] = len(names)

                    # Passende Spielinstanz erzeugen
                    if self.selected_game_mode == "CRICKET":
                        self.game_instance = CricketGame(self.screen, self.config, player_names=names)
                    else:
                        self.game_instance = X01Game(self.screen, self.config, player_names=names)

                    # In den Spielzustand wechseln
                    self.state = "GAME"

                # --- KALIBRIERUNG DIREKT AUFRUFEN ---
                elif action == "START_CALIBRATION":
                    print("[SYSTEM] Kalibrierung gestartet...")

                    # 1. Vision-Thread stoppen, um Kamerakonflikte zu vermeiden
                    if self.vision_system:
                        self.vision_system.stop()
                        pygame.time.wait(1000)

                    # 2. Externe Kalibrierung ausführen
                    calibrate.start_calibration()

                    # 3. Vision-System neu starten
                    self.start_vision_thread()
                    print("[SYSTEM] Kalibrierung beendet, Status aktualisiert.")

                # Virtuelle Tastatur
                elif action.startswith("KEY_"):
                    char = action.replace("KEY_", "")
                    idx = self.active_input_idx

                    if char == "BACK":
                        # Letztes Zeichen löschen
                        self.selected_names[idx] = self.selected_names[idx][:-1]
                        self.refresh_name_suggestions()

                    elif char == "ENTER":
                        # Namen finalisieren / normalisieren
                        current_name = self.selected_names[idx].strip()
                        if current_name:
                            player = self.db.get_or_create_player(current_name)
                            if player:
                                self.selected_names[idx] = player["name"]

                        # Zum nächsten Eingabefeld springen, wenn vorhanden
                        if self.active_input_idx < self.selected_player_count - 1:
                            self.active_input_idx += 1

                        self.refresh_name_suggestions()

                    else:
                        # Normales Zeichen an den Namen anhängen
                        self.selected_names[idx] += char
                        self.refresh_name_suggestions()

                # Sobald ein passender Button verarbeitet wurde, Funktion verlassen
                return

    def run(self):
        """
        Hauptschleife des Programms.

        Diese Schleife läuft dauerhaft und übernimmt in jedem Frame drei Hauptaufgaben:

        1. Kamera-/Vision-Ereignisse aus der Queue lesen und ans Spiel weitergeben
        2. pygame-Events (Maus, Tastatur, Fenster schließen) verarbeiten
        3. Den aktuellen Bildschirmzustand neu zeichnen

        Die Schleife läuft mit maximal 60 FPS.
        """
        while True:
            # Aktuelle Mausposition für Hover-Effekte und Klicktests
            m_pos = pygame.mouse.get_pos()

            # --- 1. KAMERA INPUTS VERARBEITEN ---
            if self.vision_system:
                while not self.hit_queue.empty():
                    item = self.hit_queue.get()

                    # Sonderkommandos als String
                    if isinstance(item, str):
                        if item == "NEXT_PLAYER" and self.state == "GAME" and self.game_instance:
                            print("[MAIN] Pfeile gezogen -> Wurfzähler Reset!")
                            self.game_instance.reset_current_throw()

                    # Trefferdaten als Dictionary
                    elif isinstance(item, dict) and self.state == "GAME" and self.game_instance:
                        # Sonderfall: Pfeil wurde als "missed" erkannt
                        if item.get("is_missed", False):
                            print("[MAIN] Pfeil im Randbereich (Missed) -> 0 Punkte")
                            self.game_instance.handle_throw(0, 1)
                        else:
                            # Standardtreffer auslesen
                            sector = item.get("sector", 0)
                            ring = item.get("ring", "single")

                            # Ringname in Multiplikator umwandeln
                            mult_map = {
                                "single": 1,
                                "double": 2,
                                "triple": 3,
                                "single_bull": 1,
                                "bull": 2
                            }
                            mult = mult_map.get(ring, 1)

                            # Bull / Single Bull wird über Sector 25 behandelt
                            if ring in ["single_bull", "bull"]:
                                sector = 25

                            print(f"[MAIN] Treffer im Board: Sector {sector} Ring {ring}")
                            self.game_instance.handle_throw(sector, mult)

            # --- 2. EVENTS ---
            for ev in pygame.event.get():

                # Fenster wurde geschlossen
                if ev.type == pygame.QUIT:
                    if self.vision_system:
                        self.vision_system.stop()
                    pygame.quit()
                    sys.exit()

                # Mausklick
                if ev.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(m_pos)

                # Tastatureingabe nur während eines laufenden Spiels relevant
                if ev.type == pygame.KEYDOWN and self.state == "GAME":
                    if ev.key == pygame.K_BACKSPACE:
                        # Sonderfall:
                        # Wenn das Spiel gerade auf Bestätigung für Remove wartet,
                        # dann bestätigen. Sonst den letzten Wurf rückgängig machen.
                        if self.game_instance.waiting_for_remove:
                            self.game_instance.confirm_remove()
                        else:
                            self.game_instance.undo_last_throw()

            # --- 3. ZEICHNEN ---
            # Hintergrund löschen
            self.screen.fill((15, 20, 30))

            # Button-Registry für den neuen Frame zurücksetzen
            self.buttons = {}

            # Je nach aktuellem Zustand den passenden Bildschirm zeichnen
            if self.state == "LOBBY":
                self.render_lobby(m_pos)
            elif self.state == "GAME_SELECT":
                self.render_game_select(m_pos)
            elif self.state == "SETTINGS":
                self.render_settings(m_pos)
            elif self.state == "GAME":
                self.game_instance.draw()

            # Alles auf dem Bildschirm anzeigen
            pygame.display.flip()

            # Framerate auf 60 FPS begrenzen
            self.clock.tick(60)


if __name__ == "__main__":
    """
    Programmeinstieg:
    - Erstellt den MainManager
    - Startet die Hauptschleife
    """
    MainManager().run()
