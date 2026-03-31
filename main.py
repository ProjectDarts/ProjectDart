import json
import os
import sys
import threading
from queue import Queue

import pygame

import calibrate  # Externe Kalibrierungslogik
from database.database import DatabaseManager
from games.cricket import CricketGame
from games.x01 import X01Game
from vision import DartVisionSystem


# =========================================================
# PFAD-LOGIK
# =========================================================
def resource_path(relative_path: str) -> str:
    """
    Liefert den absoluten Pfad zu einer Projektdatei zurück.

    Zielsystem:
    - Raspberry Pi 5
    - direkte Python-Ausführung
    - keine EXE / kein PyInstaller mehr

    Daher werden Laufzeitdateien wie Kalibrierungen, DBs und Configs
    immer relativ zum Verzeichnis von main.py gespeichert/gelesen.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


# Erlaubt Imports relativ zu main.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class MainManager:
    """
    Zentrale Steuerklasse der Anwendung.

    Aufgaben:
    - pygame initialisieren
    - UI-Zustände verwalten
    - Vision-System starten/stoppen
    - Treffer aus dem Vision-System verarbeiten
    - Menüs und Spiele rendern
    - Kalibrierung starten
    """

    def __init__(self):
        pygame.init()

        # Fenster
        self.screen = pygame.display.set_mode((1920, 1080))
        pygame.display.set_caption("ProjectDart Pro - Raspberry Pi Edition")

        self.clock = pygame.time.Clock()

        # Fonts
        self.font_title = pygame.font.SysFont("Segoe UI", 100, bold=True)
        self.font_menu = pygame.font.SysFont("Segoe UI", 36)
        self.font_menu_bold = pygame.font.SysFont("Segoe UI", 42, bold=True)
        self.font_kb = pygame.font.SysFont("Segoe UI", 30, bold=True)
        self.font_status = pygame.font.SysFont("Segoe UI", 24, italic=True)
        self.font_suggestion = pygame.font.SysFont("Segoe UI", 24)

        # Datenbank
        self.db = DatabaseManager()

        # Kommunikation mit Vision-Thread
        self.hit_queue = Queue()

        # Vision
        self.vision_system = None
        self.vision_thread = None
        self.start_vision_thread()

        # UI-State
        self.state = "LOBBY"
        self.game_instance = None
        self.selected_game_mode = "X01"

        # Lobby / Spieler-Setup
        self.selected_player_count = 0
        self.selected_names = [""] * 8
        self.active_input_idx = 0
        self.kb_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        self.name_suggestions = []

        # Spiel-Konfiguration
        self.config = {
            "start_score": 501,
            "in_mode": "Single In",
            "out_mode": "Double Out",
            "player_count": 0,
            "endlos": False,
            "legs_to_win": 3,
            "sets_to_win": 1,
            "cut_throat": False,
        }

        # Klickbare Buttons pro Frame
        self.buttons = {}

    # =====================================================
    # VISION
    # =====================================================
    def start_vision_thread(self):
        """
        Initialisiert und startet das Vision-System in einem eigenen Thread.
        """
        # Falls bereits ein Thread/System aktiv ist, vorher sauber stoppen
        self.stop_vision_thread()

        try:
            self.vision_system = DartVisionSystem(hit_callback=self.hit_queue.put)

            self.vision_thread = threading.Thread(
                target=self.vision_system.run,
                daemon=True
            )
            self.vision_thread.start()

            print("[INFO] Kamera-Thread erfolgreich gestartet.")

        except Exception as e:
            self.vision_system = None
            self.vision_thread = None
            print(f"[FEHLER] Vision System konnte nicht geladen werden: {e}")

    def stop_vision_thread(self):
        """
        Stoppt das Vision-System sauber, um Kamerakonflikte zu vermeiden.
        """
        if self.vision_system is not None:
            try:
                self.vision_system.stop()
                print("[INFO] Vision-System gestoppt.")
            except Exception as e:
                print(f"[WARN] Fehler beim Stoppen des Vision-Systems: {e}")

        if self.vision_thread is not None and self.vision_thread.is_alive():
            self.vision_thread.join(timeout=2.0)
            if self.vision_thread.is_alive():
                print("[WARN] Vision-Thread läuft nach Stop noch weiter.")

        self.vision_system = None
        self.vision_thread = None

        # Queue leeren, damit keine alten Treffer später ins Spiel laufen
        while not self.hit_queue.empty():
            try:
                self.hit_queue.get_nowait()
            except Exception:
                break

    def restart_vision_thread(self):
        """
        Stoppt und startet das Vision-System neu.
        """
        self.stop_vision_thread()
        self.start_vision_thread()

    # =====================================================
    # KALIBRIERUNGSSTATUS
    # =====================================================
    def check_calibration_status(self) -> int:
        """
        Prüft, wie viele Kameras korrekt kalibriert sind.

        Erwartet:
        - cam0_config.json
        - cam1_config.json
        - cam2_config.json

        Eine Kamera gilt als kalibriert, wenn:
        - Datei existiert
        - "points" enthalten ist
        - genau 4 Punkte enthalten sind
        """
        calibrated_cams = 0

        for i in range(3):
            config_file = resource_path(f"cam{i}_config.json")

            if not os.path.exists(config_file):
                continue

            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                points = data.get("points", None)
                if isinstance(points, list) and len(points) == 4:
                    calibrated_cams += 1

            except Exception:
                pass

        return calibrated_cams

    def run_calibration(self):
        """
        Startet die externe Kalibrierung.

        Ablauf:
        1. Vision-System stoppen
        2. Kalibrierung ausführen
        3. Vision-System neu starten
        """
        print("[SYSTEM] Kalibrierung gestartet...")

        self.stop_vision_thread()

        try:
            calibrate.start_calibration()
        except Exception as e:
            print(f"[FEHLER] Kalibrierung fehlgeschlagen: {e}")
        finally:
            self.start_vision_thread()

        print("[SYSTEM] Kalibrierung beendet, Status aktualisiert.")

    # =====================================================
    # SPIELER / NAMEN
    # =====================================================
    def refresh_name_suggestions(self):
        if self.selected_player_count <= 0:
            self.name_suggestions = []
            return

        current = self.selected_names[self.active_input_idx].strip()
        if not current:
            self.name_suggestions = []
            return

        self.name_suggestions = self.db.search_players(current, limit=3)

    def finalize_selected_players(self):
        final_names = []

        for i in range(self.selected_player_count):
            name = self.selected_names[i].strip()
            if not name:
                continue

            player = self.db.get_or_create_player(name)
            if player:
                final_name = player["name"]
                self.selected_names[i] = final_name
                final_names.append(final_name)

        self.db.touch_players_last_played(final_names)
        return final_names

    # =====================================================
    # UI
    # =====================================================
    def draw_button(self, text, x, y, w, h, m_pos, action,
                    color=(50, 70, 120), active=True, font_type="menu"):
        rect = pygame.Rect(x, y, w, h)
        curr_col = color

        if active and action != "NONE" and rect.collidepoint(m_pos):
            curr_col = tuple(min(255, c + 35) for c in color)

        pygame.draw.rect(self.screen, curr_col, rect, border_radius=12)
        pygame.draw.rect(self.screen, (255, 255, 255), rect, 2, border_radius=12)

        f = self.font_menu_bold if font_type == "bold" else (
            self.font_kb if font_type == "kb" else self.font_menu
        )

        txt_surf = f.render(
            text,
            True,
            (255, 255, 255) if active else (100, 100, 100)
        )

        self.screen.blit(
            txt_surf,
            (x + (w - txt_surf.get_width()) // 2,
             y + (h - txt_surf.get_height()) // 2)
        )

        if action != "NONE" and active:
            self.buttons[action] = rect

    def render_lobby(self, m_pos):
        title = self.font_title.render("SPIEL SETUP", True, (0, 220, 255))
        self.screen.blit(title, (960 - title.get_width() // 2, 40))

        calibrated_cams = self.check_calibration_status()
        is_ready = (calibrated_cams == 3)

        status_color = (0, 255, 100) if is_ready else (255, 50, 50)
        status_text = "SYSTEM BEREIT" if is_ready else f"KALIBRIERUNG: {calibrated_cams}/3 Kameras"

        status_surf = self.font_status.render(status_text, True, status_color)
        self.screen.blit(status_surf, (960 - status_surf.get_width() // 2, 160))

        self.draw_button(
            "Kameras Kalibrieren",
            1320, 800, 500, 100,
            m_pos,
            "START_CALIBRATION",
            color=(150, 50, 0)
        )

        for i in range(1, 9):
            x = 100 + ((i - 1) % 4) * 100
            y = 210 if i <= 4 else 310
            col = (0, 180, 100) if self.selected_player_count == i else (50, 70, 120)
            self.draw_button(str(i), x, y, 85, 85, m_pos, f"SET_COUNT_{i}", color=col)

        pygame.draw.rect(self.screen, (30, 40, 60), (950, 210, 870, 500), border_radius=20)

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

        self.draw_button("<--", 1320, 540, 160, 65, m_pos, "KEY_BACK", color=(120, 40, 40))
        self.draw_button("ENTER", 1335, 615, 435, 65, m_pos, "KEY_ENTER", color=(0, 120, 0))

        if self.selected_player_count > 0:
            for i in range(self.selected_player_count):
                y_pos = 450 + i * 70
                box_rect = pygame.Rect(180, y_pos, 450, 55)

                is_active = (self.active_input_idx == i)
                pygame.draw.rect(
                    self.screen,
                    (0, 100, 200) if is_active else (30, 40, 60),
                    box_rect,
                    border_radius=10
                )

                self.screen.blit(
                    self.font_menu.render(self.selected_names[i], True, (255, 255, 255)),
                    (195, y_pos + 5)
                )

                self.buttons[f"FOCUS_{i}"] = box_rect

            if self.name_suggestions:
                base_x = 650
                base_y = 450 + self.active_input_idx * 70

                for i, suggestion in enumerate(self.name_suggestions[:3]):
                    rect = pygame.Rect(base_x, base_y + i * 45, 260, 36)
                    hovered = rect.collidepoint(m_pos)
                    color = (55, 85, 130) if hovered else (35, 50, 80)

                    pygame.draw.rect(self.screen, color, rect, border_radius=8)
                    pygame.draw.rect(self.screen, (180, 220, 255), rect, 1, border_radius=8)

                    text = self.font_suggestion.render(suggestion, True, (255, 255, 255))
                    self.screen.blit(text, (rect.x + 10, rect.y + 5))

                    self.buttons[f"SUGGEST_{i}"] = rect

        ready = (
            self.selected_player_count > 0
            and self.selected_names[0].strip() != ""
            and is_ready
        )

        self.draw_button(
            "WEITER ZUR AUSWAHL",
            1320, 920, 500, 100,
            m_pos,
            "GO_TO_SELECT",
            active=ready,
            color=(0, 150, 70)
        )

    def render_game_select(self, m_pos):
        title = self.font_title.render("SPIELAUSWAHL", True, (0, 220, 255))
        self.screen.blit(title, (960 - title.get_width() // 2, 100))

        self.draw_button("X01 GAME", 760, 390, 400, 120, m_pos, "SELECT_X01", color=(0, 100, 200), font_type="bold")
        self.draw_button("CRICKET", 760, 560, 400, 120, m_pos, "SELECT_CRICKET", color=(0, 130, 110), font_type="bold")
        self.draw_button("ZURÜCK", 100, 920, 400, 100, m_pos, "GO_TO_LOBBY", color=(100, 20, 20))

    def render_settings(self, m_pos):
        title_text = "X01 EINSTELLUNGEN" if self.selected_game_mode == "X01" else "CRICKET EINSTELLUNGEN"
        title = self.font_title.render(title_text, True, (0, 220, 255))
        self.screen.blit(title, (960 - title.get_width() // 2, 50))

        if self.selected_game_mode == "X01":
            self.draw_button("<", 750, 250, 60, 75, m_pos, "DEC_SCORE")
            self.draw_button(f"SCORE: {self.config['start_score']}", 820, 250, 280, 75, m_pos, "NONE")
            self.draw_button(">", 1110, 250, 60, 75, m_pos, "INC_SCORE")

            self.draw_button(f"IN: {self.config['in_mode']}", 450, 400, 400, 80, m_pos, "TOGGLE_IN")
            self.draw_button(f"OUT: {self.config['out_mode']}", 1050, 400, 400, 80, m_pos, "TOGGLE_OUT")

            endlos_txt = "ENDLOS: AN" if self.config["endlos"] else "ENDLOS: AUS"
            self.draw_button(endlos_txt, 760, 550, 400, 80, m_pos, "TOGGLE_ENDLOS", color=(100, 80, 20))

            if not self.config["endlos"]:
                self.draw_button("<", 750, 650, 60, 75, m_pos, "DEC_LEGS")
                self.draw_button(f"LEGS: {self.config['legs_to_win']}", 820, 650, 280, 75, m_pos, "NONE")
                self.draw_button(">", 1110, 650, 60, 75, m_pos, "INC_LEGS")

        else:
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

            hint1 = self.font_menu.render("Wie bei X01 zuerst den Modus wählen.", True, (220, 220, 220))
            hint2 = self.font_menu.render("Danach unten rechts das Spiel starten.", True, (220, 220, 220))
            self.screen.blit(hint1, (700, 520))
            self.screen.blit(hint2, (700, 570))

        self.draw_button("ZURÜCK", 100, 920, 400, 100, m_pos, "GO_TO_SELECT", color=(100, 20, 20))
        self.draw_button("SPIEL STARTEN", 1420, 920, 400, 100, m_pos, "START_GAME", color=(20, 140, 20))

    # =====================================================
    # KLICKLOGIK
    # =====================================================
    def handle_click(self, pos):
        for action, rect in list(self.buttons.items()):
            if not rect.collidepoint(pos):
                continue

            if action.startswith("SET_COUNT_"):
                self.selected_player_count = int(action.split("_")[-1])

                if self.active_input_idx >= self.selected_player_count:
                    self.active_input_idx = max(0, self.selected_player_count - 1)

                self.refresh_name_suggestions()

            elif action.startswith("FOCUS_"):
                self.active_input_idx = int(action.split("_")[-1])
                self.refresh_name_suggestions()

            elif action.startswith("SUGGEST_"):
                sug_idx = int(action.split("_")[-1])
                if 0 <= sug_idx < len(self.name_suggestions):
                    self.selected_names[self.active_input_idx] = self.name_suggestions[sug_idx]
                    self.refresh_name_suggestions()

            elif action == "GO_TO_SELECT":
                self.finalize_selected_players()
                self.state = "GAME_SELECT"

            elif action == "GO_TO_LOBBY":
                self.state = "LOBBY"

            elif action == "SELECT_X01":
                self.selected_game_mode = "X01"
                self.state = "SETTINGS"

            elif action == "SELECT_CRICKET":
                self.selected_game_mode = "CRICKET"
                self.state = "SETTINGS"

            elif action == "INC_SCORE":
                self.config["start_score"] = min(901, self.config["start_score"] + 200)

            elif action == "DEC_SCORE":
                self.config["start_score"] = max(101, self.config["start_score"] - 200)

            elif action == "TOGGLE_IN":
                self.config["in_mode"] = "Double In" if self.config["in_mode"] == "Single In" else "Single In"

            elif action == "TOGGLE_OUT":
                modes = ["Single Out", "Double Out", "Master Out"]
                idx = (modes.index(self.config["out_mode"]) + 1) % len(modes)
                self.config["out_mode"] = modes[idx]

            elif action == "TOGGLE_ENDLOS":
                self.config["endlos"] = not self.config["endlos"]

            elif action == "INC_LEGS":
                self.config["legs_to_win"] += 1

            elif action == "DEC_LEGS":
                self.config["legs_to_win"] = max(1, self.config["legs_to_win"] - 1)

            elif action == "TOGGLE_CRICKET_MODE":
                self.config["cut_throat"] = not self.config["cut_throat"]

            elif action == "START_GAME":
                names = self.finalize_selected_players()
                self.config["player_count"] = len(names)

                if self.selected_game_mode == "CRICKET":
                    self.game_instance = CricketGame(self.screen, self.config, player_names=names)
                else:
                    self.game_instance = X01Game(self.screen, self.config, player_names=names)

                self.state = "GAME"

            elif action == "START_CALIBRATION":
                self.run_calibration()

            elif action.startswith("KEY_"):
                char = action.replace("KEY_", "")
                idx = self.active_input_idx

                if char == "BACK":
                    self.selected_names[idx] = self.selected_names[idx][:-1]
                    self.refresh_name_suggestions()

                elif char == "ENTER":
                    current_name = self.selected_names[idx].strip()
                    if current_name:
                        player = self.db.get_or_create_player(current_name)
                        if player:
                            self.selected_names[idx] = player["name"]

                    if self.active_input_idx < self.selected_player_count - 1:
                        self.active_input_idx += 1

                    self.refresh_name_suggestions()

                else:
                    self.selected_names[idx] += char
                    self.refresh_name_suggestions()

            return

    # =====================================================
    # HAUPTSCHLEIFE
    # =====================================================
    def run(self):
        while True:
            m_pos = pygame.mouse.get_pos()

            # --- Kamera/Vision-Ereignisse ---
            if self.vision_system:
                while not self.hit_queue.empty():
                    item = self.hit_queue.get()

                    if isinstance(item, str):
                        if item == "NEXT_PLAYER" and self.state == "GAME" and self.game_instance:
                            print("[MAIN] Pfeile gezogen -> Wurfzähler Reset!")
                            self.game_instance.reset_current_throw()

                    elif isinstance(item, dict) and self.state == "GAME" and self.game_instance:
                        if item.get("is_missed", False):
                            print("[MAIN] Pfeil im Randbereich (Missed) -> 0 Punkte")
                            self.game_instance.handle_throw(0, 1)
                        else:
                            sector = item.get("sector", 0)
                            ring = item.get("ring", "single")

                            mult_map = {
                                "single": 1,
                                "double": 2,
                                "triple": 3,
                                "single_bull": 1,
                                "bull": 2
                            }
                            mult = mult_map.get(ring, 1)

                            if ring in ["single_bull", "bull"]:
                                sector = 25

                            print(f"[MAIN] Treffer im Board: Sector {sector} Ring {ring}")
                            self.game_instance.handle_throw(sector, mult)

            # --- pygame Events ---
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.stop_vision_thread()
                    pygame.quit()
                    sys.exit()

                if ev.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(m_pos)

                if ev.type == pygame.KEYDOWN and self.state == "GAME":
                    if ev.key == pygame.K_BACKSPACE:
                        if self.game_instance.waiting_for_remove:
                            self.game_instance.confirm_remove()
                        else:
                            self.game_instance.undo_last_throw()

            # --- Zeichnen ---
            self.screen.fill((15, 20, 30))
            self.buttons = {}

            if self.state == "LOBBY":
                self.render_lobby(m_pos)
            elif self.state == "GAME_SELECT":
                self.render_game_select(m_pos)
            elif self.state == "SETTINGS":
                self.render_settings(m_pos)
            elif self.state == "GAME" and self.game_instance:
                self.game_instance.draw()

            pygame.display.flip()
            self.clock.tick(60)


if __name__ == "__main__":
    MainManager().run()
