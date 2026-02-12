import pygame
import sys
import os
import threading
from queue import Queue
import json
import calibrate # <--- Importiert die calibrate.py

# --- EXE-PFAF-KORREKTUR (VERBESSERT) ---
def resource_path(relative_path):
    """Ermittelt den Pfad zu Dateien, auch wenn das Skript als .exe läuft."""
    if hasattr(sys, '_MEIPASS'):
        # Wenn PyInstaller Bundle (Kompiliert)
        base_path = sys._MEIPASS
    else:
        # Wenn normales Skript (Python Interpret)
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    return os.path.join(base_path, relative_path)

# Erlaubt das Importieren aus Unterordnern
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from games.x01 import X01Game
from throw import ThrowSimulator
from database.database import DatabaseManager
from vision import DartVisionSystem 

class MainManager:
    def __init__(self):
        pygame.init()
        # Full HD Auflösung
        self.screen = pygame.display.set_mode((1920, 1080))
        # --- Update auf Version 1.9.5 (Missed Feld Fix) ---
        pygame.display.set_caption("ProjectDart Pro - Version 1.9.5 (Missed Feld Fix)")
        self.clock = pygame.time.Clock()
        
        # --- FONT INITIALISIERUNG ---
        self.font_title = pygame.font.SysFont("Segoe UI", 100, bold=True)
        self.font_menu = pygame.font.SysFont("Segoe UI", 36)
        self.font_menu_bold = pygame.font.SysFont("Segoe UI", 42, bold=True)
        self.font_kb = pygame.font.SysFont("Segoe UI", 30, bold=True)
        self.font_status = pygame.font.SysFont("Segoe UI", 24, italic=True)

        self.db = DatabaseManager()
        self.throw_manager = ThrowSimulator()
        
        # Kommunikation zwischen Kamera-Thread und Spiel
        self.hit_queue = Queue()
        
        # --- VISION INITIALISIERUNG ---
        self.vision_system = None
        self.start_vision_thread()
        
        self.state = "LOBBY" 
        self.game_instance = None
        
        # Setup Variablen
        self.selected_player_count = 0
        self.selected_names = [""] * 8
        self.active_input_idx = 0
        self.kb_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        # Spiel Konfiguration
        self.config = {
            "start_score": 501, 
            "in_mode": "Single In", 
            "out_mode": "Double Out",
            "player_count": 0, 
            "endlos": False, 
            "legs_to_win": 3, 
            "sets_to_win": 1
        }
        self.buttons = {}

    def start_vision_thread(self):
        """Initialisiert und startet das Vision System"""
        try:
            self.vision_system = DartVisionSystem(hit_callback=self.hit_queue.put)
            self.vision_thread = threading.Thread(target=self.vision_system.run, daemon=True)
            self.vision_thread.start()
            print("[INFO] Kamera-Thread erfolgreich gestartet.")
        except Exception as e:
            print(f"[FEHLER] Vision System konnte nicht geladen werden: {e}")

    def check_calibration_status(self):
        """Prüft anhand der Dateien, ob alle Kameras kalibriert sind."""
        calibrated_cams = 0
        
        for i in range(3):
            # Pfad zur Config Datei absolut sicher ermitteln
            config_file = resource_path(f"cam{i}_config.json")
            
            if os.path.exists(config_file):
                try:
                    with open(config_file, "r") as f:
                        data = json.load(f)
                        # Struktur prüfen: {"points": [[x,y], ...]}
                        if "points" in data and isinstance(data["points"], list) and len(data["points"]) == 4:
                            calibrated_cams += 1
                except Exception:
                    pass # Fehler beim Lesen ignorieren
        
        return calibrated_cams

    def draw_button(self, text, x, y, w, h, m_pos, action, color=(50, 70, 120), active=True, font_type="menu"):
        rect = pygame.Rect(x, y, w, h)
        curr_col = color
        # Hover Effekt
        if active and action != "NONE" and rect.collidepoint(m_pos):
            curr_col = tuple(min(255, c + 35) for c in color)
        
        pygame.draw.rect(self.screen, curr_col, rect, border_radius=12)
        pygame.draw.rect(self.screen, (255, 255, 255), rect, 2, border_radius=12)
        
        f = self.font_menu_bold if font_type == "bold" else (self.font_kb if font_type == "kb" else self.font_menu)
        txt_surf = f.render(text, True, (255, 255, 255) if active else (100, 100, 100))
        self.screen.blit(txt_surf, (x + (w - txt_surf.get_width()) // 2, y + (h - txt_surf.get_height()) // 2))
        
        if action != "NONE" and active:
            self.buttons[action] = rect

    def render_lobby(self, m_pos):
        # Titel
        title = self.font_title.render("SPIEL SETUP", True, (0, 220, 255))
        self.screen.blit(title, (960 - title.get_width() // 2, 40))
        
        # --- KAMERA STATUS ANZEIGE (DATEIBASIERT) ---
        calibrated_cams = self.check_calibration_status()
        
        # Hart auf True setzen, falls 3 gefunden wurden
        is_ready = (calibrated_cams == 3)
        
        status_color = (0, 255, 100) if is_ready else (255, 50, 50)
        status_text = "SYSTEM BEREIT" if is_ready else f"KALIBRIERUNG: {calibrated_cams}/3 Kameras"
        
        status_surf = self.font_status.render(status_text, True, status_color)
        self.screen.blit(status_surf, (960 - status_surf.get_width() // 2, 160))

        # --- KALIBRIERUNGS-BUTTON ---
        if not is_ready:
            self.draw_button("Kameras Kalibrieren", 1320, 800, 500, 100, m_pos, "START_CALIBRATION", color=(150, 50, 0))

        # Spieleranzahl
        for i in range(1, 9):
            x = 100 + ((i-1)%4)*100
            y = 210 if i <= 4 else 310
            col = (0, 180, 100) if self.selected_player_count == i else (50, 70, 120)
            self.draw_button(str(i), x, y, 85, 85, m_pos, f"SET_COUNT_{i}", color=col)

        # Virtuelle Tastatur
        pygame.draw.rect(self.screen, (30, 40, 60), (950, 210, 870, 500), border_radius=20)
        for i, char in enumerate(self.kb_chars):
            col, row = i % 9, i // 9
            self.draw_button(char, 980 + col*85, 240 + row*75, 75, 65, m_pos, f"KEY_{char}", color=(60, 65, 90), font_type="kb")
        self.draw_button("<--", 1320, 540, 160, 65, m_pos, "KEY_BACK", color=(120, 40, 40))
        self.draw_button("ENTER", 1335, 615, 435, 65, m_pos, "KEY_ENTER", color=(0, 120, 0))

        # Namensfelder
        if self.selected_player_count > 0:
            for i in range(self.selected_player_count):
                y_pos = 450 + i * 70
                box_rect = pygame.Rect(180, y_pos, 450, 55)
                is_active = (self.active_input_idx == i)
                pygame.draw.rect(self.screen, (0, 100, 200) if is_active else (30, 40, 60), box_rect, border_radius=10)
                self.screen.blit(self.font_menu.render(self.selected_names[i], True, (255, 255, 255)), (195, y_pos + 5))
                self.buttons[f"FOCUS_{i}"] = box_rect

        # Weiter-Button
        ready = self.selected_player_count > 0 and self.selected_names[0].strip() != "" and is_ready
        self.draw_button("WEITER ZUR AUSWAHL", 1320, 920, 500, 100, m_pos, "GO_TO_SELECT", active=ready, color=(0, 150, 70))

    def render_game_select(self, m_pos):
        title = self.font_title.render("SPIELAUSWAHL", True, (0, 220, 255))
        self.screen.blit(title, (960 - title.get_width() // 2, 100))
        self.draw_button("X01 GAME", 760, 450, 400, 120, m_pos, "SELECT_X01", color=(0, 100, 200), font_type="bold")
        self.draw_button("ZURÜCK", 100, 920, 400, 100, m_pos, "GO_TO_LOBBY", color=(100, 20, 20))

    def render_settings(self, m_pos):
        title = self.font_title.render("X01 EINSTELLUNGEN", True, (0, 220, 255))
        self.screen.blit(title, (960 - title.get_width() // 2, 50))
        
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

        self.draw_button("ZURÜCK", 100, 920, 400, 100, m_pos, "GO_TO_SELECT", color=(100, 20, 20))
        self.draw_button("SPIEL STARTEN", 1420, 920, 400, 100, m_pos, "START_GAME", color=(20, 140, 20))

    def handle_click(self, pos):
        for action, rect in list(self.buttons.items()):
            if rect.collidepoint(pos):
                if action.startswith("SET_COUNT_"): self.selected_player_count = int(action.split("_")[-1])
                elif action.startswith("FOCUS_"): self.active_input_idx = int(action.split("_")[-1])
                elif action == "GO_TO_SELECT": self.state = "GAME_SELECT"
                elif action == "GO_TO_LOBBY": self.state = "LOBBY"
                elif action == "SELECT_X01": self.state = "SETTINGS"
                elif action == "INC_SCORE": self.config["start_score"] = min(901, self.config["start_score"] + 200)
                elif action == "DEC_SCORE": self.config["start_score"] = max(101, self.config["start_score"] - 200)
                elif action == "TOGGLE_IN": self.config["in_mode"] = "Double In" if self.config["in_mode"] == "Single In" else "Single In"
                elif action == "TOGGLE_OUT":
                    modes = ["Single Out", "Double Out", "Master Out"]
                    idx = (modes.index(self.config["out_mode"]) + 1) % len(modes)
                    self.config["out_mode"] = modes[idx]
                elif action == "TOGGLE_ENDLOS": self.config["endlos"] = not self.config["endlos"]
                elif action == "INC_LEGS": self.config["legs_to_win"] += 1
                elif action == "DEC_LEGS": self.config["legs_to_win"] = max(1, self.config["legs_to_win"] - 1)
                elif action == "START_GAME":
                    names = [self.selected_names[i] for i in range(self.selected_player_count)]
                    self.config["player_count"] = self.selected_player_count
                    self.game_instance = X01Game(self.screen, self.config, player_names=names)
                    self.state = "GAME"
                
                # --- KALIBRIERUNG DIREKT AUFRUFEN ---
                elif action == "START_CALIBRATION":
                    print("[SYSTEM] Kalibrierung gestartet...")
                    
                    # 1. Thread stoppen um Kamera-Konflikte zu vermeiden
                    if self.vision_system:
                        self.vision_system.stop()
                        pygame.time.wait(1000) 
                    
                    # 2. Externe Kalibrierung ausführen
                    calibrate.start_calibration() 
                    
                    # 3. Kameras für Vision neu initialisieren
                    self.start_vision_thread()
                    print("[SYSTEM] Kalibrierung beendet, Status aktualisiert.")

                elif action.startswith("KEY_"):
                    char = action.replace("KEY_", "")
                    idx = self.active_input_idx
                    if char == "BACK": self.selected_names[idx] = self.selected_names[idx][:-1]
                    elif char == "ENTER": 
                        if self.selected_names[idx].strip(): self.db.add_player(self.selected_names[idx].strip())
                        if self.active_input_idx < self.selected_player_count - 1: self.active_input_idx += 1
                    else: self.selected_names[idx] += char

    def run(self):
        while True:
            m_pos = pygame.mouse.get_pos()
            
            # --- 1. KAMERA INPUTS VERARBEITEN ---
            if self.vision_system:
                while not self.hit_queue.empty():
                    item = self.hit_queue.get()
                    
                    if isinstance(item, str):
                        if item == "NEXT_PLAYER" and self.state == "GAME" and self.game_instance:
                            print("[MAIN] Pfeile gezogen -> Wurfzähler Reset!")
                            self.game_instance.reset_current_throw() 
                    
                    # 🚀 GEÄNDERT: Verarbeitung des Dicts aus vision_absdiff.py
                    elif isinstance(item, dict) and self.state == "GAME" and self.game_instance:
                        # item Struktur: {"sector": 1-20, "is_missed": True/False, ...}
                        
                        if item.get("is_missed", False):
                            print(f"[MAIN] Pfeil im Randbereich (Missed) -> 0 Punkte")
                            # Registriere einen Wurf mit 0 Punkten (Single Ring Multiplier)
                            self.game_instance.handle_throw(0, 1) 
                        else:
                            # Normaler Treffer im Board
                            sector = item.get("sector", 0)
                            print(f"[MAIN] Treffer im Board: Sector {sector}")
                            # Sektor an Spiel übergeben. Der Multiplier wird im Game bestimmt.
                            self.game_instance.handle_throw(sector, 1) 

            # --- 2. EVENTS ---
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT: 
                    if self.vision_system: self.vision_system.stop()
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
                    else:
                        res = self.throw_manager.handle_input(ev)
                        if res and res != "RESET": 
                            self.game_instance.handle_throw(res[0], res[1])

            # --- 3. ZEICHNEN ---
            self.screen.fill((15, 20, 30))
            self.buttons = {} # Buttons pro Frame neu registrieren
            
            if self.state == "LOBBY": 
                self.render_lobby(m_pos)
            elif self.state == "GAME_SELECT": 
                self.render_game_select(m_pos)
            elif self.state == "SETTINGS": 
                self.render_settings(m_pos)
            elif self.state == "GAME": 
                self.game_instance.draw()
            
            pygame.display.flip()
            self.clock.tick(60)

if __name__ == "__main__":
    MainManager().run()