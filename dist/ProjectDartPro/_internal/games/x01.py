import pygame
import os
import sys

# --- EXE-Pfad-Korrektur für Ressourcen ---
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
# -----------------------------------------

# EXE-kompatibler Import der Checkouts
try:
    from games.d_checkouts import get_d_checkouts
except ImportError:
    try:
        from .d_checkouts import get_d_checkouts
    except ImportError:
        def get_d_checkouts(score): return [] # Fallback falls Datei fehlt

class X01Game:
    def __init__(self, screen, config, player_names=None):
        self.screen = screen
        self.config = config
        
        # Fonts - Größen-Anpassung
        self.font_big_score = pygame.font.SysFont("Impact", 220) 
        self.font_player_name = pygame.font.SysFont("Arial", 80, bold=True)
        self.font_info = pygame.font.SysFont("Arial", 45, bold=True)
        self.font_list = pygame.font.SysFont("Arial", 38, bold=True)
        
        self.font_co_title = pygame.font.SysFont("Arial", 70, bold=True) 
        self.font_co = pygame.font.SysFont("Arial", 63, bold=True)      
        
        self.font_msg = pygame.font.SysFont("Arial", 42, bold=True) # Angepasst für kleine Box
        self.font_bust = pygame.font.SysFont("Arial", 60, bold=True)
        
        self.players = []
        names = player_names if player_names else [f"P{i+1}" for i in range(config["player_count"])]

        for name in names:
            self.players.append({
                "name": name, 
                "score": config["start_score"],
                "active_in": (config["in_mode"] == "Single In"),
                "legs": 0, "sets": 0,
                "darts_total": 0, "points_total": 0,
                "first_9_points": 0, "first_9_darts": 0,
                "avg_170_sum": 0, "avg_170_count": 0,
                "reached_170_this_leg": False,
                "visit": [], "history": [] 
            })
            
        self.current_idx = 0
        self.waiting_for_remove = False
        self.is_bust = False

    def reset_current_throw(self):
        """Wird aufgerufen, wenn die Kamera erkannt hat, dass die Pfeile gezogen wurden"""
        if self.waiting_for_remove:
            self.confirm_remove()

    def handle_throw(self, val, mult):
        """ Verarbeitet einen Wurf von Kamera oder Tastatur """
        if self.waiting_for_remove: return
        
        try: 
            val, mult = int(val), int(mult)
        except (ValueError, TypeError): 
            return
            
        p = self.players[self.current_idx]
        pts = val * mult
        
        p["history"].append({
            "score_before": p["score"], 
            "pts": pts, 
            "active_in_before": p["active_in"], 
            "f9_valid": (p["darts_total"] < 9), 
            "was_170_check": p["reached_170_this_leg"]
        })
        
        if not p["active_in"]:
            if (self.config["in_mode"] == "Double In" and mult == 2) or self.config["in_mode"] == "Single In": 
                p["active_in"] = True
            else:
                p["visit"].append(0)
                p["darts_total"] += 1
                self.check_end_visit(p)
                return
        
        # 🚀 HIER: Verarbeitung von Missed Feldern (val=0)
        if val == 0:
            target = p["score"] # Keine Punkte
            bust = False
        else:
            target = p["score"] - pts
            bust = False
            
            if target < 0: 
                bust = True
            elif target == 1:
                if self.config["out_mode"] != "Single Out": bust = True
            elif target == 0:
                if (self.config["out_mode"] == "Double Out" and mult != 2): bust = True
                elif (self.config["out_mode"] == "Master Out" and mult < 2): bust = True
        
        if bust:
            self.is_bust = True
            p["visit"].append(pts)
            remaining_darts = 3 - len(p["visit"])
            p["darts_total"] += (remaining_darts + 1)
            self.waiting_for_remove = True
        else:
            p["score"] = target
            p["points_total"] += pts
            if p["darts_total"] < 9: 
                p["first_9_points"] += pts
                p["first_9_darts"] += 1
            p["darts_total"] += 1
            p["visit"].append(pts)
            
            if p["score"] <= 170 and not p["reached_170_this_leg"]:
                c_avg = (p["points_total"] / (p["darts_total"] / 3)) if p["darts_total"] > 0 else 0
                p["avg_170_sum"] += c_avg
                p["avg_170_count"] += 1
                p["reached_170_this_leg"] = True
            
            if p["score"] == 0: 
                self.process_leg_win(p)
            else: 
                self.check_end_visit(p)

    def undo_last_throw(self):
        p = self.players[self.current_idx]
        if not p["history"]: return
        last = p["history"].pop()
        self.is_bust = False
        
        if p["reached_170_this_leg"] and not last["was_170_check"]:
            if p["avg_170_count"] > 0: p["avg_170_count"] -= 1
            p["reached_170_this_leg"] = False
            
        p["score"] = last["score_before"]
        p["active_in"] = last["active_in_before"]
        if p["visit"]: p["visit"].pop()
        p["points_total"] -= last["pts"]
        if last["f9_valid"]: 
            p["first_9_points"] -= last["pts"]
            p["first_9_darts"] -= 1
        p["darts_total"] = max(0, p["darts_total"] - 1)
        self.waiting_for_remove = False

    def check_end_visit(self, p):
        if len(p["visit"]) == 3: 
            self.waiting_for_remove = True

    def confirm_remove(self):
        """Wird aufgerufen, um einen Wurf/Leg zu bestätigen, nachdem Pfeile gezogen wurden."""
        if self.waiting_for_remove:
            p = self.players[self.current_idx]
            
            # 1. Aufnahme des aktuellen Spielers zurücksetzen
            p["visit"] = []
            
            # 2. Zum nächsten Spieler wechseln.
            self.current_idx = (self.current_idx + 1) % len(self.players)
            
            # 3. Status zurücksetzen
            self.waiting_for_remove = False
            self.is_bust = False # Bust Status zurücksetzen

    def process_leg_win(self, winner):
        winner["legs"] += 1
        if not self.config["endlos"] and winner["legs"] >= self.config["legs_to_win"]:
            winner["sets"] += 1
            for pl in self.players: pl["legs"] = 0
            
        for pl in self.players:
            pl["score"] = self.config["start_score"]
            pl["active_in"] = (self.config["in_mode"] == "Single In")
            pl["visit"], pl["reached_170_this_leg"] = [], False
            pl["history"] = []
            
        self.waiting_for_remove = True

    def draw(self):
        self.screen.fill((10, 15, 25))
        p = self.players[self.current_idx]
        
        # Hauptfeld
        pygame.draw.rect(self.screen, (25, 40, 70), (50, 40, 1040, 620), border_radius=30)
        self.screen.blit(self.font_player_name.render(p["name"], True, (255, 255, 255)), (100, 60))
        score_surf = self.font_big_score.render(str(p["score"]), True, (255, 255, 0))
        self.screen.blit(score_surf, (100, 140))
        
        if self.is_bust:
            self.screen.blit(self.font_bust.render("BUST!", True, (255, 50, 50)), (100, 390))
        else:
            v_txt = "  ".join([str(x) for x in p["visit"]])
            self.screen.blit(self.font_info.render(f"Aufnahme: {v_txt}", True, (0, 255, 150)), (100, 390))
        
        # Stats
        avg_3 = (p["points_total"] / (p["darts_total"] / 3)) if p["darts_total"] > 0 else 0.0
        f9_avg = (p["first_9_points"] / (p["first_9_darts"] / 3)) if p["first_9_darts"] > 0 else 0.0
        avg_170 = (p["avg_170_sum"] / p["avg_170_count"]) if p["avg_170_count"] > 0 else 0.0
        
        sy, lh = 460, 50
        self.screen.blit(self.font_info.render(f"ø Gesamt: {avg_3:.2f}", True, (200, 200, 200)), (100, sy))
        self.screen.blit(self.font_info.render(f"ø First 9: {f9_avg:.2f}", True, (200, 200, 200)), (100, sy + lh))
        self.screen.blit(self.font_info.render(f"ø to 170: {avg_170:.2f}", True, (0, 200, 255)), (100, sy + lh * 2))
        self.screen.blit(self.font_info.render(f"Darts: {p['darts_total']}", True, (150, 150, 150)), (850, 590))
        
        # Spielerliste rechts
        pygame.draw.rect(self.screen, (20, 25, 40), (1140, 40, 730, 620), border_radius=20)
        for i, pl in enumerate(self.players):
            y = 70 + i * 75
            c = (255, 255, 255) if i == self.current_idx else (100, 100, 110)
            l_avg = (pl["points_total"] / (pl["darts_total"] / 3)) if pl["darts_total"] > 0 else 0.0
            suffix = f"L:{pl['legs']}" if self.config["endlos"] else f"L:{pl['legs']} S:{pl['sets']}"
            txt = f"{pl['name']}: {pl['score']} (ø {l_avg:.1f}) | {suffix}"
            self.screen.blit(self.font_list.render(txt, True, c), (1170, y))

        # Checkout-Bereich
        if p["score"] <= 170:
            ways = get_d_checkouts(p["score"])
            if ways:
                pygame.draw.rect(self.screen, (10, 50, 90), (50, 680, 1820, 380), border_radius=20)
                self.screen.blit(self.font_co_title.render("Mögliche Checkwege:", True, (0, 255, 255)), (100, 700))
                for i, w in enumerate(ways[:3]):
                    self.screen.blit(self.font_co.render(f"Weg {i+1}: {w}", True, (255, 255, 255)), (100, 800 + i * 85))

        # --- KLEINES HINWEISFELD RECHTS UNTEN ---
        if self.waiting_for_remove:
            # Hintergrund-Box (Rot)
            msg_rect = pygame.Rect(1420, 950, 450, 100)
            pygame.draw.rect(self.screen, (180, 0, 0), msg_rect, border_radius=15)
            pygame.draw.rect(self.screen, (255, 255, 255), msg_rect, width=3, border_radius=15)
            
            # Texte
            t1 = self.font_msg.render("PFEILE ZIEHEN!", True, (255, 255, 255))
            t2 = pygame.font.SysFont("Arial", 24, bold=True).render("[BACKSPACE] ZUM BESTÄTIGEN", True, (255, 255, 255))
            
            # Positionierung in der Box
            self.screen.blit(t1, (msg_rect.centerx - t1.get_width()//2, msg_rect.y + 15))
            self.screen.blit(t2, (msg_rect.centerx - t2.get_width()//2, msg_rect.y + 60))