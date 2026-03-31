import pygame
import os
import sys

# --- EXE-Pfad-Korrektur für Ressourcen ---
# Diese Hilfsfunktion sorgt dafür, dass Ressourcen sowohl
# im normalen Python-Start als auch in einer gebauten EXE
# (z. B. via PyInstaller) korrekt gefunden werden.
def resource_path(relative_path):
    try:
        # Wenn das Programm als EXE läuft, legt PyInstaller
        # temporär alle Dateien in sys._MEIPASS ab.
        base_path = sys._MEIPASS
    except Exception:
        # Im normalen Entwicklungsmodus wird einfach
        # das aktuelle Arbeitsverzeichnis verwendet.
        base_path = os.path.abspath(".")
    # Baut den vollständigen Pfad zur gewünschten Ressource.
    return os.path.join(base_path, relative_path)
# -----------------------------------------

# EXE-kompatibler Import der Checkouts
# Zuerst wird versucht, die Funktion normal aus dem Paket zu importieren.
try:
    from games.d_checkouts import get_d_checkouts
except ImportError:
    try:
        # Falls das nicht funktioniert, wird ein relativer Import versucht.
        from .d_checkouts import get_d_checkouts
    except ImportError:
        # Falls die Datei komplett fehlt, gibt es einen sicheren Fallback:
        # einfach eine leere Checkout-Liste zurückgeben.
        def get_d_checkouts(score):
            return []

class X01Game:
    def __init__(self, screen, config, player_names=None):
        """
        Initialisiert ein neues X01-Spiel.

        Parameter:
        - screen: Pygame-Surface/Fenster, auf dem gezeichnet wird
        - config: Dictionary mit Spielkonfiguration
        - player_names: optionale Liste mit Spielernamen
        """
        self.screen = screen
        self.config = config

        # Fonts für die verschiedenen Anzeigeelemente
        # Großes Score-Feld in auffälliger Schrift
        self.font_big_score = pygame.font.SysFont("Impact", 220)

        # Name des aktuellen Spielers
        self.font_player_name = pygame.font.SysFont("Arial", 80, bold=True)

        # Allgemeine Info-Texte und Statistiken
        self.font_info = pygame.font.SysFont("Arial", 45, bold=True)

        # Schrift für die Spielerliste rechts
        self.font_list = pygame.font.SysFont("Arial", 38, bold=True)

        # Überschrift und Text im Checkout-Bereich
        self.font_co_title = pygame.font.SysFont("Arial", 70, bold=True)
        self.font_co = pygame.font.SysFont("Arial", 63, bold=True)

        # Schrift für Hinweisbox unten rechts
        self.font_msg = pygame.font.SysFont("Arial", 42, bold=True)

        # Extra-Schrift für Bust-Anzeige
        self.font_bust = pygame.font.SysFont("Arial", 60, bold=True)

        # Spielerliste anlegen
        self.players = []

        # Falls keine Spielernamen übergeben wurden,
        # werden Standardnamen P1, P2, ... erzeugt.
        names = player_names if player_names else [
            f"P{i+1}" for i in range(config["player_count"])
        ]

        # Für jeden Spieler wird ein Dictionary mit allen relevanten Daten erstellt.
        for name in names:
            self.players.append({
                "name": name,                       # Anzeigename des Spielers
                "score": config["start_score"],    # aktueller Restscore
                "active_in": (config["in_mode"] == "Single In"),  # ob Spieler bereits "drin" ist
                "legs": 0,                         # gewonnene Legs
                "sets": 0,                         # gewonnene Sets
                "darts_total": 0,                 # insgesamt geworfene Darts
                "points_total": 0,                # insgesamt erzielte Punkte
                "first_9_points": 0,              # Punkte in den ersten 9 Darts
                "first_9_darts": 0,               # Anzahl der First-9-Darts
                "avg_170_sum": 0,                 # Summe aller "Average to 170"-Werte
                "avg_170_count": 0,               # Anzahl, wie oft <=170 erstmals erreicht wurde
                "reached_170_this_leg": False,    # Marker: in diesem Leg schon einmal <=170 erreicht?
                "visit": [],                      # aktuelle Aufnahme (max. 3 Darts)
                "history": []                     # Wurfhistorie für Undo
            })

        # Index des aktuell aktiven Spielers
        self.current_idx = 0

        # Flag: wartet das Spiel darauf, dass die Pfeile gezogen werden?
        self.waiting_for_remove = False

        # Flag: war die aktuelle Aufnahme ein Bust?
        self.is_bust = False

    def reset_current_throw(self):
        """
        Wird aufgerufen, wenn die Kamera erkannt hat,
        dass die Pfeile aus dem Board gezogen wurden.
        """
        if self.waiting_for_remove:
            self.confirm_remove()

    def handle_throw(self, val, mult):
        """
        Verarbeitet einen einzelnen Wurf von Kamera oder Tastatur.

        Parameter:
        - val: Feldwert (z. B. 20)
        - mult: Multiplikator (1, 2, 3)
        """
        # Wenn noch auf das Ziehen der Pfeile gewartet wird,
        # darf kein neuer Wurf verarbeitet werden.
        if self.waiting_for_remove:
            return

        # Eingabewerte sicher in Integer umwandeln.
        # Falls ungültig, Wurf ignorieren.
        try:
            val, mult = int(val), int(mult)
        except (ValueError, TypeError):
            return

        # Aktiven Spieler holen
        p = self.players[self.current_idx]

        # Erzielte Punkte berechnen
        pts = val * mult

        # Vor dem Verändern des Spielerstatus wird alles,
        # was für Undo nötig ist, in der Historie gespeichert.
        p["history"].append({
            "score_before": p["score"],                 # Score vor dem Wurf
            "pts": pts,                                 # erzielte Punkte
            "active_in_before": p["active_in"],         # In-Status vorher
            "f9_valid": (p["darts_total"] < 9),         # zählte der Dart noch zu First 9?
            "was_170_check": p["reached_170_this_leg"]  # war <=170 schon erreicht?
        })

        # Wenn der Spieler noch nicht "in" ist, prüfen wir die In-Regel.
        if not p["active_in"]:
            # Double In: nur Double aktiviert den Spieler
            # Single In: Spieler ist sofort aktiv
            if (self.config["in_mode"] == "Double In" and mult == 2) or self.config["in_mode"] == "Single In":
                p["active_in"] = True
            else:
                # Dart zählt als geworfen, aber ohne Punkte
                p["visit"].append(0)
                p["darts_total"] += 1
                self.check_end_visit(p)
                return

        # Sonderfall: Miss / Fehlwurf (val == 0)
        # Keine Punkte, kein Bust durch Punkteabzug
        if val == 0:
            target = p["score"]
            bust = False
        else:
            # Neuen Restscore berechnen
            target = p["score"] - pts
            bust = False

            # Bust-Regeln prüfen
            if target < 0:
                # Unter 0 ist immer Bust
                bust = True
            elif target == 1:
                # 1 Rest ist in Double/Master Out unzulässig
                if self.config["out_mode"] != "Single Out":
                    bust = True
            elif target == 0:
                # Exaktes Finish nur mit gültigem Out
                if self.config["out_mode"] == "Double Out" and mult != 2:
                    bust = True
                elif self.config["out_mode"] == "Master Out" and mult < 2:
                    bust = True

        if bust:
            # Bust-Zustand aktivieren
            self.is_bust = True

            # Der aktuelle Dart wird noch in die Aufnahme übernommen
            p["visit"].append(pts)

            # Restliche Darts der Aufnahme gelten als verbraucht
            remaining_darts = 3 - len(p["visit"])
            p["darts_total"] += (remaining_darts + 1)

            # Jetzt muss auf das Ziehen der Pfeile gewartet werden
            self.waiting_for_remove = True
        else:
            # Kein Bust: Score aktualisieren
            p["score"] = target

            # Punkte in Gesamtstatistik eintragen
            p["points_total"] += pts

            # Falls noch innerhalb der ersten 9 Darts,
            # zur First-9-Statistik hinzufügen
            if p["darts_total"] < 9:
                p["first_9_points"] += pts
                p["first_9_darts"] += 1

            # Gesamtanzahl Darts erhöhen
            p["darts_total"] += 1

            # Wurf in aktueller Aufnahme speichern
            p["visit"].append(pts)

            # Prüfen, ob der Spieler in diesem Leg erstmals <=170 erreicht hat
            if p["score"] <= 170 and not p["reached_170_this_leg"]:
                c_avg = (p["points_total"] / (p["darts_total"] / 3)) if p["darts_total"] > 0 else 0
                p["avg_170_sum"] += c_avg
                p["avg_170_count"] += 1
                p["reached_170_this_leg"] = True

            # Wenn Score exakt 0: Leg gewonnen
            if p["score"] == 0:
                self.process_leg_win(p)
            else:
                # Ansonsten prüfen, ob die Aufnahme beendet ist
                self.check_end_visit(p)

    def undo_last_throw(self):
        """
        Macht den letzten Wurf des aktuellen Spielers rückgängig.
        """
        p = self.players[self.current_idx]

        # Wenn keine Historie vorhanden ist, gibt es nichts rückgängig zu machen.
        if not p["history"]:
            return

        # Letzten Historieneintrag holen
        last = p["history"].pop()

        # Bust-Zustand zurücksetzen
        self.is_bust = False

        # Falls durch den letzten Wurf erstmals <=170 erreicht wurde,
        # muss diese Statistik zurückgenommen werden.
        if p["reached_170_this_leg"] and not last["was_170_check"]:
            if p["avg_170_count"] > 0:
                p["avg_170_count"] -= 1
            p["reached_170_this_leg"] = False

        # Alten Score und In-Status wiederherstellen
        p["score"] = last["score_before"]
        p["active_in"] = last["active_in_before"]

        # Letzten Dart aus aktueller Aufnahme entfernen
        if p["visit"]:
            p["visit"].pop()

        # Punkte aus Gesamtstatistik zurücknehmen
        p["points_total"] -= last["pts"]

        # Falls der Dart zu First 9 gezählt hat, auch dort zurücknehmen
        if last["f9_valid"]:
            p["first_9_points"] -= last["pts"]
            p["first_9_darts"] -= 1

        # Dartanzahl um 1 reduzieren, aber nicht unter 0
        p["darts_total"] = max(0, p["darts_total"] - 1)

        # Warten auf Pfeileziehen abbrechen
        self.waiting_for_remove = False

    def check_end_visit(self, p):
        """
        Prüft, ob die aktuelle Aufnahme beendet ist.
        Eine Aufnahme endet nach 3 Darts.
        """
        if len(p["visit"]) == 3:
            self.waiting_for_remove = True

    def confirm_remove(self):
        """
        Wird aufgerufen, um eine Aufnahme oder ein beendetes Leg zu bestätigen,
        nachdem die Pfeile gezogen wurden.
        """
        if self.waiting_for_remove:
            p = self.players[self.current_idx]

            # 1. Aktuelle Aufnahme des Spielers leeren
            p["visit"] = []

            # 2. Zum nächsten Spieler wechseln
            self.current_idx = (self.current_idx + 1) % len(self.players)

            # 3. Status zurücksetzen
            self.waiting_for_remove = False
            self.is_bust = False

    def process_leg_win(self, winner):
        """
        Verarbeitet einen Leg-Gewinn und setzt bei Bedarf
        Legs/Sets sowie das nächste Leg zurück.
        """
        # Gewinner bekommt ein Leg
        winner["legs"] += 1

        # Falls nicht im Endlosmodus gespielt wird:
        # Prüfen, ob genug Legs für ein Set erreicht wurden
        if not self.config["endlos"] and winner["legs"] >= self.config["legs_to_win"]:
            winner["sets"] += 1

            # Nach Setgewinn werden alle Legs zurückgesetzt
            for pl in self.players:
                pl["legs"] = 0

        # Neues Leg vorbereiten: alle Spieler zurücksetzen
        for pl in self.players:
            pl["score"] = self.config["start_score"]
            pl["active_in"] = (self.config["in_mode"] == "Single In")
            pl["visit"], pl["reached_170_this_leg"] = [], False
            pl["history"] = []

        # Vor dem nächsten Leg auf Bestätigung/Pfeileziehen warten
        self.waiting_for_remove = True

    def draw(self):
        """
        Zeichnet die komplette Spielansicht auf den Bildschirm.
        """
        # Hintergrundfarbe des gesamten Screens
        self.screen.fill((10, 15, 25))

        # Aktiven Spieler holen
        p = self.players[self.current_idx]

        # =========================
        # Hauptfeld links
        # =========================
        pygame.draw.rect(self.screen, (25, 40, 70), (50, 40, 1040, 620), border_radius=30)

        # Spielername anzeigen
        self.screen.blit(
            self.font_player_name.render(p["name"], True, (255, 255, 255)),
            (100, 60)
        )

        # Großen Restscore rendern
        score_surf = self.font_big_score.render(str(p["score"]), True, (255, 255, 0))
        self.screen.blit(score_surf, (100, 140))

        # Entweder Bust-Anzeige oder aktuelle Aufnahme anzeigen
        if self.is_bust:
            self.screen.blit(
                self.font_bust.render("BUST!", True, (255, 50, 50)),
                (100, 390)
            )
        else:
            v_txt = "  ".join([str(x) for x in p["visit"]])
            self.screen.blit(
                self.font_info.render(f"Aufnahme: {v_txt}", True, (0, 255, 150)),
                (100, 390)
            )

        # =========================
        # Statistiken des aktuellen Spielers
        # =========================

        # 3-Dart-Average
        avg_3 = (p["points_total"] / (p["darts_total"] / 3)) if p["darts_total"] > 0 else 0.0

        # First-9-Average
        f9_avg = (p["first_9_points"] / (p["first_9_darts"] / 3)) if p["first_9_darts"] > 0 else 0.0

        # Average bis zum ersten Erreichen von 170 oder weniger
        avg_170 = (p["avg_170_sum"] / p["avg_170_count"]) if p["avg_170_count"] > 0 else 0.0

        # Startposition und Zeilenhöhe für Statistik-Text
        sy, lh = 460, 50

        self.screen.blit(
            self.font_info.render(f"ø Gesamt: {avg_3:.2f}", True, (200, 200, 200)),
            (100, sy)
        )
        self.screen.blit(
            self.font_info.render(f"ø First 9: {f9_avg:.2f}", True, (200, 200, 200)),
            (100, sy + lh)
        )
        self.screen.blit(
            self.font_info.render(f"ø to 170: {avg_170:.2f}", True, (0, 200, 255)),
            (100, sy + lh * 2)
        )
        self.screen.blit(
            self.font_info.render(f"Darts: {p['darts_total']}", True, (150, 150, 150)),
            (850, 590)
        )

        # =========================
        # Spielerliste rechts
        # =========================
        pygame.draw.rect(self.screen, (20, 25, 40), (1140, 40, 730, 620), border_radius=20)

        for i, pl in enumerate(self.players):
            y = 70 + i * 75

            # Aktiver Spieler wird heller dargestellt
            c = (255, 255, 255) if i == self.current_idx else (100, 100, 110)

            # Average des jeweiligen Spielers
            l_avg = (pl["points_total"] / (pl["darts_total"] / 3)) if pl["darts_total"] > 0 else 0.0

            # Anzeige je nach Spielmodus:
            # Endlosmodus zeigt nur Legs, sonst Legs + Sets
            suffix = f"L:{pl['legs']}" if self.config["endlos"] else f"L:{pl['legs']} S:{pl['sets']}"

            txt = f"{pl['name']}: {pl['score']} (ø {l_avg:.1f}) | {suffix}"

            self.screen.blit(self.font_list.render(txt, True, c), (1170, y))

        # =========================
        # Checkout-Bereich unten
        # =========================
        if p["score"] <= 170:
            # Mögliche Checkout-Wege für den aktuellen Score holen
            ways = get_d_checkouts(p["score"])

            if ways:
                pygame.draw.rect(self.screen, (10, 50, 90), (50, 680, 1820, 380), border_radius=20)

                self.screen.blit(
                    self.font_co_title.render("Mögliche Checkwege:", True, (0, 255, 255)),
                    (100, 700)
                )

                # Maximal die ersten 3 Wege anzeigen
                for i, w in enumerate(ways[:3]):
                    self.screen.blit(
                        self.font_co.render(f"Weg {i+1}: {w}", True, (255, 255, 255)),
                        (100, 800 + i * 85)
                    )

        # =========================
        # Hinweisfeld unten rechts
        # =========================
        if self.waiting_for_remove:
            # Hintergrundbox
            msg_rect = pygame.Rect(1420, 950, 450, 100)
            pygame.draw.rect(self.screen, (180, 0, 0), msg_rect, border_radius=15)
            pygame.draw.rect(self.screen, (255, 255, 255), msg_rect, width=3, border_radius=15)

            # Hinweistext
            t1 = self.font_msg.render("PFEILE ZIEHEN!", True, (255, 255, 255))
            t2 = pygame.font.SysFont("Arial", 24, bold=True).render(
                "[BACKSPACE] ZUM BESTÄTIGEN", True, (255, 255, 255)
            )

            # Texte mittig in der Box platzieren
            self.screen.blit(t1, (msg_rect.centerx - t1.get_width() // 2, msg_rect.y + 15))
            self.screen.blit(t2, (msg_rect.centerx - t2.get_width() // 2, msg_rect.y + 60))
