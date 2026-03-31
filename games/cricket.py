# games/cricket.py
import pygame


class CricketGame:
    # Alle im Cricket relevanten Felder:
    # Die Zahlen 20 bis 15 sowie das Bull.
    CRICKET_FIELDS = [20, 19, 18, 17, 16, 15, "BULL"]

    def __init__(self, screen, config, player_names=None):
        # Referenz auf die Pygame-Zeichenfläche
        self.screen = screen

        # Konfigurationsdaten des Spiels
        self.config = config

        # Spielmodus:
        # False = normales Cricket
        # True  = Cut Throat Cricket
        self.cut_throat = bool(config.get("cut_throat", False))

        # Schriftarten für die Anzeige
        # Optisch an x01.py angelehnt
        self.font_big_score = pygame.font.SysFont("Impact", 220)
        self.font_player_name = pygame.font.SysFont("Arial", 80, bold=True)
        self.font_info = pygame.font.SysFont("Arial", 45, bold=True)
        self.font_list = pygame.font.SysFont("Arial", 38, bold=True)

        self.font_co_title = pygame.font.SysFont("Arial", 70, bold=True)
        self.font_co = pygame.font.SysFont("Arial", 48, bold=True)

        self.font_msg = pygame.font.SysFont("Arial", 42, bold=True)
        self.font_bust = pygame.font.SysFont("Arial", 60, bold=True)

        # Spielernamen bestimmen:
        # Wenn Namen übergeben wurden, diese verwenden.
        # Sonst Standardnamen P1, P2, P3, ... anhand der player_count-Konfiguration erzeugen.
        names = player_names if player_names else [f"P{i+1}" for i in range(config.get("player_count", 1))]

        # Liste aller Spielerobjekte
        self.players = []

        # Für jeden Spieler ein Dictionary mit allen relevanten Spieldaten anlegen
        for name in names:
            self.players.append({
                "name": name,  # Anzeigename des Spielers
                "score": 0,    # Punktestand
                "marks": {field: 0 for field in self.CRICKET_FIELDS},  # Treffer/Marks pro Feld
                "visit": [],   # Aktuelle Aufnahme (max. 3 Darts)
                "history": []  # Historie für Undo
            })

        # Index des aktuell aktiven Spielers
        self.current_idx = 0

        # True, wenn nach einer Aufnahme erst das Ziehen der Pfeile bestätigt werden muss
        self.waiting_for_remove = False

        # Gewinner-Index, falls das Spiel beendet ist; sonst None
        self.winner_idx = None

        # Letzter Aktionstext für Statusanzeigen / Debug / mögliche UI-Erweiterungen
        self.last_action_text = ""

    def reset_current_throw(self):
        """Wird aufgerufen, wenn die Kamera erkannt hat, dass die Pfeile gezogen wurden."""
        # Falls das Spiel gerade auf eine Bestätigung zum Pfeileziehen wartet,
        # wird die Aufnahme abgeschlossen.
        if self.waiting_for_remove:
            self.confirm_remove()

    def _normalize_throw(self, val, mult):
        """
        Normalisiert einen Wurf in die interne Repräsentation.

        Erwartet:
        - val  = getroffene Zahl (z. B. 20, 19, 25)
        - mult = Multiplikator (1 = Single, 2 = Double, 3 = Triple)

        Rückgabe:
        - (field, hits)
          field = 15..20 oder "BULL"
          hits  = 1..3 bzw. bei Bull nur 1 oder 2
        - (None, None) bei ungültigem / irrelevantem Wurf
        """
        try:
            val = int(val)
            mult = int(mult)
        except (ValueError, TypeError):
            return None, None

        # Bull-Behandlung:
        # 25 wird intern als "BULL" behandelt.
        # Single Bull = 1 Hit
        # Double Bull = 2 Hits
        if val == 25:
            return "BULL", 2 if mult >= 2 else 1

        # Nur Cricket-relevante Zahlen zulassen: 15 bis 20
        # Der Multiplikator wird auf 1..3 begrenzt.
        if val in [15, 16, 17, 18, 19, 20]:
            return val, max(1, min(3, mult))

        # Alles andere ist im Cricket irrelevant
        return None, None

    def handle_throw(self, val, mult):
        """Verarbeitet einen Wurf von Kamera oder Tastatur."""
        # Keine neuen Würfe zulassen,
        # wenn noch auf das Ziehen der Pfeile gewartet wird
        # oder bereits ein Gewinner feststeht.
        if self.waiting_for_remove or self.winner_idx is not None:
            return

        # Wurf normalisieren
        field, hits = self._normalize_throw(val, mult)

        # Aktiven Spieler holen
        p = self.players[self.current_idx]

        # Undo-Snapshot vorbereiten:
        # Vor jeder Änderung wird der komplette relevante Spielzustand gesichert.
        history_entry = {
            "field": field,
            "hits": hits,
            "scores_before": [pl["score"] for pl in self.players],
            "marks_before": {pl_idx: dict(pl["marks"]) for pl_idx, pl in enumerate(self.players)},
            "visit_before": list(p["visit"]),
            "last_action_text_before": self.last_action_text,
            "winner_before": self.winner_idx,
        }
        p["history"].append(history_entry)

        # Miss / irrelevantes Feld:
        # Wird als "MISS" in der Aufnahme gespeichert.
        if field is None:
            p["visit"].append("MISS")
            self.last_action_text = "Miss"
            self.check_end_visit(p)
            return

        # Alte Mark-Anzahl auf dem Feld
        old_marks = p["marks"][field]

        # Neue Gesamtanzahl der Hits für dieses Feld
        total_after = old_marks + hits

        # Maximal 3 Marks werden gespeichert (Feld ist dann "geschlossen")
        applied_marks = min(3, total_after)

        # Alles über 3 hinaus ist Overflow und kann ggf. Punkte erzeugen
        overflow = max(0, total_after - 3)

        # Marks aktualisieren
        p["marks"][field] = applied_marks

        # Punkte anwenden, falls Overflow vorhanden ist
        scored_points = self._apply_scoring(field, overflow)

        # Feldnamen für die Anzeige formatieren
        label = self._field_label(field)

        # Treffertext abhängig vom Multiplikator erzeugen
        if hits == 1:
            hit_txt = f"S {label}"
        elif hits == 2:
            hit_txt = f"D {label}"
        else:
            hit_txt = f"T {label}"

        # Anzeige- und Aufnahmetext abhängig davon,
        # ob durch den Wurf Punkte entstanden sind
        if scored_points > 0:
            if self.cut_throat:
                # Im Cut-Throat-Modus erhalten Gegner die Punkte
                self.last_action_text = f"{hit_txt} | Gegner +{scored_points}"
                p["visit"].append(f"{hit_txt} (+{scored_points})")
            else:
                # Im normalen Modus bekommt der aktive Spieler die Punkte
                self.last_action_text = f"{hit_txt} | +{scored_points}"
                p["visit"].append(f"{hit_txt} (+{scored_points})")
        else:
            # Kein Scoring, nur Treffer dokumentieren
            self.last_action_text = hit_txt
            p["visit"].append(hit_txt)

        # Nach jedem Wurf prüfen, ob jemand gewonnen hat
        self._check_for_win()

        # Falls noch niemand gewonnen hat:
        # prüfen, ob die Aufnahme mit 3 Darts beendet ist.
        if self.winner_idx is None:
            self.check_end_visit(p)
        else:
            # Bei Gewinn wird ebenfalls auf Bestätigung / Pfeileziehen gewartet
            self.waiting_for_remove = True

    def _apply_scoring(self, field, overflow):
        """
        Vergibt Punkte aus Overflow-Treffern.

        Regeln:
        - Nur Treffer über die 3 Marks hinaus geben Punkte.
        - Punkte zählen nur, solange mindestens ein Gegner dieses Feld
          noch nicht geschlossen hat.
        - Normal:
            aktiver Spieler bekommt die Punkte.
        - Cut Throat:
            alle noch offenen Gegner bekommen die Punkte dazu.
        """
        if overflow <= 0:
            return 0

        # Alle Gegner finden, die dieses Feld noch nicht geschlossen haben
        open_opponents = [
            pl for idx, pl in enumerate(self.players)
            if idx != self.current_idx and pl["marks"][field] < 3
        ]

        # Wenn kein Gegner mehr offen ist, gibt es keine Punkte
        if not open_opponents:
            return 0

        # Punktewert des Felds bestimmen
        # Bull zählt 25, sonst der Zahlenwert des Feldes
        points = overflow * (25 if field == "BULL" else field)

        if self.cut_throat:
            # Im Cut-Throat-Modus bekommen alle offenen Gegner die Punkte aufaddiert
            for opp in open_opponents:
                opp["score"] += points
        else:
            # Im normalen Cricket bekommt der aktive Spieler die Punkte
            self.players[self.current_idx]["score"] += points

        return points

    def _all_closed(self, player):
        """Prüft, ob ein Spieler alle Cricket-Felder geschlossen hat."""
        return all(player["marks"][field] >= 3 for field in self.CRICKET_FIELDS)

    def _check_for_win(self):
        """
        Ermittelt, ob es einen Gewinner gibt.

        Gewinnbedingung:
        - Der Spieler muss alle Felder geschlossen haben.
        - Normal Cricket:
            unter allen geschlossenen Spielern gewinnt der mit den meisten Punkten.
        - Cut Throat:
            unter allen geschlossenen Spielern gewinnt der mit den wenigsten Punkten.
        """
        candidates = []

        # Alle Spieler sammeln, die bereits alles geschlossen haben
        for idx, pl in enumerate(self.players):
            if self._all_closed(pl):
                candidates.append((idx, pl))

        # Niemand hat alles geschlossen -> kein Gewinner
        if not candidates:
            self.winner_idx = None
            return

        if self.cut_throat:
            # Im Cut-Throat gewinnt der geschlossene Spieler
            # mit dem niedrigsten Score
            lowest_score = min(pl["score"] for _, pl in candidates)
            for idx, pl in candidates:
                if pl["score"] == lowest_score:
                    self.winner_idx = idx
                    return
        else:
            # Im normalen Cricket gewinnt der geschlossene Spieler
            # mit dem höchsten Score
            highest_score = max(pl["score"] for _, pl in candidates)
            for idx, pl in candidates:
                if pl["score"] == highest_score:
                    self.winner_idx = idx
                    return

    def undo_last_throw(self):
        """
        Macht den letzten Wurf des aktuellen Spielers rückgängig.

        Einschränkungen:
        - Nur möglich, wenn Historie vorhanden ist.
        - Nicht möglich, während auf Pfeileziehen gewartet wird.
        """
        p = self.players[self.current_idx]
        if not p["history"] or self.waiting_for_remove:
            return

        # Letzten gespeicherten Zustand laden
        last = p["history"].pop()

        # Scores aller Spieler zurücksetzen
        for idx, score in enumerate(last["scores_before"]):
            self.players[idx]["score"] = score

        # Marks aller Spieler zurücksetzen
        for idx in range(len(self.players)):
            self.players[idx]["marks"] = dict(last["marks_before"][idx])

        # Aktuelle Aufnahme des Spielers wiederherstellen
        p["visit"] = list(last["visit_before"])

        # Letzten Anzeigetext wiederherstellen
        self.last_action_text = last["last_action_text_before"]

        # Gewinnerstatus zurücksetzen
        self.winner_idx = last["winner_before"]

    def check_end_visit(self, p):
        """
        Prüft, ob die aktuelle Aufnahme beendet ist.
        Nach 3 Darts muss das Ziehen der Pfeile bestätigt werden.
        """
        if len(p["visit"]) == 3:
            self.waiting_for_remove = True

    def confirm_remove(self):
        """Wird aufgerufen, um einen Zug zu bestätigen, nachdem Pfeile gezogen wurden."""
        if self.waiting_for_remove:
            if self.winner_idx is None:
                # Nur wenn das Spiel noch nicht gewonnen ist:
                # Aufnahme leeren und zum nächsten Spieler wechseln
                self.players[self.current_idx]["visit"] = []
                self.current_idx = (self.current_idx + 1) % len(self.players)

            # Wartezustand beenden
            self.waiting_for_remove = False

    def _field_label(self, field):
        """Gibt eine Feldbezeichnung als String zurück."""
        return "BULL" if field == "BULL" else str(field)

    def _marks_text(self, value):
        """
        Wandelt die Anzahl der Marks in einen Text für die Anzeige um.
        0 -> "-"
        1 -> "X"
        2 -> "XX"
        3+ -> "XXX"
        """
        if value <= 0:
            return "-"
        if value == 1:
            return "X"
        if value == 2:
            return "XX"
        return "XXX"

    def _leader_text(self):
        """
        Liefert einen Text für die aktuelle Führung.

        - Normal Cricket: höchster Score führt
        - Cut Throat: niedrigster Score führt
        """
        if self.cut_throat:
            sorted_players = sorted(self.players, key=lambda pl: pl["score"])
            if sorted_players:
                return f"Führung: {sorted_players[0]['name']} mit {sorted_players[0]['score']} Punkten"
        else:
            sorted_players = sorted(self.players, key=lambda pl: pl["score"], reverse=True)
            if sorted_players:
                return f"Führung: {sorted_players[0]['name']} mit {sorted_players[0]['score']} Punkten"
        return "Führung: -"

    def draw(self):
        """Zeichnet die komplette Cricket-Spielansicht."""
        # Hintergrundfarbe des gesamten Screens
        self.screen.fill((10, 15, 25))

        # Aktiven Spieler holen
        p = self.players[self.current_idx]

        # --- HAUPTFELD LINKS (wie x01) ---
        # Großer linker Bereich mit Infos zum aktiven Spieler
        pygame.draw.rect(self.screen, (25, 40, 70), (50, 40, 1040, 620), border_radius=30)

        # Spielername anzeigen
        self.screen.blit(self.font_player_name.render(p["name"], True, (255, 255, 255)), (100, 60))

        # Spielmodus anzeigen
        mode_text = "CUT THROAT" if self.cut_throat else "NORMAL"
        mode_color = (255, 120, 120) if self.cut_throat else (0, 255, 200)
        self.screen.blit(self.font_info.render(f"Cricket - {mode_text}", True, mode_color), (100, 155))

        # Punktestand des aktiven Spielers groß darstellen
        score_surf = self.font_big_score.render(str(p["score"]), True, (255, 255, 0))
        self.screen.blit(score_surf, (100, 205))

        # Bisherige Würfe der aktuellen Aufnahme anzeigen
        visit_text = "  ".join(p["visit"]) if p["visit"] else "-"
        self.screen.blit(self.font_info.render(f"Aufnahme: {visit_text}", True, (0, 255, 150)), (100, 470))

        # Anzahl geschlossener Felder des aktiven Spielers anzeigen
        closed_count = sum(1 for f in self.CRICKET_FIELDS if p["marks"][f] >= 3)
        self.screen.blit(
            self.font_info.render(f"Geschlossen: {closed_count}/7", True, (200, 200, 200)),
            (100, 540)
        )

        # Führungsanzeige
        leader_text = self._leader_text()
        self.screen.blit(
            self.font_info.render(leader_text, True, (0, 200, 255)),
            (100, 590)
        )

        # --- SPIELERLISTE RECHTS (wie x01) ---
        # Rechter Bereich mit allen Spielern
        pygame.draw.rect(self.screen, (20, 25, 40), (1140, 40, 730, 620), border_radius=20)

        for i, pl in enumerate(self.players):
            y = 70 + i * 75

            # Standardfarbe:
            # aktiver Spieler hell, andere dunkler
            c = (255, 255, 255) if i == self.current_idx else (100, 100, 110)

            # Gewinner hervorheben
            if self.winner_idx == i:
                c = (0, 255, 120)

            # Geschlossene Felder dieses Spielers zählen
            closed = sum(1 for f in self.CRICKET_FIELDS if pl["marks"][f] >= 3)

            # Spielereintrag zusammenbauen
            txt = f"{pl['name']}: {pl['score']} | Closed: {closed}/7"
            self.screen.blit(self.font_list.render(txt, True, c), (1170, y))

        # --- UNTERER INFOBEREICH (anstatt Checkout-Bereich) ---
        # Unterer Bereich für Cricket-Marks und Regelhinweise
        pygame.draw.rect(self.screen, (10, 50, 90), (50, 680, 1820, 380), border_radius=20)

        self.screen.blit(
            self.font_co_title.render("Cricket Übersicht:", True, (0, 255, 255)),
            (100, 700)
        )

        # Kopfzeile mit den relevanten Feldern
        marks_header = "Feld        20        19        18        17        16        15       BULL"
        self.screen.blit(self.font_co.render(marks_header, True, (255, 255, 255)), (100, 790))

        # Marks des aktiven Spielers anzeigen
        active_marks = "Aktiv      "
        for field in self.CRICKET_FIELDS:
            txt = self._marks_text(p["marks"][field])
            active_marks += f"{txt:<10}"
        self.screen.blit(self.font_co.render(active_marks, True, (255, 255, 0)), (100, 855))

        # Vergleichsspieler auswählen:
        # aktuell simpel:
        # - wenn aktiver Spieler nicht Spieler 0 ist -> Spieler 0
        # - sonst Spieler 1, falls vorhanden
        compare_idx = 0 if self.current_idx != 0 else (1 if len(self.players) > 1 else 0)
        compare_player = self.players[compare_idx]

        # Marks des Vergleichsspielers anzeigen
        compare_marks = f"{compare_player['name'][:8]:<10}"
        for field in self.CRICKET_FIELDS:
            txt = self._marks_text(compare_player["marks"][field])
            compare_marks += f"{txt:<10}"
        self.screen.blit(self.font_co.render(compare_marks, True, (200, 200, 200)), (100, 915))

        # Regeltext je nach Spielmodus
        if self.cut_throat:
            rule_text = "Regel: Überschüssige Treffer geben Punkte an noch offene Gegner. Wenigste Punkte gewinnt."
        else:
            rule_text = "Regel: Überschüssige Treffer geben Punkte, solange Gegner das Feld noch nicht geschlossen haben."

        self.screen.blit(
            pygame.font.SysFont("Arial", 28, bold=True).render(rule_text, True, (220, 220, 220)),
            (100, 990)
        )

        # --- HINWEISFELD RECHTS UNTEN (wie x01) ---
        # Meldungsbox, wenn auf Ziehen der Pfeile / Bestätigung gewartet wird
        if self.waiting_for_remove:
            msg_rect = pygame.Rect(1420, 950, 450, 100)

            # Box-Hintergrund und Rahmen
            pygame.draw.rect(self.screen, (180, 0, 0), msg_rect, border_radius=15)
            pygame.draw.rect(self.screen, (255, 255, 255), msg_rect, width=3, border_radius=15)

            if self.winner_idx is not None:
                # Gewinnmeldung anzeigen
                t1 = self.font_msg.render(f"{self.players[self.winner_idx]['name']} GEWINNT!", True, (255, 255, 255))
                t2 = pygame.font.SysFont("Arial", 24, bold=True).render("[BACKSPACE] ZUM BESTÄTIGEN", True, (255, 255, 255))
            else:
                # Standardhinweis zum Pfeileziehen
                t1 = self.font_msg.render("PFEILE ZIEHEN!", True, (255, 255, 255))
                t2 = pygame.font.SysFont("Arial", 24, bold=True).render("[BACKSPACE] ZUM BESTÄTIGEN", True, (255, 255, 255))

            # Texte mittig in der Box platzieren
            self.screen.blit(t1, (msg_rect.centerx - t1.get_width() // 2, msg_rect.y + 15))
            self.screen.blit(t2, (msg_rect.centerx - t2.get_width() // 2, msg_rect.y + 60))
