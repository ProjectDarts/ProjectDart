# games/cricket.py
import pygame


class CricketGame:
    CRICKET_FIELDS = [20, 19, 18, 17, 16, 15, "BULL"]

    def __init__(self, screen, config, player_names=None):
        self.screen = screen
        self.config = config
        self.cut_throat = bool(config.get("cut_throat", False))

        # Optisch an x01.py angelehnt
        self.font_big_score = pygame.font.SysFont("Impact", 220)
        self.font_player_name = pygame.font.SysFont("Arial", 80, bold=True)
        self.font_info = pygame.font.SysFont("Arial", 45, bold=True)
        self.font_list = pygame.font.SysFont("Arial", 38, bold=True)

        self.font_co_title = pygame.font.SysFont("Arial", 70, bold=True)
        self.font_co = pygame.font.SysFont("Arial", 48, bold=True)

        self.font_msg = pygame.font.SysFont("Arial", 42, bold=True)
        self.font_bust = pygame.font.SysFont("Arial", 60, bold=True)

        names = player_names if player_names else [f"P{i+1}" for i in range(config.get("player_count", 1))]
        self.players = []

        for name in names:
            self.players.append({
                "name": name,
                "score": 0,
                "marks": {field: 0 for field in self.CRICKET_FIELDS},
                "visit": [],
                "history": []
            })

        self.current_idx = 0
        self.waiting_for_remove = False
        self.winner_idx = None
        self.last_action_text = ""

    def reset_current_throw(self):
        """Wird aufgerufen, wenn die Kamera erkannt hat, dass die Pfeile gezogen wurden."""
        if self.waiting_for_remove:
            self.confirm_remove()

    def _normalize_throw(self, val, mult):
        try:
            val = int(val)
            mult = int(mult)
        except (ValueError, TypeError):
            return None, None

        if val == 25:
            return "BULL", 2 if mult >= 2 else 1

        if val in [15, 16, 17, 18, 19, 20]:
            return val, max(1, min(3, mult))

        return None, None

    def handle_throw(self, val, mult):
        """Verarbeitet einen Wurf von Kamera oder Tastatur."""
        if self.waiting_for_remove or self.winner_idx is not None:
            return

        field, hits = self._normalize_throw(val, mult)
        p = self.players[self.current_idx]

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

        # Miss / irrelevant field
        if field is None:
            p["visit"].append("MISS")
            self.last_action_text = "Miss"
            self.check_end_visit(p)
            return

        old_marks = p["marks"][field]
        total_after = old_marks + hits
        applied_marks = min(3, total_after)
        overflow = max(0, total_after - 3)
        p["marks"][field] = applied_marks

        scored_points = self._apply_scoring(field, overflow)

        label = self._field_label(field)
        if hits == 1:
            hit_txt = f"S {label}"
        elif hits == 2:
            hit_txt = f"D {label}"
        else:
            hit_txt = f"T {label}"

        if scored_points > 0:
            if self.cut_throat:
                self.last_action_text = f"{hit_txt} | Gegner +{scored_points}"
                p["visit"].append(f"{hit_txt} (+{scored_points})")
            else:
                self.last_action_text = f"{hit_txt} | +{scored_points}"
                p["visit"].append(f"{hit_txt} (+{scored_points})")
        else:
            self.last_action_text = hit_txt
            p["visit"].append(hit_txt)

        self._check_for_win()

        if self.winner_idx is None:
            self.check_end_visit(p)
        else:
            self.waiting_for_remove = True

    def _apply_scoring(self, field, overflow):
        if overflow <= 0:
            return 0

        open_opponents = [
            pl for idx, pl in enumerate(self.players)
            if idx != self.current_idx and pl["marks"][field] < 3
        ]

        if not open_opponents:
            return 0

        points = overflow * (25 if field == "BULL" else field)

        if self.cut_throat:
            for opp in open_opponents:
                opp["score"] += points
        else:
            self.players[self.current_idx]["score"] += points

        return points

    def _all_closed(self, player):
        return all(player["marks"][field] >= 3 for field in self.CRICKET_FIELDS)

    def _check_for_win(self):
        candidates = []
        for idx, pl in enumerate(self.players):
            if self._all_closed(pl):
                candidates.append((idx, pl))

        if not candidates:
            self.winner_idx = None
            return

        if self.cut_throat:
            lowest_score = min(pl["score"] for _, pl in candidates)
            for idx, pl in candidates:
                if pl["score"] == lowest_score:
                    self.winner_idx = idx
                    return
        else:
            highest_score = max(pl["score"] for _, pl in candidates)
            for idx, pl in candidates:
                if pl["score"] == highest_score:
                    self.winner_idx = idx
                    return

    def undo_last_throw(self):
        p = self.players[self.current_idx]
        if not p["history"] or self.waiting_for_remove:
            return

        last = p["history"].pop()

        for idx, score in enumerate(last["scores_before"]):
            self.players[idx]["score"] = score

        for idx in range(len(self.players)):
            self.players[idx]["marks"] = dict(last["marks_before"][idx])

        p["visit"] = list(last["visit_before"])
        self.last_action_text = last["last_action_text_before"]
        self.winner_idx = last["winner_before"]

    def check_end_visit(self, p):
        if len(p["visit"]) == 3:
            self.waiting_for_remove = True

    def confirm_remove(self):
        """Wird aufgerufen, um einen Zug zu bestätigen, nachdem Pfeile gezogen wurden."""
        if self.waiting_for_remove:
            if self.winner_idx is None:
                self.players[self.current_idx]["visit"] = []
                self.current_idx = (self.current_idx + 1) % len(self.players)

            self.waiting_for_remove = False

    def _field_label(self, field):
        return "BULL" if field == "BULL" else str(field)

    def _marks_text(self, value):
        if value <= 0:
            return "-"
        if value == 1:
            return "X"
        if value == 2:
            return "XX"
        return "XXX"

    def _leader_text(self):
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
        self.screen.fill((10, 15, 25))
        p = self.players[self.current_idx]

        # --- HAUPTFELD LINKS (wie x01) ---
        pygame.draw.rect(self.screen, (25, 40, 70), (50, 40, 1040, 620), border_radius=30)

        self.screen.blit(self.font_player_name.render(p["name"], True, (255, 255, 255)), (100, 60))

        mode_text = "CUT THROAT" if self.cut_throat else "NORMAL"
        mode_color = (255, 120, 120) if self.cut_throat else (0, 255, 200)
        self.screen.blit(self.font_info.render(f"Cricket - {mode_text}", True, mode_color), (100, 155))

        score_surf = self.font_big_score.render(str(p["score"]), True, (255, 255, 0))
        self.screen.blit(score_surf, (100, 205))

        visit_text = "  ".join(p["visit"]) if p["visit"] else "-"
        self.screen.blit(self.font_info.render(f"Aufnahme: {visit_text}", True, (0, 255, 150)), (100, 470))

        closed_count = sum(1 for f in self.CRICKET_FIELDS if p["marks"][f] >= 3)
        self.screen.blit(
            self.font_info.render(f"Geschlossen: {closed_count}/7", True, (200, 200, 200)),
            (100, 540)
        )

        leader_text = self._leader_text()
        self.screen.blit(
            self.font_info.render(leader_text, True, (0, 200, 255)),
            (100, 590)
        )

        # --- SPIELERLISTE RECHTS (wie x01) ---
        pygame.draw.rect(self.screen, (20, 25, 40), (1140, 40, 730, 620), border_radius=20)

        for i, pl in enumerate(self.players):
            y = 70 + i * 75
            c = (255, 255, 255) if i == self.current_idx else (100, 100, 110)
            if self.winner_idx == i:
                c = (0, 255, 120)

            closed = sum(1 for f in self.CRICKET_FIELDS if pl["marks"][f] >= 3)
            txt = f"{pl['name']}: {pl['score']} | Closed: {closed}/7"
            self.screen.blit(self.font_list.render(txt, True, c), (1170, y))

        # --- UNTERER INFOBEREICH (anstatt Checkout-Bereich) ---
        pygame.draw.rect(self.screen, (10, 50, 90), (50, 680, 1820, 380), border_radius=20)

        self.screen.blit(
            self.font_co_title.render("Cricket Übersicht:", True, (0, 255, 255)),
            (100, 700)
        )

        # Kopfzeile Marks
        marks_header = "Feld        20        19        18        17        16        15       BULL"
        self.screen.blit(self.font_co.render(marks_header, True, (255, 255, 255)), (100, 790))

        # Aktiver Spieler
        active_marks = "Aktiv      "
        for field in self.CRICKET_FIELDS:
            txt = self._marks_text(p["marks"][field])
            active_marks += f"{txt:<10}"
        self.screen.blit(self.font_co.render(active_marks, True, (255, 255, 0)), (100, 855))

        # Führender / Vergleichsspieler
        compare_idx = 0 if self.current_idx != 0 else (1 if len(self.players) > 1 else 0)
        compare_player = self.players[compare_idx]
        compare_marks = f"{compare_player['name'][:8]:<10}"
        for field in self.CRICKET_FIELDS:
            txt = self._marks_text(compare_player["marks"][field])
            compare_marks += f"{txt:<10}"
        self.screen.blit(self.font_co.render(compare_marks, True, (200, 200, 200)), (100, 915))

        # Regeln / Hinweise
        if self.cut_throat:
            rule_text = "Regel: Überschüssige Treffer geben Punkte an noch offene Gegner. Wenigste Punkte gewinnt."
        else:
            rule_text = "Regel: Überschüssige Treffer geben Punkte, solange Gegner das Feld noch nicht geschlossen haben."

        self.screen.blit(
            pygame.font.SysFont("Arial", 28, bold=True).render(rule_text, True, (220, 220, 220)),
            (100, 990)
        )

        # --- HINWEISFELD RECHTS UNTEN (wie x01) ---
        if self.waiting_for_remove:
            msg_rect = pygame.Rect(1420, 950, 450, 100)
            pygame.draw.rect(self.screen, (180, 0, 0), msg_rect, border_radius=15)
            pygame.draw.rect(self.screen, (255, 255, 255), msg_rect, width=3, border_radius=15)

            if self.winner_idx is not None:
                t1 = self.font_msg.render(f"{self.players[self.winner_idx]['name']} GEWINNT!", True, (255, 255, 255))
                t2 = pygame.font.SysFont("Arial", 24, bold=True).render("[BACKSPACE] ZUM BESTÄTIGEN", True, (255, 255, 255))
            else:
                t1 = self.font_msg.render("PFEILE ZIEHEN!", True, (255, 255, 255))
                t2 = pygame.font.SysFont("Arial", 24, bold=True).render("[BACKSPACE] ZUM BESTÄTIGEN", True, (255, 255, 255))

            self.screen.blit(t1, (msg_rect.centerx - t1.get_width() // 2, msg_rect.y + 15))
            self.screen.blit(t2, (msg_rect.centerx - t2.get_width() // 2, msg_rect.y + 60))
