import pygame


class CricketGame:
    CRICKET_FIELDS = [20, 19, 18, 17, 16, 15, "BULL"]

    def __init__(self, screen, config, player_names=None):
        self.screen = screen
        self.config = config
        self.cut_throat = bool(config.get("cut_throat", False))

        self.font_title = pygame.font.SysFont("Arial", 72, bold=True)
        self.font_player = pygame.font.SysFont("Arial", 50, bold=True)
        self.font_score = pygame.font.SysFont("Arial", 54, bold=True)
        self.font_grid_head = pygame.font.SysFont("Arial", 42, bold=True)
        self.font_grid_cell = pygame.font.SysFont("Arial", 40, bold=True)
        self.font_info = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_msg = pygame.font.SysFont("Arial", 42, bold=True)

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
        if self.waiting_for_remove or self.winner_idx is not None:
            return

        field, hits = self._normalize_throw(val, mult)
        p = self.players[self.current_idx]

        hist = {
            "field": field,
            "hits": hits,
            "score_changes": [pl["score"] for pl in self.players],
            "marks_before": {f: p["marks"][f] for f in self.CRICKET_FIELDS},
            "visit_before": list(p["visit"]),
            "last_action_text": self.last_action_text,
        }
        p["history"].append(hist)

        if field is None:
            p["visit"].append("MISS")
            self.last_action_text = "Miss"
            self._check_end_visit(p)
            return

        old_marks = p["marks"][field]
        new_marks = min(3, old_marks + hits)
        overflow = max(0, old_marks + hits - 3)
        p["marks"][field] = new_marks

        scored_on_open_field = self._apply_scoring(field, overflow)
        label = self._field_label(field)
        if scored_on_open_field > 0:
            self.last_action_text = f"{label} +{scored_on_open_field}"
        else:
            self.last_action_text = label
        p["visit"].append(self.last_action_text)

        self._check_for_win()
        if self.winner_idx is None:
            self._check_end_visit(p)
        else:
            self.waiting_for_remove = True

    def _apply_scoring(self, field, overflow):
        if overflow <= 0:
            return 0

        open_opponents = [pl for idx, pl in enumerate(self.players)
                          if idx != self.current_idx and pl["marks"][field] < 3]
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
        for idx, pl in enumerate(self.players):
            if not self._all_closed(pl):
                continue
            if self.cut_throat:
                lowest = min(p["score"] for p in self.players)
                if pl["score"] == lowest:
                    self.winner_idx = idx
                    return
            else:
                highest = max(p["score"] for p in self.players)
                if pl["score"] == highest:
                    self.winner_idx = idx
                    return

    def undo_last_throw(self):
        p = self.players[self.current_idx]
        if not p["history"] or self.waiting_for_remove:
            return

        last = p["history"].pop()
        for idx, score in enumerate(last["score_changes"]):
            self.players[idx]["score"] = score
        p["marks"] = dict(last["marks_before"])
        p["visit"] = list(last["visit_before"])
        self.last_action_text = last.get("last_action_text", "")
        self.winner_idx = None

    def _check_end_visit(self, player):
        if len(player["visit"]) >= 3:
            self.waiting_for_remove = True

    def confirm_remove(self):
        if not self.waiting_for_remove:
            return

        if self.winner_idx is None:
            self.players[self.current_idx]["visit"] = []
            self.current_idx = (self.current_idx + 1) % len(self.players)
        self.waiting_for_remove = False

    def _field_label(self, field):
        return "BULL" if field == "BULL" else str(field)

    def _marks_text(self, value):
        if value <= 0:
            return "-"
        return "X" * min(value, 3)

    def draw(self):
        self.screen.fill((10, 15, 25))
        active = self.players[self.current_idx]

        mode_name = "CUT THROAT" if self.cut_throat else "NORMAL"
        title = self.font_title.render(f"CRICKET - {mode_name}", True, (0, 220, 255))
        self.screen.blit(title, (960 - title.get_width() // 2, 35))

        pygame.draw.rect(self.screen, (25, 40, 70), (50, 140, 760, 330), border_radius=28)
        current_name = self.font_player.render(active["name"], True, (255, 255, 255))
        self.screen.blit(current_name, (90, 170))

        objective = "Wenigste Punkte gewinnt" if self.cut_throat else "Meiste Punkte gewinnt"
        obj_surf = self.font_info.render(objective, True, (200, 220, 240))
        self.screen.blit(obj_surf, (90, 235))

        score_surf = self.font_score.render(f"Punkte: {active['score']}", True, (255, 220, 0))
        self.screen.blit(score_surf, (90, 300))

        visit_text = "  |  ".join(active["visit"]) if active["visit"] else "-"
        visit_surf = self.font_info.render(f"Aufnahme: {visit_text}", True, (0, 255, 150))
        self.screen.blit(visit_surf, (90, 375))

        grid_x = 860
        grid_y = 150
        row_h = 90
        col_w_name = 280
        col_w = 95

        pygame.draw.rect(self.screen, (20, 25, 40), (830, 120, 1040, 720), border_radius=24)

        headers = ["Spieler", "20", "19", "18", "17", "16", "15", "B"]
        x = grid_x
        self.screen.blit(self.font_grid_head.render(headers[0], True, (255, 255, 255)), (x, grid_y))
        x += col_w_name
        for header in headers[1:]:
            txt = self.font_grid_head.render(header, True, (255, 255, 255))
            self.screen.blit(txt, (x + (col_w - txt.get_width()) // 2, grid_y))
            x += col_w
        score_head = self.font_grid_head.render("Punkte", True, (255, 255, 255))
        self.screen.blit(score_head, (1650, grid_y))

        for idx, pl in enumerate(self.players):
            row_y = grid_y + 70 + idx * row_h
            color = (255, 255, 255) if idx == self.current_idx else (170, 170, 180)
            if self.winner_idx == idx:
                color = (0, 255, 120)
            self.screen.blit(self.font_grid_cell.render(pl["name"], True, color), (grid_x, row_y))

            x = grid_x + col_w_name
            for field in self.CRICKET_FIELDS:
                cell_txt = self._marks_text(pl["marks"][field])
                txt = self.font_grid_cell.render(cell_txt, True, color)
                self.screen.blit(txt, (x + (col_w - txt.get_width()) // 2, row_y))
                x += col_w

            pts = self.font_grid_cell.render(str(pl["score"]), True, color)
            self.screen.blit(pts, (1680, row_y))

        info_rect = pygame.Rect(50, 520, 760, 240)
        pygame.draw.rect(self.screen, (15, 50, 90), info_rect, border_radius=22)
        lines = [
            "Single = 1, Double = 2, Triple = 3 Marks",
            "Bull = 25, Double Bull = 2 Marks",
            self.last_action_text if self.last_action_text else "Warte auf ersten Wurf"
        ]
        for i, line in enumerate(lines):
            surf = self.font_info.render(line, True, (255, 255, 255))
            self.screen.blit(surf, (80, 560 + i * 55))

        if self.waiting_for_remove:
            msg_rect = pygame.Rect(1320, 930, 550, 110)
            pygame.draw.rect(self.screen, (180, 0, 0), msg_rect, border_radius=15)
            pygame.draw.rect(self.screen, (255, 255, 255), msg_rect, width=3, border_radius=15)
            if self.winner_idx is not None:
                t1 = self.font_msg.render(f"{self.players[self.winner_idx]['name']} GEWINNT!", True, (255, 255, 255))
                t2 = pygame.font.SysFont("Arial", 24, bold=True).render("[BACKSPACE] FÜR NÄCHSTEN SCREEN", True, (255, 255, 255))
            else:
                t1 = self.font_msg.render("PFEILE ZIEHEN!", True, (255, 255, 255))
                t2 = pygame.font.SysFont("Arial", 24, bold=True).render("[BACKSPACE] ZUM BESTÄTIGEN", True, (255, 255, 255))
            self.screen.blit(t1, (msg_rect.centerx - t1.get_width() // 2, msg_rect.y + 18))
            self.screen.blit(t2, (msg_rect.centerx - t2.get_width() // 2, msg_rect.y + 66))
