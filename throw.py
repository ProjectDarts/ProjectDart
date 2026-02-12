import pygame

class ThrowSimulator:
    def __init__(self):
        # Speicher für manuelle Eingaben (Simulator-Logik)
        self.current_input = ""
        self.multiplier = 1

    def handle_input(self, event):
        """
        Verarbeitet Tastatur-Eingaben für manuelle Würfe.
        Gibt (Wert, Multiplikator) zurück, wenn ENTER gedrückt wurde.
        """
        if event.type == pygame.KEYDOWN:
            # Multiplikator-Umschalter
            if event.key == pygame.K_d:
                self.multiplier = 2
                print("[INPUT] Nächster Wurf: DOUBLE")
                return None
            
            if event.key == pygame.K_t:
                self.multiplier = 3
                print("[INPUT] Nächster Wurf: TRIPLE")
                return None

            # Zahlen-Eingabe
            if pygame.K_0 <= event.key <= pygame.K_9:
                char = event.unicode
                self.current_input += char
                print(f"[INPUT] Aktuelle Zahl: {self.current_input}")
                return None

            # Bestätigung mit Enter
            if event.key == pygame.K_RETURN:
                if self.current_input != "":
                    try:
                        value = int(self.current_input)
                        mult = self.multiplier
                        
                        # Reset für den nächsten Wurf
                        self.reset_input()
                        
                        print(f"[HIT] Manuelle Eingabe: {value} x {mult}")
                        return (value, mult)
                    except ValueError:
                        self.reset_input()
                        return None
                return None

            # Abbruch/Löschen mit Escape
            if event.key == pygame.K_ESCAPE:
                self.reset_input()
                print("[INPUT] Eingabe gelöscht")
                return None

        return None

    def reset_input(self):
        """Setzt die manuelle Eingabe zurück."""
        self.current_input = ""
        self.multiplier = 1

    def format_hit_to_string(self, hit):
        """Hilfsfunktion für die Anzeige (z.B. (20, 3) -> 'T20')"""
        if not hit or not isinstance(hit, tuple) or len(hit) != 2:
            return ""
            
        val, mult = hit
        if val == 0: return "Miss"
        if val == 25: return "DBULL" if mult == 2 else "SBULL"
        
        prefix = ""
        if mult == 3: prefix = "T"
        elif mult == 2: prefix = "D"
        
        return f"{prefix}{val}"