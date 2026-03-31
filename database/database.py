import sqlite3
import os


class DatabaseManager:
    def __init__(self):
        # Konstruktor der Klasse.
        # Hier wird zuerst der Speicherort der SQLite-Datenbank festgelegt
        # und anschließend sichergestellt, dass die Datenbank initialisiert wird.

        # 1. Pfad-Logik für EXE-Sicherheit:
        # Die Datenbank soll nicht im aktuellen Arbeitsverzeichnis liegen,
        # weil das bei einer .exe oder bei wechselnden Startorten problematisch sein kann.
        # Stattdessen wird ein dauerhafter Ordner im lokalen AppData-Bereich verwendet.
        #
        # Beispiel unter Windows:
        # C:\Users\<Name>\AppData\Local\ProjectDart
        #
        # Falls LOCALAPPDATA nicht gesetzt ist, wird als Fallback das Home-Verzeichnis verwendet.
        app_data_dir = os.path.join(
            os.environ.get('LOCALAPPDATA', os.path.expanduser('~')),
            'ProjectDart'
        )

        # Falls der Zielordner noch nicht existiert, wird er erstellt.
        if not os.path.exists(app_data_dir):
            os.makedirs(app_data_dir)

        # Vollständiger Pfad zur SQLite-Datenbankdatei.
        # Die Datenbank liegt damit an einem festen, persistenten Ort.
        self.db_path = os.path.join(app_data_dir, "projectdart.db")

        # Initialisiert die Datenbankstruktur
        # (Tabellen anlegen, Migrationen ausführen).
        self.init_db()

    def get_connection(self):
        # Erstellt und liefert eine neue SQLite-Verbindung
        # zur Datenbankdatei zurück.
        return sqlite3.connect(self.db_path)

    def init_db(self):
        # Initialisiert die Datenbank.
        # Dabei werden alle benötigten Tabellen angelegt,
        # falls sie noch nicht existieren.
        # Zusätzlich werden einfache Migrationen für bestehende Datenbanken durchgeführt.
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Tabelle für feste Spieler:
            # - id: eindeutige ID
            # - name: eindeutiger Spielername
            # - created_at: Zeitpunkt der Erstellung
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS players (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Migration für bestehende Datenbanken:
            # Prüft, ob die Spalte "when_last_played" bereits existiert.
            # Falls nicht, wird sie nachträglich hinzugefügt.
            #
            # Diese Spalte speichert, wann ein Spieler zuletzt gespielt hat.
            columns = [row[1] for row in cursor.execute("PRAGMA table_info(players)").fetchall()]
            if "when_last_played" not in columns:
                cursor.execute("ALTER TABLE players ADD COLUMN when_last_played TIMESTAMP")

            # Tabelle für Spiele / Matches:
            # - mode: Spielmodus (z. B. 301, 501, etc.)
            # - start_score: Startpunktzahl des Spiels
            # - winner_id: Verweis auf den Gewinner in der players-Tabelle
            # - timestamp: Zeitpunkt, wann das Match gespeichert wurde
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mode TEXT,
                    start_score INTEGER,
                    winner_id INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (winner_id) REFERENCES players (id)
                )
            ''')

            # Tabelle für einzelne Würfe:
            # Diese Tabelle speichert jeden Wurf separat.
            #
            # Felder:
            # - match_id: Zu welchem Match gehört der Wurf?
            # - player_id: Welcher Spieler hat geworfen?
            # - segment: Getroffenes Segment (z. B. 20, 19, Bull)
            # - multiplier: Einfach / Double / Triple
            # - x_rel, y_rel: Relative Trefferposition für Heatmap-Auswertungen
            # - timestamp: Zeitpunkt des Wurfs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS throws (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id INTEGER,
                    player_id INTEGER,
                    segment INTEGER,
                    multiplier INTEGER,
                    x_rel REAL,
                    y_rel REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (match_id) REFERENCES matches (id),
                    FOREIGN KEY (player_id) REFERENCES players (id)
                )
            ''')

            # Speichert alle Änderungen dauerhaft in der Datenbank.
            conn.commit()

    def add_player(self, name):
        """
        Legacy-kompatibel:
        Fügt einen Spieler hinzu, falls dieser noch nicht existiert.

        Rückgabe:
        - True  -> Spieler wurde erfolgreich angelegt
        - False -> Name leer oder Spieler existiert bereits
        """
        # Absicherung: None vermeiden und Leerzeichen am Anfang/Ende entfernen
        name = (name or "").strip()

        # Leere Namen werden nicht gespeichert
        if not name:
            return False

        try:
            with self.get_connection() as conn:
                # Versucht, den Spieler einzufügen
                conn.execute("INSERT INTO players (name) VALUES (?)", (name,))
                conn.commit()
            return True

        except sqlite3.IntegrityError:
            # Tritt z. B. auf, wenn wegen UNIQUE ein Spieler mit demselben Namen
            # bereits existiert.
            return False

    def get_all_players(self):
        # Gibt alle Spielernamen alphabetisch sortiert zurück.
        # LOWER(name) sorgt für eine case-insensitive Sortierung.
        with self.get_connection() as conn:
            rows = conn.execute("SELECT name FROM players ORDER BY LOWER(name) ASC").fetchall()
            return [row[0] for row in rows]

    def search_players(self, prefix, limit=3):
        """
        Liefert bis zu 'limit' Spielernamen, die mit dem übergebenen Präfix beginnen.

        Sortierung:
        1. Spieler, die bereits gespielt haben, zuerst
        2. Danach nach 'when_last_played' absteigend
           (zuletzt gespielte zuerst)
        3. Danach alphabetisch

        Beispiel:
        prefix='ma' findet z. B. 'Marco', 'Marie', 'Max'
        """
        # Eingabe bereinigen
        prefix = (prefix or "").strip()

        # Ohne Präfix keine Suche
        if not prefix:
            return []

        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT name
                FROM players
                WHERE LOWER(name) LIKE LOWER(?)
                ORDER BY
                    CASE WHEN when_last_played IS NULL THEN 1 ELSE 0 END,
                    when_last_played DESC,
                    LOWER(name) ASC
                LIMIT ?
            """, (prefix + "%", limit)).fetchall()

        # Nur die Namenliste zurückgeben
        return [row[0] for row in rows]

    def get_or_create_player(self, name):
        """
        Holt einen existierenden Spieler case-insensitive
        oder legt ihn neu an, falls er noch nicht existiert.

        Rückgabe:
            {"id": int, "name": str, "created": bool}

        Beispiele:
            Existiert schon:
                {"id": 5, "name": "Max", "created": False}

            Neu angelegt:
                {"id": 8, "name": "Lisa", "created": True}
        """
        # Eingabe bereinigen
        name = (name or "").strip()

        # Leerer Name ist ungültig
        if not name:
            return None

        with self.get_connection() as conn:
            # Suche nach bestehendem Spieler unabhängig von Groß-/Kleinschreibung
            row = conn.execute("""
                SELECT id, name
                FROM players
                WHERE LOWER(name) = LOWER(?)
                LIMIT 1
            """, (name,)).fetchone()

            # Falls gefunden: vorhandenen Datensatz zurückgeben
            if row:
                return {"id": row[0], "name": row[1], "created": False}

            # Falls nicht gefunden: neuen Spieler anlegen
            cur = conn.execute(
                "INSERT INTO players (name) VALUES (?)",
                (name,)
            )
            conn.commit()

            # lastrowid enthält die neu vergebene ID
            return {"id": cur.lastrowid, "name": name, "created": True}

    def touch_player_last_played(self, name):
        """
        Aktualisiert 'when_last_played' für genau einen Spieler
        auf den aktuellen Zeitpunkt.
        """
        # Eingabe bereinigen
        name = (name or "").strip()

        # Leeren Namen ignorieren
        if not name:
            return

        with self.get_connection() as conn:
            conn.execute("""
                UPDATE players
                SET when_last_played = CURRENT_TIMESTAMP
                WHERE LOWER(name) = LOWER(?)
            """, (name,))
            conn.commit()

    def touch_players_last_played(self, names):
        """
        Aktualisiert 'when_last_played' für mehrere Spieler.

        Besonderheiten:
        - Leere Namen werden ignoriert
        - Doppelte Namen werden case-insensitive entfernt
          (z. B. 'Max' und 'max' gelten als derselbe Spieler)
        """
        clean_names = []
        seen = set()

        # Eingabeliste bereinigen und Duplikate entfernen
        for name in names:
            n = (name or "").strip()
            key = n.lower()

            # Nur nicht-leere Namen übernehmen,
            # und jeden Namen nur einmal verarbeiten
            if n and key not in seen:
                clean_names.append(n)
                seen.add(key)

        # Falls nach Bereinigung nichts übrig bleibt, sofort beenden
        if not clean_names:
            return

        with self.get_connection() as conn:
            # Für jeden bereinigten Namen den letzten Spielzeitpunkt aktualisieren
            for name in clean_names:
                conn.execute("""
                    UPDATE players
                    SET when_last_played = CURRENT_TIMESTAMP
                    WHERE LOWER(name) = LOWER(?)
                """, (name,))
            conn.commit()
