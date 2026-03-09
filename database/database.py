import sqlite3
import os


class DatabaseManager:
    def __init__(self):
        # 1. Pfad-Logik für EXE-Sicherheit
        # Erstellt einen permanenten Ordner unter C:\Benutzer\Name\AppData\Local\ProjectDart
        app_data_dir = os.path.join(
            os.environ.get('LOCALAPPDATA', os.path.expanduser('~')),
            'ProjectDart'
        )

        if not os.path.exists(app_data_dir):
            os.makedirs(app_data_dir)

        # Datenbank liegt nun sicher und permanent hier
        self.db_path = os.path.join(app_data_dir, "projectdart.db")
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Tabelle für feste Spieler
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS players (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Migration für bestehende Datenbanken:
            # when_last_played hinzufügen, falls noch nicht vorhanden
            columns = [row[1] for row in cursor.execute("PRAGMA table_info(players)").fetchall()]
            if "when_last_played" not in columns:
                cursor.execute("ALTER TABLE players ADD COLUMN when_last_played TIMESTAMP")

            # Tabelle für Spiele (Matches)
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

            # Tabelle für jeden einzelnen Wurf (Heatmap-Daten)
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

            conn.commit()

    def add_player(self, name):
        """Legacy-kompatibel: fügt Spieler hinzu, falls nicht vorhanden."""
        name = (name or "").strip()
        if not name:
            return False

        try:
            with self.get_connection() as conn:
                conn.execute("INSERT INTO players (name) VALUES (?)", (name,))
                conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def get_all_players(self):
        with self.get_connection() as conn:
            rows = conn.execute("SELECT name FROM players ORDER BY LOWER(name) ASC").fetchall()
            return [row[0] for row in rows]

    def search_players(self, prefix, limit=3):
        """
        Liefert bis zu 'limit' Spielernamen, die mit prefix beginnen,
        sortiert nach zuletzt gespielt.
        """
        prefix = (prefix or "").strip()
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

        return [row[0] for row in rows]

    def get_or_create_player(self, name):
        """
        Holt existierenden Spieler case-insensitive oder legt ihn neu an.
        Rückgabe:
            {"id": int, "name": str, "created": bool}
        """
        name = (name or "").strip()
        if not name:
            return None

        with self.get_connection() as conn:
            row = conn.execute("""
                SELECT id, name
                FROM players
                WHERE LOWER(name) = LOWER(?)
                LIMIT 1
            """, (name,)).fetchone()

            if row:
                return {"id": row[0], "name": row[1], "created": False}

            cur = conn.execute(
                "INSERT INTO players (name) VALUES (?)",
                (name,)
            )
            conn.commit()
            return {"id": cur.lastrowid, "name": name, "created": True}

    def touch_player_last_played(self, name):
        """Aktualisiert when_last_played für einen Spieler."""
        name = (name or "").strip()
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
        """Aktualisiert when_last_played für mehrere Spieler."""
        clean_names = []
        seen = set()

        for name in names:
            n = (name or "").strip()
            key = n.lower()
            if n and key not in seen:
                clean_names.append(n)
                seen.add(key)

        if not clean_names:
            return

        with self.get_connection() as conn:
            for name in clean_names:
                conn.execute("""
                    UPDATE players
                    SET when_last_played = CURRENT_TIMESTAMP
                    WHERE LOWER(name) = LOWER(?)
                """, (name,))
            conn.commit()
