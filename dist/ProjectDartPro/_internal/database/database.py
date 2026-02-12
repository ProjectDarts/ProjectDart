import sqlite3
import os

class DatabaseManager:
    def __init__(self):
        # 1. Pfad-Logik für EXE-Sicherheit
        # Erstellt einen permanenten Ordner unter C:\Benutzer\Name\AppData\Local\ProjectDart
        app_data_dir = os.path.join(os.environ.get('LOCALAPPDATA', os.path.expanduser('~')), 'ProjectDart')
        
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
        try:
            with self.get_connection() as conn:
                conn.execute("INSERT INTO players (name) VALUES (?)", (name,))
                conn.commit()
        except sqlite3.IntegrityError:
            pass 

    def get_all_players(self):
        with self.get_connection() as conn:
            #fetchall() gibt Listen von Tuples zurück, wir bereiten das hier direkt auf
            rows = conn.execute("SELECT name FROM players").fetchall()
            return [row[0] for row in rows]