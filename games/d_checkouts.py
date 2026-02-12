# games/d_checkouts.py

D_CHECKOUT_GUIDE = {
    # --- 170 bis 160 ---
    170: [["T20", "T20", "Bull"]],
    167: [["T20", "T19", "Bull"]],
    164: [["T20", "T18", "Bull"]],
    161: [["T20", "T17", "Bull"]],
    160: [["T20", "T20", "D20"]],
    
    # --- 150er Bereich ---
    158: [["T20", "T20", "D19"]],
    157: [["T20", "T19", "D20"]],
    156: [["T20", "T20", "D18"]],
    155: [["T20", "T19", "D19"]],
    154: [["T20", "T18", "D20"]],
    153: [["T20", "T19", "D18"]],
    152: [["T20", "T20", "D16"]],
    151: [["T20", "T17", "D20"]],
    150: [["T20", "T18", "D18"], ["T19", "T19", "D18"], ["Bull", "Bull", "Bull"]],
    
    # --- 140er Bereich ---
    149: [["T20", "T19", "D16"], ["T19", "T20", "D16"]],
    148: [["T20", "T16", "D20"], ["T18", "T18", "D20"]],
    147: [["T20", "T17", "D18"], ["T19", "T18", "D18"]],
    146: [["T20", "T18", "D16"], ["T19", "T19", "D16"]],
    145: [["T20", "T15", "D20"], ["T19", "T16", "D20"]],
    144: [["T20", "T20", "D12"], ["T18", "T18", "D18"]],
    143: [["T20", "T17", "D16"], ["T19", "T18", "D16"]],
    142: [["T20", "T14", "D20"], ["T19", "T15", "D20"]],
    141: [["T20", "T19", "D12"], ["T20", "T15", "D18"], ["T17", "T20", "D15"]],
    140: [["T20", "T16", "D16"], ["T18", "T18", "D16"], ["T20", "T20", "D10"]],
    
    # --- 130er Bereich ---
    139: [["T20", "T13", "D20"], ["T19", "T14", "D20"], ["T20", "T19", "D11"]],
    138: [["T20", "T18", "D12"], ["T19", "T15", "D18"], ["T20", "T14", "D18"]],
    137: [["T19", "T20", "D10"], ["T17", "T18", "D16"], ["T20", "T19", "D10"]],
    136: [["T20", "T20", "D8"], ["T18", "T14", "D20"], ["T20", "T16", "D14"]],
    135: [["T20", "T15", "D15"], ["T20", "T17", "D12"], ["Bull", "T15", "D20"]],
    134: [["T20", "T14", "D16"], ["T18", "T20", "D10"], ["T17", "T11", "D25"]],
    133: [["T20", "T19", "D8"], ["T17", "T18", "D14"], ["T19", "T12", "D20"]],
    132: [["Bull", "Bull", "D16"], ["T20", "T16", "D12"], ["T19", "T15", "D15"]],
    131: [["T20", "T13", "D16"], ["T19", "T18", "D10"], ["T17", "S20", "D15"]],
    130: [["T20", "S20", "Bull"], ["T19", "T11", "D20"], ["T20", "T10", "D20"]],
    
    # --- 120er Bereich ---
    128: [["T20", "T20", "D4"], ["T18", "T14", "D16"], ["T20", "T16", "D10"]],
    125: [["25", "T20", "D20"], ["Bull", "25", "Bull"], ["T20", "S15", "D25"]],
    121: [["T20", "T15", "D8"], ["T17", "T10", "D20"], ["T19", "S14", "D25"]],
    120: [["T20", "S20", "D20"], ["S20", "S20", "T20"], ["T18", "T18", "D6"]],
    
    # --- Beispielhafter Block für niedrigere krumme Zahlen (100-60) ---
    100: [["T20", "D20"], ["T19", "S11", "D16"], ["S20", "S20", "T20"]],
    96:  [["T20", "D18"], ["T19", "S7", "D16"], ["S20", "T20", "D8"]],
    88:  [["T16", "D20"], ["T20", "D14"], ["S16", "T20", "D6"]],
    84:  [["T16", "D18"], ["T20", "D12"], ["S20", "T16", "D8"]],
    80:  [["T20", "D10"], ["T16", "D16"], ["D20", "D20"]],
    70:  [["T10", "D20"], ["T18", "D8"], ["S20", "S10", "D20"]],
    60:  [["S20", "D20"], ["S10", "Bull"], ["D15", "D15"]],
    
    # --- Der Rest wird durch die get_d_checkouts Funktion berechnet ---
}

def get_d_checkouts(score):
    if score > 170: return []
    
    # 1. Datenbank-Abfrage
    ways = D_CHECKOUT_GUIDE.get(score, [])
    
    # 2. Falls weniger als 3 Wege in der DB sind, füllen wir dynamisch auf
    if len(ways) < 3:
        # Hier ist eine "Sicherheits-Logik", die echte Wege generiert
        # Wir suchen nach einfachen Kombinationen:
        
        # Weg A: Triple + Rest
        for t in [20, 19, 18, 17, 16]:
            val = t * 3
            rem = score - val
            if rem > 0 and rem <= 40 and rem % 2 == 0:
                new_way = [f"T{t}", f"D{rem//2}"]
                if new_way not in ways: ways.append(new_way)
        
        # Weg B: Single + Rest
        for s in [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]:
            rem = score - s
            if rem > 0 and rem <= 40 and rem % 2 == 0:
                new_way = [f"S{s}", f"D{rem//2}"]
                if new_way not in ways: ways.append(new_way)

        # Weg C: Nur Doppel (falls score <= 40)
        if score <= 40 and score % 2 == 0:
            new_way = [f"D{score//2}"]
            if new_way not in ways: ways.insert(0, new_way)

    # Rückgabe als String-Liste, auf 3 begrenzt
    return [" - ".join(w) for w in ways][:3]