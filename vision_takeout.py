import cv2
import numpy as np


class TakeoutDetector:
    """
    Erkennt im entzerrten Board-Bild ("Boardspace"), ob das Dartboard wieder leer ist.

    Idee:
    - Es gibt ein Referenzbild des leeren Boards (`clean_board`).
    - Das aktuelle Board-Bild wird mit diesem Referenzbild verglichen.
    - Wenn kaum Unterschiede übrig bleiben, gilt das Board als leer.
    - Wenn noch deutliche Unterschiede vorhanden sind, steckt vermutlich noch ein Dart im Board.

    Begriffe:
    - Boardspace / warped:
      Das Bild wurde bereits perspektivisch entzerrt, sodass das Board frontal betrachtet wird.
    - clean_board:
      Graustufen-Referenzbild des leeren Boards ohne Darts.
    - check_takeout(...):
      Prüft, ob das Board aktuell leer ist.

    Rückgabe von check_takeout(...):
    - is_empty=True  -> Board ist stabil leer
    - is_empty=False -> Es steckt vermutlich noch etwas im Board
    """

    def __init__(self, board_mask):
        """
        Initialisiert den Takeout-Detector.

        Parameter:
        - board_mask:
          Binärmaske des Dartboards im entzerrten Bild.
          Nur innerhalb dieser Maske werden Unterschiede ausgewertet.

        Interne Parameter:
        - thr:
          Threshold für das Differenzbild.
          Pixel mit Differenz > thr werden als "verändert" betrachtet.
        - min_nonzero:
          Maximale Anzahl aktiver Pixel, die noch als "leer" toleriert wird.
        - min_cnt_area:
          Mindestfläche für Konturen, damit kleine Rauschflecken ignoriert werden.
        """
        self.board_mask = board_mask

        # Referenzbild des leeren Boards in Graustufen.
        # Wird erst später über set_clean_board(...) gesetzt.
        self.clean_board = None  # gray

        # Schwellwert für die Differenzbildung:
        # Je höher, desto unempfindlicher gegen kleine Helligkeitsschwankungen.
        self.thr = 22

        # Maximale Anzahl "auffälliger" Pixel, die noch akzeptiert wird.
        # Liegt der Wert darüber, ist das Board vermutlich nicht leer.
        self.min_nonzero = 220

        # Mindestfläche einer Kontur, damit sie als relevant zählt.
        # Kleine Artefakte / Rauschen werden so ignoriert.
        self.min_cnt_area = 120

        # Kernel für Erosion:
        # Damit wird die Board-Maske etwas nach innen verkleinert.
        # Vorteil: Randbereiche des Boards, die oft unruhig oder ungenau sind,
        # werden bei der Takeout-Erkennung ausgeblendet.
        kernel = np.ones((5, 5), np.uint8)

        # Innere Board-Maske:
        # Eine leicht verkleinerte Version der ursprünglichen Board-Maske.
        self.inner_board_mask = cv2.erode(self.board_mask, kernel, iterations=1)

    def _prepare_gray(self, frame_bgr):
        """
        Wandelt ein BGR-Bild in ein geglättetes Graustufenbild um.

        Schritte:
        1. Umwandlung von BGR nach Gray
        2. Gauß-Blur zur Rauschreduktion

        Warum Blur?
        - Kleine Pixel-Schwankungen und Sensorrauschen werden reduziert.
        - Die spätere Differenzbildung wird stabiler.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray

    def set_clean_board(self, frame_bgr):
        """
        Setzt das Referenzbild des leeren Boards.

        Parameter:
        - frame_bgr:
          Aktuelles entzerrtes Board-Bild in BGR, auf dem kein Dart steckt.

        Das Bild wird direkt in Graustufen vorbereitet und gespeichert.
        """
        self.clean_board = self._prepare_gray(frame_bgr)

    def check_takeout(self, warped_frame_bgr, last_hit_contours):
        """
        Prüft, ob das Board leer ist.

        Parameter:
        - warped_frame_bgr:
          Aktuelles entzerrtes Board-Bild in BGR.
        - last_hit_contours:
          Konturen des zuletzt erkannten Treffers.
          Wenn zuvor sicher ein Dart erkannt wurde, wird die Prüfung strenger.

        Rückgabe:
        - is_empty:
          True, wenn das Board als leer eingeschätzt wird.
          False, wenn vermutlich noch ein Dart steckt.
        - debug_img:
          Debug-Bild mit eingezeichneten relevanten Restkonturen.
        """

        # Falls noch kein leeres Referenzbild gesetzt wurde,
        # kann keine sinnvolle Takeout-Prüfung durchgeführt werden.
        # In diesem Fall:
        # - False zurückgeben (nicht leer / unbekannt)
        # - Originalbild als Debug-Bild liefern
        if self.clean_board is None:
            return False, warped_frame_bgr

        # Aktuelles Board-Bild ebenfalls in geglättete Graustufen umwandeln.
        gray = self._prepare_gray(warped_frame_bgr)

        # Absolutedifferenz zwischen Referenzbild und aktuellem Bild:
        # Hohe Werte bedeuten: An dieser Stelle hat sich etwas verändert.
        diff = cv2.absdiff(self.clean_board, gray)

        # Nur innerhalb der inneren Board-Maske auswerten.
        # Alles außerhalb des interessanten Bereichs wird entfernt.
        diff = cv2.bitwise_and(diff, self.inner_board_mask)

        # Binärschwellwert:
        # Nur deutliche Unterschiede bleiben übrig.
        # Pixel > self.thr werden weiß (255), alle anderen schwarz (0).
        _, thr = cv2.threshold(diff, self.thr, 255, cv2.THRESH_BINARY)

        # Kleiner Kernel zum Entfernen von Einzelpixeln / feinem Rauschen.
        kernel_open = np.ones((3, 3), np.uint8)

        # Etwas größerer Kernel zum Schließen kleiner Lücken in erkannten Bereichen.
        kernel_close = np.ones((5, 5), np.uint8)

        # MORPH_OPEN = Erosion + Dilation
        # Entfernt kleine helle Störungen.
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # MORPH_CLOSE = Dilation + Erosion
        # Verbindet nahe zusammenliegende Bereiche und schließt kleine Löcher.
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        # Sicherheitshalber erneut auf die innere Board-Maske begrenzen.
        thr = cv2.bitwise_and(thr, self.inner_board_mask)

        # Anzahl aller weißen Pixel im bereinigten Differenzbild.
        # Dieser Wert ist ein einfacher globaler Indikator dafür,
        # wie stark sich das aktuelle Bild vom leeren Board unterscheidet.
        nonzero = cv2.countNonZero(thr)

        # Konturen aller zusammenhängenden weißen Bereiche finden.
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Nur größere Konturen behalten.
        # Kleine Flächen gelten meist als Rauschen oder unwichtige Restartefakte.
        large_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_cnt_area:
                large_contours.append(cnt)

        # Debug-Bild erzeugen:
        # Auf einer Kopie des aktuellen Bildes werden relevante Restkonturen rot eingezeichnet.
        debug_img = warped_frame_bgr.copy()
        if large_contours:
            cv2.drawContours(debug_img, large_contours, -1, (0, 0, 255), 1)

        # Entscheidungslogik:
        #
        # Fall 1: Es gab vorher einen echten Treffer (last_hit_contours vorhanden)
        # ------------------------------------------------------------
        # Dann sind wir vorsichtiger.
        # Schon moderate Restunterschiede sprechen dafür,
        # dass der Dart vielleicht noch nicht vollständig entfernt wurde.
        #
        # Bedingung für "leer":
        # - wenige aktive Pixel insgesamt
        # - keine große Restkontur vorhanden
        if last_hit_contours:
            is_empty = (nonzero < self.min_nonzero) and (len(large_contours) == 0)
        else:
            # Fall 2: Es gab keinen bekannten letzten Treffer
            # ------------------------------------------------------------
            # Dann darf die Entscheidung etwas lockerer sein,
            # weil wir nicht explizit auf das Entfernen eines gerade erkannten Darts warten.
            #
            # Hier wird der Grenzwert für nonzero reduziert:
            # Nur wenn wirklich sehr wenig Aktivität vorhanden ist, gilt das Board als leer.
            is_empty = (nonzero < (self.min_nonzero * 0.75)) and (len(large_contours) == 0)

        # Ergebnis + Debug-Bild zurückgeben.
        return is_empty, debug_img
