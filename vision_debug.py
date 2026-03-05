import cv2
import time
from vision import DartVisionSystem

def debug_callback(msg):
    print("[DEBUG EVENT]", msg)

def main():
    vs = DartVisionSystem(hit_callback=debug_callback)

    # Monkey: run loop but show current camera frames
    # simplest: duplicate minimal read loop here if you want overlays later

    try:
        # Just run core (no windows). For real debug, we’d extend core with a "debug sink".
        vs.run()
    except KeyboardInterrupt:
        pass
    finally:
        vs.stop()

if __name__ == "__main__":
    main()
