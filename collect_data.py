import cv2
import numpy as np
import csv
import os
import time
from mediapipe import solutions as mp_solutions

OUT_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "isl_twohands_dataset.csv")

LABELS = [
    "a","b","c","d","e","f","g","h","i","j","k","l","m",
    "n","o","p","q","r","s","t","u","v","w","x","y","z",
    "hi","thankyou","sorry","how_are_you","good"
]

mp_hands = mp_solutions.hands
hands_model = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

def extract_twohand_landmarks_from_frame(frame_bgr):
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = hands_model.process(image_rgb)

    if not result.multi_hand_landmarks:
        return None

    hands_dict = {}
    for lm, handed in zip(result.multi_hand_landmarks, result.multi_handedness):
        hands_dict[handed.classification[0].label] = lm

    coords = []
    for side in ["Left", "Right"]:
        if side in hands_dict:
            for p in hands_dict[side].landmark:
                coords.extend([p.x, p.y, p.z])
        else:
            coords.extend([0.0] * 63)
    return coords

def main():
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data"), exist_ok=True)

    file_exists = os.path.isfile(OUT_CSV)
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = [f"f{i}" for i in range(126)] + ["label"]
            writer.writerow(header)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open camera")
            return

        print("Label index list:")
        for i, lbl in enumerate(LABELS):
            print(f"{i}: {lbl}")
        print("")
        print("Controls:")
        print("- Type an index (e.g. 10) then press Enter to select label")
        print("- Or type index and pause ~0.8s to auto-select")
        print("- Backspace to edit typed index")
        print("- n = next label, p = previous label")
        print("- c = capture sample, q = quit")

        current_idx = 0
        current_label = LABELS[current_idx]

        typed = ""               # buffer for multi-digit index
        last_digit_time = None
        SELECT_TIMEOUT_SEC = 0.8

        def apply_typed_index():
            nonlocal typed, current_idx, current_label
            if typed == "":
                return
            idx = int(typed)
            typed = ""
            if 0 <= idx < len(LABELS):
                current_idx = idx
                current_label = LABELS[current_idx]
                print(f"Switched label to [{current_idx}]: {current_label}")
            else:
                print(f"Index {idx} out of range (0..{len(LABELS)-1})")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            overlay = f"Label [{current_idx}]: {current_label}"
            if typed != "":
                overlay += f"   (typing index: {typed})"

            cv2.putText(
                frame,
                overlay,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Collect ISL Data", frame)
            key = cv2.waitKey(1) & 0xFF
            now = time.time()

            # If user paused after typing digits, auto-apply
            if typed != "" and last_digit_time is not None and (now - last_digit_time) >= SELECT_TIMEOUT_SEC:
                apply_typed_index()
                last_digit_time = None

            if key == 255:
                continue

            # Quit
            if key == ord("q"):
                break

            # Capture
            if key == ord("c"):
                landmarks = extract_twohand_landmarks_from_frame(frame)
                if landmarks is not None:
                    writer.writerow(landmarks + [current_label])
                    print(f"Captured sample for {current_label}")
                else:
                    print("No hands detected, sample skipped")
                continue

            # Next/prev label
            if key == ord("n"):
                current_idx = (current_idx + 1) % len(LABELS)
                current_label = LABELS[current_idx]
                typed = ""
                last_digit_time = None
                print(f"Switched label to [{current_idx}]: {current_label}")
                continue

            if key == ord("p"):
                current_idx = (current_idx - 1) % len(LABELS)
                current_label = LABELS[current_idx]
                typed = ""
                last_digit_time = None
                print(f"Switched label to [{current_idx}]: {current_label}")
                continue

            # Enter applies typed index (Enter can be 13 or 10)
            if key in (10, 13):
                apply_typed_index()
                last_digit_time = None
                continue

            # Backspace edits typed index
            if key == 8:
                typed = typed[:-1]
                last_digit_time = now if typed != "" else None
                continue

            # Digits build multi-digit index
            if ord("0") <= key <= ord("9"):
                typed += chr(key)
                last_digit_time = now
                continue

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()