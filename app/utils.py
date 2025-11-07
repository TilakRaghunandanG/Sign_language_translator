import numpy as np

# Try to import MediaPipe; if not available, provide safe fallbacks so the
# main app can run without MediaPipe/TensorFlow. The fallback functions return
# (None, annotated_frame) for detection and pass-through normalization.
try:
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    def extract_landmarks_from_bgr(frame_bgr, max_hands=1):
        """Extract hand landmarks using MediaPipe. Returns (landmarks_flat, annotated_frame) or (None, frame)."""
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            frame_rgb = frame_bgr[:, :, ::-1]
            result = hands.process(frame_rgb)
            annotated = frame_bgr.copy()
            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                return np.array(coords, dtype=np.float32), annotated
            return None, frame_bgr

    def normalize_landmarks(landmarks_flat):
        xyz = landmarks_flat.reshape(-1, 3)
        wrist = xyz[0]
        xyz_norm = xyz - wrist
        scale = np.max(np.abs(xyz_norm))
        if scale > 0:
            xyz_norm /= scale
        return xyz_norm.reshape(-1)

except Exception:
    # MediaPipe not available — provide inert fallback implementations.
    def extract_landmarks_from_bgr(frame_bgr, max_hands=1):
        """Fallback: no detection possible. Return (None, annotated_frame).

        The app will continue to run but will not produce predictions.
        """
        # Return None for landmarks and a copy of the frame as "annotated".
        try:
            annotated = frame_bgr.copy()
        except Exception:
            # If frame isn't a numpy array, just return it unchanged
            annotated = frame_bgr
        return None, annotated

    def normalize_landmarks(landmarks_flat):
        # Pass-through (no-op) for compatibility
        return landmarks_flat
