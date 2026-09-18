import unittest
from modules.gesture_classifier import classify_gesture

class TestGestureClassifier(unittest.TestCase):

    def make_features(self, fingers, ti=0.20):
        return {
            "fingers": fingers,
            "thumb_index_distance": ti,
            "wrist_index_distance": 0.30
        }

    def test_open_palm(self):
        f = self.make_features({
            "thumb": True, "index": True, "middle": True,
            "ring": True, "pinky": True
        })
        self.assertEqual(classify_gesture(f)[0], "Open Palm")

    def test_fist(self):
        f = self.make_features({
            "thumb": False, "index": False, "middle": False,
            "ring": False, "pinky": False
        })
        self.assertEqual(classify_gesture(f)[0], "Fist")

    def test_victory(self):
        f = self.make_features({
            "thumb": False, "index": True, "middle": True,
            "ring": False, "pinky": False
        })
        self.assertEqual(classify_gesture(f)[0], "Victory")

    def test_thumbs_up(self):
        f = self.make_features({
            "thumb": True, "index": False, "middle": False,
            "ring": False, "pinky": False
        })
        self.assertEqual(classify_gesture(f)[0], "Thumbs Up")

    def test_ok(self):
        f = self.make_features({
            "thumb": True, "index": False, "middle": True,
            "ring": True, "pinky": True
        }, ti=0.04)
        self.assertEqual(classify_gesture(f)[0], "OK")

if __name__ == "__main__":
    unittest.main()
