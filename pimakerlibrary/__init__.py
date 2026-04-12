from .core import make_pi
from .object_detect import detect_objects
from .camera import open_camera, detect_fingertip
from .eye_detect import detect_eyes
from .vision.canvas import open_canvas
from .air_mouse import start_air_mouse
from .virtual_instrument import start_virtual_piano
from .face_swap import start_face_swap
from .rock_paper_scissors import play_rock_paper_scissors
from .games import play_game
from .gesture_controller import start_gesture_controller
from .eye_scroller import start_eye_scroller

__all__ = ["make_pi", "open_camera", "detect_fingertip", "detect_eyes", "open_canvas", "detect_objects", "start_air_mouse", "start_virtual_piano", "start_face_swap", "play_rock_paper_scissors", "play_game", "start_gesture_controller", "start_eye_scroller"]