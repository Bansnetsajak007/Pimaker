from .fingertip_backends import run_legacy_webcam, run_tasks_webcam
from .fingertip_draw import annotate_legacy_frame, annotate_tasks_frame

__all__ = [
    "run_legacy_webcam",
    "run_tasks_webcam",
    "annotate_legacy_frame",
    "annotate_tasks_frame",
]
