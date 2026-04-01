import types

import pimakerlibrary.camera as camera


class FakeCap:
    def __init__(self, opened=True, frames=None):
        self._opened = opened
        self._frames = frames or [(True, "frame")]
        self._idx = 0
        self.released = False

    def isOpened(self):
        return self._opened

    def read(self):
        if self._idx >= len(self._frames):
            return (False, None)
        item = self._frames[self._idx]
        self._idx += 1
        return item

    def release(self):
        self.released = True


def test_open_camera_raises_when_camera_cannot_open(monkeypatch):
    fake_cv2 = types.SimpleNamespace(
        VideoCapture=lambda _: FakeCap(opened=False),
        imshow=lambda *_: None,
        waitKey=lambda *_: ord("q"),
        destroyAllWindows=lambda: None,
    )
    monkeypatch.setattr(camera, "cv2", fake_cv2)

    try:
        camera.open_camera()
        assert False, "Expected an exception when camera cannot be opened"
    except Exception as exc:
        assert "Could not open camera" in str(exc)


def test_open_camera_reads_and_releases(monkeypatch):
    cap = FakeCap(opened=True, frames=[(True, "frame")])

    fake_cv2 = types.SimpleNamespace(
        VideoCapture=lambda _: cap,
        imshow=lambda *_: None,
        waitKey=lambda *_: ord("q"),
        destroyAllWindows=lambda: None,
    )
    monkeypatch.setattr(camera, "cv2", fake_cv2)

    camera.open_camera()

    assert cap.released is True