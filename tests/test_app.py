from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_optimizes_example():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert not at.error
    labels = [m.label for m in at.metric]
    assert "Expected return" in labels
    assert float(at.metric[0].value.replace(",", "")) > 0
