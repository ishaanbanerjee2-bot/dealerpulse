"""Shared AppTest helpers.

Streamlit 1.50's AppTest ButtonGroup assumes multi-select values and iterates a single-select string
character by character. This patch (test-only) wraps scalar values so st.segmented_control can be tested.
"""
from pathlib import Path
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1 import element_tree as et

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def _indices(self):
    val = self.value
    if val is None:
        return []
    if not isinstance(val, (list, tuple)):
        val = [val]
    return [self.options.index(self.format_func(v)) for v in val]


et.ButtonGroup.indices = property(_indices)


def new_app(timeout=180) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=timeout)
    at.run()
    return at


def goto(at: AppTest, view: str) -> AppTest:
    at.session_state["nav"] = view
    at.run()
    return at


def problems(at: AppTest):
    return [e.value for e in at.exception] + [e.value for e in at.error]
