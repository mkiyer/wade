"""matplotlib draws off-screen in the tests, so they need no display and no GUI toolkit: on a
headless Windows runner the default, Tk, can fail to start."""
import os

os.environ.setdefault("MPLBACKEND", "Agg")
