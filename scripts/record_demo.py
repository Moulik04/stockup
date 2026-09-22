"""Re-record docs/demo.gif from the real dashboard, headlessly and reproducibly.

The previous recording was a hand-made screen capture and went stale twice: it predates calibrated
serving (it showed the uncalibrated reorder path the README documents as broken) and the cost
sliders. This script drives the running Streamlit app with Playwright, takes a screenshot at each
step and assembles the GIF, so the demo can be regenerated whenever the dashboard changes.

    uv run --with playwright python scripts/record_demo.py

Needs a Chromium that Playwright can find (`python -m playwright install chromium`) and the
production artifacts (`make train`). Playwright is deliberately not a project dependency — it is
tooling for this one script. Per the repo convention this script does not import `reorderpoint`.
"""

from __future__ import annotations

import io
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page, sync_playwright

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "demo.gif"
PORT = 8599
URL = f"http://localhost:{PORT}"
WIDTH = 1280
HEIGHT = 900
GIF_WIDTH = 900
FRAME_MS = 1700


def wait_for_server(timeout: float = 600.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"{URL}/_stcore/health", timeout=30)
            return
        except OSError:
            time.sleep(1)
    raise RuntimeError("streamlit did not start")


def settle(page: Page, ms: int = 2500) -> None:
    """Streamlit reruns the script after every widget change; wait for it to finish."""
    page.wait_for_timeout(ms)
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=300000
    )
    page.wait_for_timeout(500)


def shot(page: Page, frames: list[Image.Image], hold: int = 1) -> None:
    img = Image.open(io.BytesIO(page.screenshot())).convert("RGB")
    h = round(img.height * GIF_WIDTH / img.width)
    img = img.resize((GIF_WIDTH, h), Image.LANCZOS)
    frames.extend([img] * hold)


SLIDER = 'div[data-testid="stSlider"]'


def press(page: Page, index: int, key: str, times: int) -> None:
    """Each Streamlit slider (`st.slider`/`st.select_slider`) renders as a `div[data-testid=
    stSlider]` wrapping a keyboard-focusable `<input tabindex="0">` thumb — not `[role="slider"]`
    (checked against the live DOM; that attribute isn't present in this Streamlit version)."""
    # Streamlit rebuilds the slider DOM nodes on rerun; wait for at least `index + 1` of them to
    # exist (not just the usual `settle()`) before touching one, or a slow rerun races the click.
    page.wait_for_function(
        f"(sel) => document.querySelectorAll(sel).length > {index}", arg=SLIDER, timeout=60000
    )
    thumb = page.locator(SLIDER).nth(index).locator("input[tabindex]")
    # `.click()` fights the slider track div for the pointer-event target and can retry forever;
    # `.focus()` sets focus directly on the input without needing a clickable point.
    thumb.focus(timeout=60000)
    for _ in range(times):
        thumb.press(key)
        page.wait_for_timeout(120)


def main() -> None:
    # Prefer the project's own interpreter over `uv run`, which takes an environment lock and can
    # stall behind another process using it; fall back to `uv run` if there is no local .venv.
    venv_python = REPO / ".venv" / "bin" / "python"
    launcher = (
        [str(venv_python), "-m", "streamlit"]
        if venv_python.exists()
        else ["uv", "run", "streamlit"]
    )
    server = subprocess.Popen(
        [
            *launcher,
            "run",
            "reorderpoint/dashboard.py",
            "--server.headless",
            "true",
            "--server.port",
            str(PORT),
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    frames: list[Image.Image] = []
    try:
        wait_for_server()
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
            page.set_default_timeout(60000)
            page.goto(URL)
            page.wait_for_selector("text=Reorder decision", timeout=600000)
            settle(page, 4000)

            # 1. the reorder decision — calibrated sizing, first thing on the page
            shot(page, frames, hold=2)

            # 2. put stock on hand: the recommendation and status change
            box = page.get_by_label("Current on-hand stock")
            box.fill("6")
            box.press("Enter")
            settle(page)
            shot(page, frames)

            # 3. the model-cost section: which model is cheapest at the default costs
            page.get_by_text("Which model is cheapest to run?").scroll_into_view_if_needed()
            settle(page, 800)
            shot(page, frames, hold=2)

            # 4. holding-rate slider (the third slider: horizon, lost-sale cost, holding rate):
            #    well above the crossover -> the ranking flips
            press(page, 2, "ArrowRight", 8)
            settle(page)
            shot(page, frames, hold=2)

            # 5. back down, then the lost-sale cost slider: a tiny margin flips it too
            press(page, 2, "ArrowLeft", 8)
            press(page, 1, "ArrowLeft", 5)
            settle(page)
            shot(page, frames, hold=2)
            browser.close()
    except Exception:
        try:
            page.screenshot(path=str(REPO / "docs" / "demo_failure.png"))
            print(f"Saved failure screenshot to {REPO / 'docs' / 'demo_failure.png'}")
        except Exception:
            pass
        raise
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()

    quantised = [f.quantize(colors=128, method=Image.MEDIANCUT, dither=Image.NONE) for f in frames]
    quantised[0].save(
        OUT, save_all=True, append_images=quantised[1:], duration=FRAME_MS, loop=0, optimize=True
    )
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    sys.exit(main())
