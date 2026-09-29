"""One Python minor for local development, CI and the serving image.

The Dockerfile once ran a different Python from the rest and had no way to say so: wheel
availability differs per minor (statsforecast 2.0.1 has no wheel for 3.13), and the image broke
without any check noticing. `.python-version` is the single source; the Dockerfile's default and
CI's build argument must follow it.
"""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _pinned() -> str:
    return (ROOT / ".python-version").read_text().strip()


def test_the_pin_is_inside_requires_python():
    spec = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["requires-python"]
    low, high = re.findall(r"(\d+)\.(\d+)", spec)
    minor = tuple(int(x) for x in _pinned().split(".")[:2])
    assert (int(low[0]), int(low[1])) <= minor < (int(high[0]), int(high[1]))


def test_the_dockerfile_default_is_the_pinned_minor():
    dockerfile = (ROOT / "Dockerfile").read_text()
    default = re.search(r"^ARG PYTHON_VERSION=(\S+)$", dockerfile, re.MULTILINE)
    assert default and default.group(1) == _pinned()
    assert "FROM python:${PYTHON_VERSION}-slim" in dockerfile
    # the version file has to reach the build, or uv would resolve a Python of its own choosing
    assert re.search(r"^COPY .*\.python-version", dockerfile, re.MULTILINE)


def test_the_image_uses_the_images_python_not_a_downloaded_one():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "UV_PYTHON_DOWNLOADS=never" in dockerfile
    assert "UV_PYTHON_PREFERENCE=only-system" in dockerfile


def test_ci_builds_the_image_with_the_pinned_version_and_runs_it():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "--build-arg PYTHON_VERSION=$(cat .python-version)" in ci
    assert "docker run" in ci and "scripts/docker_smoke.py check" in ci
