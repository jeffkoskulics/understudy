"""Install an optional dependency into the interpreter that will import it.

Both optional routes -- whisper for on-device transcription, sentence models
for activity labelling -- fail the same way: the package is missing, the user
runs `pip install`, and it lands somewhere else. The launcher in `bin/` runs
the project venv, so a bare `pip` on PATH is usually a different Python
altogether. Everything here goes through `sys.executable` for that reason.
"""
import importlib.util
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DependencyError(RuntimeError):
    pass


def requirements_path(name):
    return os.path.join(ROOT, name)


def is_installed(module):
    """Is `module` importable here? -> bool"""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def install_command(requirements, packages):
    """The pip command that installs into *this* interpreter. -> argv list"""
    path = requirements_path(requirements)
    if os.path.exists(path):
        return [sys.executable, "-m", "pip", "install", "-r", path]
    # Installed away from the clone: the pins live in the file, so name the
    # packages directly rather than pointing at a path that is not there.
    return [sys.executable, "-m", "pip", "install"] + list(packages)


def install(module, requirements, packages, progress=None, what=None):
    """Install and verify an optional dependency. -> None"""
    what = what or module
    cmd = install_command(requirements, packages)
    if progress:
        progress("Installing %s..." % what)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = (proc.stdout or b"").decode("utf-8", "replace").strip()
        tail = "\n".join(tail.splitlines()[-10:])
        raise DependencyError(
            "Installing %s failed (pip exited %d). Run it by hand to see the "
            "whole log:\n    %s\n\n%s"
            % (what, proc.returncode, " ".join(cmd), tail))
    if not is_installed(module):
        raise DependencyError(
            "pip reported success but %s still does not import. Try again by "
            "hand:\n    %s" % (what, " ".join(cmd)))


def missing_message(what, why, install_flag, requirements, packages,
                    alternative=None):
    msg = ("%s is not installed. It is optional, because %s. Install it "
           "with:\n    %s\nwhich is the same as:\n    %s"
           % (what, why, install_flag,
              " ".join(install_command(requirements, packages))))
    if alternative:
        msg += "\n" + alternative
    return msg
