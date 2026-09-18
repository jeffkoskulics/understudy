"""Frontmost window: identity, title and bounds.

In an application built like an OS -- features scattered across many windows,
opened and closed and dragged around -- the window itself is the strongest
structural signal available, and it costs nothing to sample. Three things come
back, and each earns its place:

  app, title   what the user would say they were doing, free of OCR
  window_id    a stable identity, so a later stage can diff a window against
               the last time *that window* was in front rather than against
               whatever unrelated window happened to precede it
  bounds       enough to tell a window being *moved* from a window whose
               contents changed; dragging redraws most of the screen while
               conveying nothing

Both backends fail soft: a session with no window data is degraded, not broken.
"""
import sys

_UNKNOWN = {"app": "", "title": "", "window_id": None, "bounds": None}


def _mac_backend():
    from Quartz import (
        CGWindowListCopyWindowInfo,
        kCGWindowListOptionOnScreenOnly,
        kCGWindowListExcludeDesktopElements,
        kCGNullWindowID,
    )

    opts = kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements

    def frontmost():
        # Ordered front-to-back; layer 0 is the normal window layer, which
        # skips the menu bar, Dock and overlay panels.
        for w in CGWindowListCopyWindowInfo(opts, kCGNullWindowID) or []:
            if w.get("kCGWindowLayer", -1) != 0:
                continue
            b = w.get("kCGWindowBounds") or {}
            bounds = None
            if b:
                bounds = [int(b.get("X", 0)), int(b.get("Y", 0)),
                          int(b.get("Width", 0)), int(b.get("Height", 0))]
            return {
                "app": w.get("kCGWindowOwnerName") or "",
                "title": w.get("kCGWindowName") or "",
                "window_id": w.get("kCGWindowNumber"),
                "bounds": bounds,
            }
        return dict(_UNKNOWN)

    return frontmost


def _win_backend():
    import ctypes
    from ctypes import wintypes

    u32 = ctypes.windll.user32
    k32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi

    def frontmost():
        hwnd = u32.GetForegroundWindow()
        if not hwnd:
            return dict(_UNKNOWN)

        n = u32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        u32.GetWindowTextW(hwnd, buf, n + 1)

        rect = wintypes.RECT()
        bounds = None
        if u32.GetWindowRect(hwnd, ctypes.byref(rect)):
            bounds = [rect.left, rect.top,
                      rect.right - rect.left, rect.bottom - rect.top]

        pid = wintypes.DWORD()
        u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        app = ""
        # PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ
        h = k32.OpenProcess(0x1000 | 0x0010, False, pid.value)
        if h:
            try:
                nbuf = ctypes.create_unicode_buffer(260)
                if psapi.GetModuleBaseNameW(h, None, nbuf, 260):
                    app = nbuf.value
            finally:
                k32.CloseHandle(h)

        return {"app": app, "title": buf.value or "",
                "window_id": int(hwnd), "bounds": bounds}

    return frontmost


def get_frontmost():
    """Return a callable -> dict(app, title, window_id, bounds), never raising."""
    try:
        raw = _mac_backend() if sys.platform == "darwin" else _win_backend()
    except Exception:
        return lambda: dict(_UNKNOWN)

    def safe():
        try:
            return raw()
        except Exception:
            return dict(_UNKNOWN)

    return safe
