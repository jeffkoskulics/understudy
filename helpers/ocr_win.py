"""Windows.Media.Ocr helper -- the Windows twin of ocr_mac.swift.

Same contract: image paths on stdin, one per line; one JSON object per line on
stdout, flushed per image:

  {"file": ..., "w": W, "h": H,
   "lines": [{"text": ..., "conf": 1.0, "box": [x, y, w, h]}, ...]}
  {"file": ..., "error": ...}

Boxes are top-left-origin pixels of the ORIGINAL image, so they share a frame
with recorded click coordinates. Windows OCR already reports top-left pixels;
the only conversion needed is undoing any downscale done to fit the engine's
maximum image dimension.

Windows OCR gives no confidence score, so conf is always 1.0. That means
pack.CONF_MIN filters nothing on Windows.

Windows OCR boxes words, not lines, so a line's box is the union of its words.

Requires, in the same interpreter that runs understudy:
  python -m pip install winrt-runtime winrt-Windows.Foundation
    winrt-Windows.Globalization winrt-Windows.Graphics.Imaging
    winrt-Windows.Media.Ocr winrt-Windows.Storage winrt-Windows.Storage.Streams
"""
import asyncio
import json
import os
import sys

from winrt.windows.globalization import Language
from winrt.windows.graphics.imaging import (
    BitmapAlphaMode, BitmapDecoder, BitmapInterpolationMode, BitmapPixelFormat,
    BitmapTransform, ColorManagementMode, ExifOrientationMode)
from winrt.windows.media.ocr import OcrEngine
from winrt.windows.storage import FileAccessMode, StorageFile


def make_engine():
    lang = Language("en-US")
    if OcrEngine.is_language_supported(lang):
        return OcrEngine.try_create_from_language(lang)
    return OcrEngine.try_create_from_user_profile_languages()


def emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


async def ocr(engine, path):
    full = os.path.abspath(path)
    try:
        f = await StorageFile.get_file_from_path_async(full)
        stream = await f.open_async(FileAccessMode.READ)
    except Exception as e:
        return {"file": path, "error": "unreadable: %s" % e}
    try:
        dec = await BitmapDecoder.create_async(stream)
        w, h = dec.pixel_width, dec.pixel_height
        limit = OcrEngine.max_image_dimension
        scale = min(1.0, limit / max(w, h))
        xf = BitmapTransform()
        if scale < 1.0:
            xf.scaled_width = max(1, int(w * scale))
            xf.scaled_height = max(1, int(h * scale))
            xf.interpolation_mode = BitmapInterpolationMode.FANT
        bmp = await dec.get_software_bitmap_async(
            BitmapPixelFormat.BGRA8, BitmapAlphaMode.PREMULTIPLIED, xf,
            ExifOrientationMode.IGNORE_EXIF_ORIENTATION,
            ColorManagementMode.DO_NOT_COLOR_MANAGE)
        result = await engine.recognize_async(bmp)
    except Exception as e:
        return {"file": path, "error": str(e)}
    finally:
        stream.close()

    inv = 1.0 / scale
    lines = []
    for line in result.lines:
        rects = [wd.bounding_rect for wd in line.words]
        if not rects:
            continue
        x0 = min(r.x for r in rects)
        y0 = min(r.y for r in rects)
        x1 = max(r.x + r.width for r in rects)
        y1 = max(r.y + r.height for r in rects)
        lines.append({
            "text": line.text,
            "conf": 1.0,
            "box": [round(x0 * inv), round(y0 * inv),
                    round((x1 - x0) * inv), round((y1 - y0) * inv)],
        })
    return {"file": path, "w": w, "h": h, "lines": lines}


async def main():
    engine = make_engine()
    for raw in sys.stdin:
        path = raw.strip()
        if not path:
            continue
        if engine is None:
            emit({"file": path, "error": "no OCR language installed"})
            continue
        emit(await ocr(engine, path))


if __name__ == "__main__":
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
