#!/bin/sh
# Understudy installer for macOS and Linux.
#
#     sh install.sh
#
# or straight from the web, which clones into ./understudy first:
#     curl -fsSL https://raw.githubusercontent.com/jeffkoskulics/understudy/main/install.sh | sh
#
# Pass --with-whisper to also install on-device transcription (a few hundred
# MB more); without it, `understudy transcribe` hands off to a chat window.
set -e

with_whisper=no
for arg in "$@"; do
    case "$arg" in
        --with-whisper) with_whisper=yes ;;
        *) echo "usage: sh install.sh [--with-whisper]"; exit 2 ;;
    esac
done

root=$(cd "$(dirname "$0")" 2>/dev/null && pwd || true)
if [ -z "$root" ] || [ ! -f "$root/understudy/__main__.py" ]; then
    # Piped from the web: no clone yet.
    command -v git >/dev/null 2>&1 || { echo "git is required"; exit 1; }
    root="$(pwd)/understudy"
    [ -d "$root" ] || git clone https://github.com/jeffkoskulics/understudy "$root"
fi
cd "$root"

command -v python3 >/dev/null 2>&1 || { echo "python3 3.9+ is required"; exit 1; }
[ -x .venv/bin/python ] || python3 -m venv .venv
./.venv/bin/pip install --upgrade pip -q
./.venv/bin/pip install -r requirements.txt -q

if [ "$with_whisper" = yes ]; then
    echo "Installing faster-whisper for on-device transcription..."
    ./.venv/bin/pip install -r requirements-whisper.txt -q
fi

if [ "$(uname -s)" = "Darwin" ]; then
    if command -v swiftc >/dev/null 2>&1; then
        swiftc -O helpers/ocr_mac.swift -o helpers/ocr_mac
    else
        echo "swiftc not found -- install the Xcode command line tools with"
        echo "    xcode-select --install"
        echo "then re-run this script to build the OCR helper."
    fi
fi

echo
echo "Understudy is installed. Record a workflow with:"
echo "    $root/bin/understudy record --out ~/Recordings"
echo
if [ "$with_whisper" = yes ]; then
    echo "On-device transcription is installed too:"
    echo "    $root/bin/understudy transcribe <session>"
else
    echo "Transcription hands off to a chat window by default. For on-device"
    echo "transcription instead (a few hundred MB more):"
    echo "    $root/bin/understudy transcribe --install"
fi
