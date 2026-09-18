"""understudy - capture a GUI workflow and turn it into teaching material."""
import sys


def usage():
    print(__doc__.strip())
    print("""
  understudy record [options]    capture a session
  understudy handoff <session>   prepare audio + prompt for a chat LLM
  understudy merge   <session>   merge a pasted transcript (reads stdin)
  understudy pack    <session>   build workflow.md, ready to paste
""")
    return 2


def main(argv):
    if not argv:
        return usage()
    cmd, rest = argv[0], argv[1:]

    if cmd == "record":
        from .record import main as record_main
        return record_main(rest)

    if cmd == "handoff":
        from .handoff import prepare
        info = prepare(rest[0])
        print("Audio:  %s" % info["audio"])
        print("Prompt: %s" % info["prompt"])
        print("\n%d speech segments, %.1fs of narration." %
              (info["segments"], info["speech_seconds"]))
        print("""
Next:
  1. Open a chat session with a model that accepts audio uploads.
  2. Drag in the audio file above, and paste the contents of the prompt file.
  3. Copy the whole reply, then run:
       understudy merge %s
     and paste it in, ending with Ctrl-D.""" % rest[0])
        return 0

    if cmd == "merge":
        from .handoff import merge
        print("Paste the reply, then press Ctrl-D:", file=sys.stderr)
        written, missing = merge(rest[0], sys.stdin.read())
        print("Merged %d segments." % written)
        if missing:
            # Worth stating plainly: a gap here means narration is absent for
            # that stretch, not that it has been silently shifted onto a
            # neighbouring step.
            print("Missing segments: %s" % ", ".join(map(str, missing)))
            print("Re-run the transcription for those, or continue without them.")
        return 0

    if cmd == "pack":
        from .pack import pack
        md, js, n = pack(rest[0])
        print("%s  (%d chars, ~%d tokens)" % (md, n, n // 4))
        print(js)
        return 0

    return usage()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
