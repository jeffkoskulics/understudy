import json

from understudy import participants


def test_name_list_roundtrip(tmp_path, capsys):
    d = str(tmp_path)
    with open(tmp_path / "speakers.jsonl", "w") as fh:
        for r in [(0, 2, "S1"), (2, 7, "S2"), (8, 9, "S2"), (9, 10, "local")]:
            fh.write(json.dumps({"t0": r[0], "t1": r[1], "source": "system",
                                 "speaker_id": r[2], "confidence": 1}) + "\n")
    assert participants.main([d, "S2", "Maria"]) == 0
    assert participants.main([d, "list", "--local-name", "Jeff"]) == 0
    out = capsys.readouterr().out
    assert "Maria" in out and "6.0s" in out and "t=2.0s" in out
    assert participants.load(d) == {"S2": "Maria", "local": "Jeff"}
    assert [r["name"] for r in participants.read_speakers(d) if "name" in r] == ["Maria", "Maria", "Jeff"]
    participants.name(d, "S2", "")
    assert "S2" not in participants.load(d)
