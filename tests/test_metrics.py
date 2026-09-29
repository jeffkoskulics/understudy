import json
import threading

from understudy.live import metrics as M


def test_sample_aggregates_and_resets(tmp_path):
    m = M.Metrics(tmp_path)
    for v in (0.1, 0.2, 0.3, 0.4, 1.0):
        m.timing("vision.step", v)
    m.count("vision.dropped", 2)
    m.count("vision.dropped")
    m.gauge("vision.queue", 4)
    r = m.sample()
    assert r["timings"]["vision.step"]["n"] == 5
    assert r["timings"]["vision.step"]["p50"] == 0.3
    assert r["timings"]["vision.step"]["max"] == 1.0
    assert r["counters"]["vision.dropped"] == 3
    assert r["gauges"]["vision.queue"] == 4
    assert "sampler_overhead_s" in r
    assert m.sample()["timings"] == {}  # reset; counters persist
    assert m.sample()["counters"]["vision.dropped"] == 3


def test_timer_and_threads(tmp_path):
    m = M.Metrics(tmp_path)
    with m.timer("x"):
        pass
    ts = [threading.Thread(target=lambda: [m.count("c") for _ in range(500)])
          for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    r = m.sample()
    assert r["timings"]["x"]["n"] == 1
    assert r["counters"]["c"] == 2000


def test_sampler_writes_jsonl_and_summary(tmp_path):
    m = M.Metrics(tmp_path, interval=0.05).start()
    m.timing("stage", 0.5)
    m.count("stage.dropped", 7)
    import time
    time.sleep(0.3)
    m.stop()
    lines = [json.loads(x) for x in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(lines) >= 2
    s = M.summarize(tmp_path)
    assert s["samples"] == len(lines)
    assert s["latency"]["stage"]["max"] == 0.5
    assert s["drops"] == {"stage.dropped": 7}
    assert M.main([str(tmp_path)]) == 0


def test_summarize_fake_file(tmp_path):
    recs = [{"t": i, "cpu_pct": 10 * (i + 1), "rss_mb": 100 + i, "counters": {"a.dropped": i},
             "timings": {"s": {"n": 2, "p50": 0.1, "p95": 0.2, "max": 0.3}}} for i in range(3)]
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    s = M.summarize(tmp_path)
    assert s["cpu_pct"] == {"mean": 20.0, "peak": 30}
    assert s["latency"]["s"]["n"] == 6
    assert s["drops"] == {"a.dropped": 2}


def test_write_system_json(tmp_path):
    M.write_system_json(tmp_path, {"models": {"whisper": "small"}})
    d = json.loads((tmp_path / "system.json").read_text())
    for k in ("os", "cpu_model", "cores_logical", "ram_total_mb", "gpus", "python", "packages"):
        assert k in d
    assert d["models"]["whisper"] == "small"
