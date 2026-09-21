"""Name the activities, with a small text embedding model running locally.

`activity.py` says what kind of input a segment was -- typing, reading,
navigating. That is the part the event stream knows and the pixels do not. It
cannot say what the work *was*, because that lives in the words on screen: the
window title, the text that appeared, what the user said while doing it.

That text is already extracted, so this stage embeds text rather than images.
It is the cheap end of the trade: a 90 MB sentence model on CPU against a
multi-gigabyte vision model on a GPU, over data OCR has already produced. The
frames themselves stay unread here.

Two ways to get a name, and they answer different questions:

  clustering  what activities does this session actually contain? Use it first,
              on a few real recordings, to find out what the label set should
              be -- guessing it up front is how you end up with labels that
              never fire.
  zero-shot   given a label set you have settled on, which one is this? Each
              label is embedded as a sentence and matched by cosine similarity,
              so adding a label costs nothing and needs no training data.

Both attach to the same segment records, beside the rule-based label rather
than on top of it: `label` stays what the input stream proves, `name` is what
the text suggests, and a disagreement between them is informative.
"""
import json
import math
import os
import re

from . import deps

MODULE = "sentence_transformers"
REQUIREMENTS = "requirements-vision.txt"
PACKAGES = ["sentence-transformers>=2.2"]
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Zero-shot works noticeably better on a sentence than on a bare noun phrase:
# "spreadsheet" alone sits near any mention of numbers, while the full frame
# pulls it towards the activity.
LABEL_TEMPLATE = "a person at a computer doing %s"

# Below this cosine, the best label is not a match, it is just the one that
# happened to win. Saying nothing is more useful than a confident wrong name:
# the rule-based label is still there, and an unnamed segment is the signal
# that the label set has a gap in it.
MIN_SCORE = 0.15

STOP = set("""a an the and or of to in on for with is are was were it its this
that at by from as be been being do does did you your they them he she his her
i we our us not no if then than so such can could will would should may might
new open close click type press window app file files page text line lines said
""".split())


class EmbedError(RuntimeError):
    pass


def is_installed():
    return deps.is_installed(MODULE)


def install_command():
    return deps.install_command(REQUIREMENTS, PACKAGES)


def install(progress=None):
    try:
        deps.install(MODULE, REQUIREMENTS, PACKAGES, progress=progress,
                     what="sentence-transformers (a few hundred MB)")
    except deps.DependencyError as exc:
        raise EmbedError(str(exc))


def missing_message():
    return deps.missing_message(
        "sentence-transformers",
        "torch and the weights are a few hundred MB that the rule-based "
        "labels do not need",
        "understudy activity --install", REQUIREMENTS, PACKAGES,
        alternative="The rule-based labels in activity.jsonl work without it.")


def load_model(name=DEFAULT_MODEL, progress=None):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise EmbedError(missing_message())
    if progress:
        progress("Loading the %s embedding model..." % name)
    try:
        return SentenceTransformer(name)
    except Exception as exc:
        raise EmbedError(
            "Could not load the '%s' model (%s).\n"
            "The first run downloads it, so this needs network access once; "
            "afterwards it is cached in ~/.cache/huggingface and runs "
            "offline." % (name, exc))


# -- what a segment says ------------------------------------------------
def _rows(session_dir, name):
    path = os.path.join(session_dir, name)
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def segment_text(session_dir, seg, steps=None, narration=None, limit=40):
    """The words belonging to one segment. -> str

    Window titles first and repeated per distinct window: they are the
    shortest accurate description of where the work happened, and a title
    beats a screenful of chrome text for telling one activity from another.
    """
    parts = []
    for app in seg.get("apps", {}):
        if app:
            parts.append(app)
    if seg.get("window"):
        parts.append(seg["window"])
    parts.append(seg["label"].replace("-", " "))

    start, end = seg["start"], seg["end"]
    for step in steps or []:
        if not (start <= step.get("t", -1) < end):
            continue
        if step.get("action"):
            parts.append(step["action"])
        parts.extend(step.get("new", [])[:6])
        parts.extend(step.get("said", []))
    for said in narration or []:
        if said["start"] < end and said["end"] > start:
            parts.append(said["text"])

    words = " ".join(p for p in parts if p).split()
    return " ".join(words[:limit * 8])


def texts_for(session_dir, segs):
    """Segment texts, using whatever earlier stages have produced."""
    steps = []
    spath = os.path.join(session_dir, "steps.json")
    if os.path.exists(spath):
        steps = json.load(open(spath, encoding="utf-8"))
    narration = []
    npath = os.path.join(session_dir, "transcript.json")
    if os.path.exists(npath):
        narration = json.load(open(npath, encoding="utf-8"))
    return [segment_text(session_dir, s, steps, narration) for s in segs]


# -- vector helpers -----------------------------------------------------
def _normalise(vectors):
    import numpy as np
    arr = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def kmeans(vectors, k, iters=40, seed=0):
    """k-means on unit vectors, so a dot product is the cosine. -> labels

    Written out rather than pulled from scikit-learn: it is twenty lines on
    normalised vectors, and the alternative is another hundred-megabyte
    dependency for one function.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    n = len(vectors)
    k = max(1, min(k, n))

    # k-means++ seeding. Random centres on text embeddings regularly produce
    # an empty cluster, which then has to be special-cased anyway.
    centres = [vectors[rng.integers(n)]]
    for _ in range(1, k):
        d = 1.0 - np.max(np.asarray(centres) @ vectors.T, axis=0)
        d = np.clip(d, 0, None)
        total = d.sum()
        pick = rng.integers(n) if total <= 0 else rng.choice(n, p=d / total)
        centres.append(vectors[pick])
    centres = np.asarray(centres)

    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        new = np.argmax(vectors @ centres.T, axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            members = vectors[labels == c]
            if len(members):
                centre = members.mean(axis=0)
                norm = np.linalg.norm(centre)
                centres[c] = centre / norm if norm else centre
    return labels


def suggest_k(n):
    """A sane cluster count for n segments, when none was asked for."""
    return max(2, min(8, int(round(math.sqrt(n / 2.0))) or 2))


def _terms(text):
    return [w for w in re.findall(r"[a-z][a-z0-9'-]+", text.lower())
            if len(w) > 2 and w not in STOP]


def name_clusters(texts, labels):
    """Name each cluster by the terms that distinguish it. -> {label: name}

    Plain tf-idf over the segment texts. The point is a name a human can read
    on a timeline, not a canonical taxonomy -- clusters are for discovering
    what the label set should be, and then you write the labels down and use
    zero-shot instead.
    """
    docs = {}
    for text, cluster in zip(texts, labels):
        docs.setdefault(int(cluster), []).extend(_terms(text))
    seen = {}
    for terms in docs.values():
        for term in set(terms):
            seen[term] = seen.get(term, 0) + 1
    total = len(docs) or 1

    names = {}
    for cluster, terms in docs.items():
        counts = {}
        for term in terms:
            counts[term] = counts.get(term, 0) + 1
        scored = sorted(counts.items(),
                        key=lambda kv: -kv[1] * math.log(total / (1.0 + seen[kv[0]]) + 1.0))
        top = [term for term, _ in scored[:3]]
        names[cluster] = "/".join(top) if top else "cluster %d" % cluster
    return names


# -- the two routes -----------------------------------------------------
def zero_shot(model, texts, labels):
    """Cosine-match each text against a label set. -> [(label, score)]"""
    vectors = _normalise(model.encode(texts))
    label_vectors = _normalise(model.encode([LABEL_TEMPLATE % l for l in labels]))
    scores = vectors @ label_vectors.T
    best = scores.argmax(axis=1)
    out = []
    for row, i in enumerate(best):
        score = float(scores[row, i])
        out.append((labels[int(i)] if score >= MIN_SCORE else None,
                    round(score, 3)))
    return out


def discover(model, texts, k=None, seed=0):
    """Cluster the texts and name each cluster. -> [(name, cluster index)]"""
    vectors = _normalise(model.encode(texts))
    k = k or suggest_k(len(texts))
    assignments = kmeans(vectors, k, seed=seed)
    names = name_clusters(texts, assignments)
    return [(names[int(c)], int(c)) for c in assignments]


def label(session_dir, segs, labels=None, clusters=None, model=None,
          progress=None):
    """Add a semantic name to each segment and rewrite activity.jsonl.

    Mutates `segs` in place and returns it, so a caller that already has the
    rule-based segments keeps working with the same objects.
    """
    from . import activity

    if not segs:
        return segs
    texts = texts_for(session_dir, segs)
    model = model or load_model(progress=progress)

    if labels:
        if progress:
            progress("Matching %d segments against %d labels..."
                     % (len(segs), len(labels)))
        named = zero_shot(model, texts, labels)
        for seg, (name, score) in zip(segs, named):
            seg["name"] = name
            seg["name_score"] = score
            seg["name_from"] = "zero-shot"
        missed = sum(1 for name, _ in named if name is None)
        if missed and progress:
            progress("%d of %d segments matched no label above %.2f; they keep "
                     "their rule-based label." % (missed, len(named), MIN_SCORE))
    else:
        if progress:
            progress("Clustering %d segments..." % len(segs))
        for seg, (name, cluster) in zip(segs, discover(model, texts, clusters)):
            seg["name"] = name
            seg["cluster"] = cluster
            seg["name_from"] = "cluster"

    activity.write(session_dir, segs)
    return segs
