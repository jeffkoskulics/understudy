"""Upload bundles directly to a Cloudflare R2 bucket (S3 API via boto3)."""
import argparse
import getpass
import json
import os
import shutil
import socket
import stat
import sys
import tempfile

try:
    import tomllib
except ImportError:  # Python < 3.11
    tomllib = None

CONFIG_PATH = os.path.join("~", ".understudy", "upload.toml")
KEYS = ("account_id", "bucket", "access_key_id", "secret_access_key")
ENV = {k: "UNDERSTUDY_R2_" + k.upper() for k in KEYS}


def config_path():
    return os.path.expanduser(CONFIG_PATH)


def _parse_toml_min(text):
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "[")) and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').replace('\\"', '"').replace("\\\\", "\\")
    return out


def load_config():
    """Env vars win over the file. Raises RuntimeError if incomplete."""
    cfg = {}
    p = config_path()
    if os.path.exists(p):
        with open(p, "rb") as fh:
            text = fh.read().decode("utf-8")
        cfg = tomllib.loads(text) if tomllib else _parse_toml_min(text)
    for k, ev in ENV.items():
        if os.environ.get(ev):
            cfg[k] = os.environ[ev]
    missing = [k for k in KEYS if not cfg.get(k)]
    if missing:
        raise RuntimeError("upload not configured (missing %s). Run: understudy upload --configure"
                           % ", ".join(missing))
    return {k: cfg[k] for k in KEYS}


def save_config(cfg):
    p = config_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    body = "".join('%s = "%s"\n' % (k, str(cfg[k]).replace("\\", "\\\\").replace('"', '\\"'))
                   for k in KEYS)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(body)
    try:
        os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    return p


def make_client(cfg):
    import boto3
    from botocore.config import Config
    return boto3.client(
        "s3", endpoint_url="https://%s.r2.cloudflarestorage.com" % cfg["account_id"],
        region_name="auto", aws_access_key_id=cfg["access_key_id"],
        aws_secret_access_key=cfg["secret_access_key"],
        config=Config(signature_version="s3v4", retries={"max_attempts": 5}))


def test_connection(client, bucket):
    key = "understudy-configure-test/%s.txt" % socket.gethostname()
    client.put_object(Bucket=bucket, Key=key, Body=b"ok")
    client.delete_object(Bucket=bucket, Key=key)


test_connection.__test__ = False  # not a pytest test


def configure(input_fn=input, secret_fn=getpass.getpass, client_factory=make_client):
    print("Paste the values from the Cloudflare dashboard (see docs/R2-SETUP.md).")
    cfg = {"account_id": input_fn("Account ID: ").strip(),
           "bucket": input_fn("Bucket name: ").strip(),
           "access_key_id": input_fn("Access Key ID: ").strip(),
           "secret_access_key": secret_fn("Secret Access Key (hidden): ").strip()}
    if not all(cfg.values()):
        raise RuntimeError("all four values are required")
    test_connection(client_factory(cfg), cfg["bucket"])
    print("Test upload OK.")
    print("Saved to " + save_config(cfg))
    return cfg


def key_prefix(manifest, hostname=None):
    return "bundles/%s/%s/%s/" % (hostname or socket.gethostname(),
                                  manifest["session"], manifest["slice"])


def upload(bundle_dir, dry_run=False, progress=None, client=None, cfg=None, hostname=None):
    """Upload a built bundle folder.

    Returns {"uploaded": [...], "skipped": [...], "prefix": str}. progress(rel, status, size)
    is called per file; status is "upload", "skip", or "would-upload" (dry run).
    Files already present with the same size are skipped. index.json goes last.
    """
    with open(os.path.join(bundle_dir, "bundle-manifest.json"), encoding="utf-8") as fh:
        bm = json.load(fh)
    prefix = key_prefix(bm, hostname)
    bucket = None
    if not dry_run:
        cfg = cfg or load_config()
        bucket = cfg["bucket"]
        client = client or make_client(cfg)
        from boto3.s3.transfer import TransferConfig
        tc = TransferConfig(multipart_threshold=8 << 20, multipart_chunksize=8 << 20,
                            max_concurrency=4)
    res = {"uploaded": [], "skipped": [], "prefix": prefix}
    rels = [f["path"] for f in bm["files"] if f["path"] != "bundle-manifest.json"]
    rels.append("bundle-manifest.json")
    for rel in rels:
        path = os.path.join(bundle_dir, *rel.split("/"))
        size = os.path.getsize(path)
        key = prefix + rel
        if dry_run:
            res["uploaded"].append(rel)
            if progress:
                progress(rel, "would-upload", size)
            continue
        same = False
        try:
            same = client.head_object(Bucket=bucket, Key=key)["ContentLength"] == size
        except Exception:
            pass
        if same:
            res["skipped"].append(rel)
            if progress:
                progress(rel, "skip", size)
            continue
        client.upload_file(path, bucket, key, Config=tc)
        res["uploaded"].append(rel)
        if progress:
            progress(rel, "upload", size)
    if not dry_run:
        index = dict(bm, uploaded_by=hostname or socket.gethostname(), prefix=prefix)
        client.put_object(Bucket=bucket, Key=prefix + "index.json",
                          Body=json.dumps(index, indent=2).encode(),
                          ContentType="application/json")
    return res


def _print_progress(rel, status, size):
    print("%-13s %10d  %s" % (status, size, rel))


def main(argv=None):
    from . import bundle
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--configure" in argv:
        try:
            configure()
        except Exception as e:
            print("configure failed: %s" % e, file=sys.stderr)
            return 1
        return 0
    p = argparse.ArgumentParser(prog="understudy upload")
    bundle.add_slice_args(p)
    p.add_argument("--configure", action="store_true", help="enter and test R2 credentials")
    p.add_argument("--dry-run", action="store_true", help="list what would be sent")
    a = p.parse_args(argv)
    tmp = tempfile.mkdtemp(prefix="understudy-upload-")
    try:
        d = bundle.build(a.session, a.t_start, a.t_end, a.no_audio, a.no_frames,
                         a.max_frames, os.path.join(tmp, "bundle"))
        try:
            res = upload(d, a.dry_run, _print_progress)
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            return 1
        print("%s%d uploaded, %d skipped -> %s" % ("(dry run) " if a.dry_run else "",
              len(res["uploaded"]), len(res["skipped"]), res["prefix"]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0
