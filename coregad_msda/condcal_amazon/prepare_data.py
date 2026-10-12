"""Download and verify the Reddit and Amazon archives in the final protocol."""
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile


FILES = {
    "reddit.pt.zip": {
        "url": "https://raw.githubusercontent.com/pygod-team/data/main/reddit.pt.zip",
        "sha256": "276432d36219349b7ad98e974649f73c94e124e11f5282f53b1ec41086c1e18a",
    },
    "FraudAmazon.zip": {
        "url": "https://data.dgl.ai/dataset/FraudAmazon.zip",
        "sha256": "c1b7a1f8cd8621467d5121b690fef61a7d398e4f582232bb3e8a154424f7a5b9",
    },
}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    root = Path(__file__).resolve().parent / "data"
    root.mkdir(parents=True, exist_ok=True)
    provenance = []
    for filename, specification in FILES.items():
        path = root / filename
        if not path.exists():
            print("Downloading", filename, flush=True)
            with urllib.request.urlopen(specification["url"], timeout=300) as response:
                path.write_bytes(response.read())
        actual = digest(path)
        if actual != specification["sha256"]:
            raise RuntimeError(f"Checksum mismatch for {filename}: {actual}")
        provenance.append({
            "filename": filename,
            "url": specification["url"],
            "sha256": actual,
            "bytes": path.stat().st_size,
        })
    with zipfile.ZipFile(root / "reddit.pt.zip") as archive:
        (root / "reddit.pt").write_bytes(archive.read("reddit.pt"))
    amazon_dir = root / "amazon_raw"
    amazon_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(root / "FraudAmazon.zip") as archive:
        (amazon_dir / "Amazon.mat").write_bytes(archive.read("Amazon.mat"))
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print("All archives verified and extracted.")


if __name__ == "__main__":
    main()
