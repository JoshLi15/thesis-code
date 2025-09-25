import csv
import os
from pathlib import Path

import torch
import torchaudio


# IMPORTANT:
# + this is an example script, paths need to be adapted!

CSV_PATH = Path("/path/set_splits/all_waves_to_anonymize.csv")
OUT_BASE = Path("/path/data/anon_sa_1069")
SOURCE_BASE = "/path/audio/VoxCeleb2/data/"
TARGET_ID = "1069"
PRINT_EVERY = 1000
USE_GPU_IF_AVAILABLE = True

model = None

def derive_output_path(in_path: str) -> Path:
    """
    <speaker_id>/<utterance_id>/<clip_id>.wav
    -> /path/data/anon_sa_1069/<speaker_id>/<utterance_id>/<clip_id>.wav
    """
    p = Path(in_path)
    speaker_id = p.parent.parent.name  # <speaker_id>
    utterance_id = p.parent.name       # <utterance_id>
    clip_id = p.stem                   # <clip_id>
    return OUT_BASE / speaker_id / utterance_id / f"{clip_id}.wav"


def load_model(device: torch.device):
    model = torch.hub.load(
        "deep-privacy/SA-toolkit",
        "anonymization",
        trust_repo=True
    )
    try:
        model.eval()
    except Exception:
        pass
    try:
        model.to(device)
    except Exception:
        pass
    return model

def main():
    device = torch.device("cuda" if (USE_GPU_IF_AVAILABLE and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device} | CUDA available: {torch.cuda.is_available()}", flush=True)

    print("Load Model", flush=True)
    model = load_model(device)
    print("Model loaded", flush=True)

    total = 0
    processed = 0
    skipped = 0
    failed = 0
    failures = []

    with open(CSV_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        rows = rows[600000:]

    print(f"{len(rows)} Entries loaded from CSV: {CSV_PATH}", flush=True)

    for idx, row in enumerate(rows, start=1):
        in_path = row.get("path", "").strip()
        if not in_path:
            failed += 1
            failures.append((f"(empty path {idx})", "Empty path in CSV"))
            continue

        total += 1
        out_path = derive_output_path(in_path)

        if out_path.exists():
            skipped += 1
            if total % PRINT_EVERY == 0:
                print(f"{total} seen | {processed} anonymised | {skipped} skipped | {failed} failed", flush=True)
            continue

        try:
            waveform, sr = torchaudio.load(SOURCE_BASE + in_path)

            wf = waveform.to(device)

            with torch.no_grad():
                anon_waveform = model.convert(wf, target=TARGET_ID)

            out_path.parent.mkdir(parents=True, exist_ok=True)

            tmp_path = out_path.with_name(out_path.stem + ".part.wav")

            torchaudio.save(
                str(tmp_path),
                anon_waveform.detach().cpu(),
                sr,
                format="wav"
            )
            os.replace(tmp_path, out_path)

            processed += 1

        except Exception as e:
            failed += 1
            msg = f"{type(e).__name__}: {e}"
            print(msg, flush=True)
            failures.append((in_path, msg))
            try:
                tmp_path = out_path.with_name(out_path.stem + ".part.wav")
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

        if total % PRINT_EVERY == 0:
            print(f"{total} seen | {processed} anonymised | {skipped} skipped | {failed} failed", flush=True)

    print("\n===== Summary =====", flush=True)
    print(f"Total Entries      : {total}", flush=True)
    print(f"Anonymised         : {processed}", flush=True)
    print(f"Skipped (exists): {skipped}", flush=True)
    print(f"Failed       : {failed}", flush=True)

    if failures:
        report_path = OUT_BASE / "failed_list.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as rf:
            for pth, err in failures:
                rf.write(f"{pth}\t{err}\n")
        print("First 5 Errors:")
        for pth, err in failures[:5]:
            print(f"  - {pth}: {err}", flush=True)


if __name__ == "__main__":
    main()
