import csv
import os
from pathlib import Path
import time

import torch
import torchaudio

# IMPORTANT:
# + this is an example script, paths need to be adapted!

CSV_PATH = Path("/path/all_waves_to_anonymize.csv")
OUT_BASE = Path("/path/data/anon_knn_vc")
REF_WAV_PATHS = ["path/VoxxCeleb2/data/id00012/21Uxsk56VDQ/00001.wav",
        "path/VoxCeleb2/data/id00012/21Uxsk56VDQ/00002.wav",
        "path/VoxCeleb2/data/id00012/21Uxsk56VDQ/00003.wav",
        "path/VoxCeleb2/data/id00012/21Uxsk56VDQ/00004.wav",
        "path/VoxCeleb2/data/id00012/21Uxsk56VDQ/00005.wav",
        "path/VoxCeleb2/data/id00012/21Uxsk56VDQ/00006.wav",
        "path/VoxCeleb2/data/id00012/21Uxsk56VDQ/00007.wav",
        "path/VoxCeleb2/data/id00012/21Uxsk56VDQ/00008.wav",
        "path/VoxCeleb2/data/id00012/21Uxsk56VDQ/00009.wav",
        "path/VoxCeleb2/data/id00012/21Uxsk56VDQ/00010.wav",
        "path/VoxCeleb2/data/id00012/21Uxsk56VDQ/00011.wav",
        "path/VoxCeleb2/data/id00012/21Uxsk56VDQ/00012.wav",
        "path/VoxCeleb2/data/id00012/21Uxsk56VDQ/00013.wav",
        "path/VoxCeleb2/data/id00012/21Uxsk56VDQ/00014.wav",
        "path/VoxCeleb2/data/id00012/21Uxsk56VDQ/00015.wav",
        "path/VoxCeleb2/data/id00012/2DLq_Kkc1r8/00016.wav",
        "path/VoxCeleb2/data/id00012/2DLq_Kkc1r8/00017.wav",
        "path/VoxCeleb2/data/id00012/2DLq_Kkc1r8/00018.wav",
        "path/VoxCeleb2/data/id00012/73OrGYvy4ng/00019.wav",
        "path/VoxCeleb2/data/id00012/aE4Om0EEiuk/00115.wav",
        "path/VoxCeleb2/data/id00012/aE4Om0EEiuk/00116.wav",
        "path/VoxCeleb2/data/id00012/aE4Om0EEiuk/00117.wav",
        "path/VoxCeleb2/data/id00012/aE4Om0EEiuk/00118.wav",
        "path/VoxCeleb2/data/id00012/aE4Om0EEiuk/00119.wav",
        "path/VoxCeleb2/data/id00012/aE4Om0EEiuk/00120.wav"]

SOURCE_BASE = "/path/audio/VoxCeleb2/data/"
PRINT_EVERY = 1000
TOPK = 4 # nbr neighbours
USE_GPU = True
FALLBACK_SR = 16000

knn_vc = None
device = None
matching_set = None
SAMPLE_RATE = FALLBACK_SR

def derive_output_path(in_path: str) -> Path:
    """
    <speaker_id>/<utterance_id>/<clip_id>.wav
    -> path/data/anon_knn_vc/<speaker_id>/<utterance_id>/<clip_id>.wav
    """
    p = Path(in_path)
    speaker_id = p.parent.parent.name  # <speaker_id>
    utterance_id = p.parent.name       # <utterance_id>
    clip_id = p.stem                   # <clip_id>
    return OUT_BASE / speaker_id / utterance_id / f"{clip_id}.wav"


def load_model_and_matching_set():
    global knn_vc, device, matching_set, SAMPLE_RATE

    device = torch.device("cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}", flush=True)

    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)  # focus on GPU

    print("Load knn_vc…", flush=True)
    knn_vc = torch.hub.load(
        'bshall/knn-vc', 'knn_vc',
        prematched=True, trust_repo=True, pretrained=True
    )
    try: knn_vc.eval()
    except Exception: pass
    try: knn_vc.to(device)
    except Exception: pass

    # get sample rate
    sr_attr = getattr(knn_vc, "sample_rate", None) or getattr(knn_vc, "sr", None)
    SAMPLE_RATE = int(sr_attr) if sr_attr else FALLBACK_SR
    print(f"Sample-Rate: {SAMPLE_RATE}", flush=True)

    print("Building Matching-Set", flush=True)
    with torch.inference_mode():
        matching_set = knn_vc.get_matching_set(REF_WAV_PATHS)
    print(f"Matching-Set done building (N={len(REF_WAV_PATHS)})", flush=True)

def anonymize_file(in_path: str):
    global knn_vc, matching_set, SAMPLE_RATE

    out_path = derive_output_path(in_path)
    if out_path.exists():
        return (in_path, "skipped", None)

    tmp_path = out_path.with_name(out_path.stem + ".part.wav")

    try:
        with torch.inference_mode():
            query_seq = knn_vc.get_features(SOURCE_BASE + in_path)           # liest/encodiert Audio
            anon_waveform = knn_vc.match(query_seq, matching_set, topk=TOPK)

        # torchaudio.save expects (channels, frames) & float32
        if anon_waveform.dtype != torch.float32:
            anon_waveform = anon_waveform.float()
        if anon_waveform.dim() == 1:
            anon_waveform = anon_waveform.unsqueeze(0)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(tmp_path), anon_waveform.cpu(), SAMPLE_RATE, format="wav")
        os.replace(tmp_path, out_path)
        return (in_path, "processed", None)

    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        try:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return (in_path, "failed", msg)

    

def main():
    print("Start Anonymising", flush=True)
    load_model_and_matching_set()
    print("Loaded Model and Matching-Set", flush=True)

    with open(CSV_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        paths = [row.get("path", "").strip() for row in reader if row.get("path", "").strip()]
        paths = paths[:600000]
    total = len(paths)
    print(f"Loaded {total} CSV Entries: {CSV_PATH}", flush=True)

    processed = 0
    skipped = 0
    failed = 0
    failures = []
    last_time = time.time()

    for idx, pth in enumerate(paths, start=1):
        in_path, status, err = anonymize_file(pth)
        if status == "processed":
            processed += 1
        elif status == "skipped":
            skipped += 1
        else:
            failed += 1
            failures.append((in_path, err))

        if idx % PRINT_EVERY == 0 or idx == total:
            elapsed = time.time() - last_time
            print(
                f"PROGRESS {idx}/{total} | "
                f"verarbeitet: {processed} | "
                f"skipped: {skipped} | "
                f"failed: {failed} | "
                f"time passed: {elapsed:.2f}s",
                flush=True
            )
            last_time = time.time()

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