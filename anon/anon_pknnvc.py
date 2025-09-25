import csv
import os
from pathlib import Path
import time

# see https://github.com/carlosfranzreb/private_knnvc for further details
from demo_model import Converter

import torch
import torchaudio

# IMPORTANT:
# + this is an example script, paths need to be adapted!

CSV_PATH = Path("/path/all_waves_to_anonymize.csv")
OUT_BASE = Path("/path/data/anon_private_knn_vc")
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

p_knn_vc = None
device = None
SAMPLE_RATE = FALLBACK_SR

def derive_output_path(in_path: str) -> Path:
    """
    <speaker_id>/<utterance_id>/<clip_id>.wav
    -> /path/data/anon_private_knn_vc/<speaker_id>/<utterance_id>/<clip_id>.wav
    """
    p = Path(in_path)
    speaker_id = p.parent.parent.name  # <speaker_id>
    utterance_id = p.parent.name       # <utterance_id>
    clip_id = p.stem                   # <clip_id>
    return OUT_BASE / speaker_id / utterance_id / f"{clip_id}.wav"


def load_model():
    global p_knn_vc, device, SAMPLE_RATE

    device = torch.device("cuda:0" if (USE_GPU and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}", flush=True)

    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)  # focus on GPU

    # use the pknn-vc model
    p_knn_vc = Converter(device=str(device), dur_w=0.3, n_phone_clusters=16)

    sr_attr = getattr(p_knn_vc, "sample_rate", None) or getattr(p_knn_vc, "sr", None)
    SAMPLE_RATE = int(sr_attr) if sr_attr else FALLBACK_SR
    print(f"Sample-Rate: {SAMPLE_RATE}", flush=True)

def anonymize_file(in_path: str):
    global p_knn_vc, SAMPLE_RATE

    out_path = derive_output_path(in_path)
    if out_path.exists():
        return (in_path, "skipped", None)

    tmp_path = out_path.with_name(out_path.stem + ".part.wav")

    try:
        with torch.inference_mode():
            waveform, sr = torchaudio.load(SOURCE_BASE + in_path)

            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            if sr != SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
                sr = SAMPLE_RATE

            waveform = waveform.to(device, non_blocking=True)
            audio_anon = p_knn_vc.run(waveform, 2)

        if not isinstance(audio_anon, torch.Tensor):
            audio_anon = torch.as_tensor(audio_anon)

        if audio_anon.dim() == 1:
            audio_anon = audio_anon.unsqueeze(0)
        if audio_anon.dtype != torch.float32:
            audio_anon = audio_anon.float()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(tmp_path), audio_anon.cpu(), SAMPLE_RATE, format="wav")
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
    load_model()
    print("Loaded Model", flush=True)

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
            if failed > 0:
                print(f"Letzter Fehler: {in_path} -> {err}", flush=True)
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