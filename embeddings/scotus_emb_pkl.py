import torch
import torchaudio
import numpy as np
from speechbrain.inference import SpeakerRecognition

import os
import pickle
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

# IMPORTANT:
# + this is an example script, paths need to be adapted!
# + this script is for SA-Anonymized SCOTUS data, for the original data or other 
#   anonymisations minor changes might be needed

# SCOTUS is in Clips.wav, no sorting in speaker directories like in VoxCeleb!
# 2009.08-876-t01_0223.wav    2014.13-7120-t02_0107.wav   2020.65-orig-t01_0472.wav
# 2009.08-876-t01_0224.wav    2014.13-7120-t02_0108.wav

# Labels in:
# /path/scotus_segment_sets.csv
# set is train, test, val or none
# segment_id,age_bucket,set
# 2005.03-1238-t01_0000,50s,train
# 2005.03-1238-t01_0001,50s,train


def create_embeddings():
    input_dir = "/path/data/scotus_anon_sa_1069"
    output_dir = "/path/embeddings/scotus_anon_sa_1069"
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/ecapa",
        run_opts={"device": device}
    )

    # process one clip
    def process_clip(clip):
        if not clip.endswith(".wav"):
            return None
        
        clip_name = os.path.splitext(clip)[0]
        out_path = os.path.join(output_dir, f"{clip_name}.npy")

        if os.path.exists(out_path):
            return True # embedding already exists

        clip_path = os.path.join(input_dir, clip)
        try:
            signal, fs = torchaudio.load(clip_path)
            if fs != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
                signal = resampler(signal)

            # Move to correct device
            signal = signal.to(device)

            # Run model
            emb = model.encode_batch(signal).detach().cpu().squeeze()
            if emb.ndim == 2 and emb.shape[0] != 192:
                emb = emb[0]  # take first entry, if (N,192)
            emb = emb.numpy()

            # Save to .npy
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.save(out_path, emb)
            return True

        except Exception as e:
            print(f"[ERROR] Failed on {clip}: {e}", flush=True)
            return False
        
    # get all .wav files
    all_clips = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
    print(f"Found {len(all_clips)} .wav files.", flush=True)

    success_count = 0
    processed = 0
    total = len(all_clips)

    # process parallel (I/O-bound, so threads are safe)
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(process_clip, clip): clip for clip in all_clips}
        for future in as_completed(futures):
            result = future.result()
            processed += 1
            if result:
                success_count += 1
            if processed % 10000 == 0 or processed == total:
                print(f"Progress: {processed}/{total} processed, {success_count} successful", flush=True)

    print(f"\nsuccessfully processed {success_count}/{len(all_clips)} WAV files.")
    print(f"embeddings saved to: {output_dir}", flush=True)

def create_split_pkls():
    label_file = "/path/set_splits/scotus_segment_sets.csv"
    embedding_dir = "/path/embeddings/scotus_anon_sa_1069"
    output_dir = "/path/embeddings/scotus_anon_sa_1069_pkls"

    splits = {
        "train": {"features": [], "age": [], "utt_ids": []},
        "val": {"features": [], "age": [], "utt_ids": []},
        "test": {"features": [], "age": [], "utt_ids": []},
    }

    label_dict = {}  # segment_id --> (age_bucket, split)
    skipped = 0

    # load label csv
    with open(label_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            segment_id = row["segment_id"].strip()
            age_bucket = row["age_bucket"].strip()
            split = row["set"].strip().lower()
            if split in splits:
                label_dict[segment_id] = (age_bucket, split)
            else:
                skipped += 1

    print(f"Loaded {len(label_dict)} labeled entries from {label_file}", flush=True)
    if skipped > 0:
        print(f"Skipped {skipped} entries with invalid split", flush=True)

    # load embeddings
    num_found = 0
    for fname in os.listdir(embedding_dir):
        if fname.endswith(".npy"):
            segment_id = fname.replace(".npy", "")
            if segment_id in label_dict:
                emb = np.load(os.path.join(embedding_dir, fname))
                age_bucket, split = label_dict[segment_id]
                splits[split]["features"].append(emb)
                splits[split]["age"].append(age_bucket)
                splits[split]["utt_ids"].append(segment_id)
                num_found += 1
                if num_found % 10000 == 0 and num_found > 0:
                    print(f"Found {num_found} embeddings with labels so far...", flush=True)

    print(f"Loaded {num_found} embeddings with labels", flush=True)

    # save
    os.makedirs(output_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        out_path = os.path.join(output_dir, f"{split}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(splits[split], f)
        print(f"Saved {len(splits[split]['features'])} items to {out_path}", flush=True)

    total = sum(len(s["features"]) for s in splits.values())
    print(f"Total included samples in all splits: {total}", flush=True)
    

if __name__ == "__main__":
    # STEP 1: Create embeddings
    create_embeddings()
    # STEP 2: Create pkl files from embeddings and labels
    create_split_pkls()
