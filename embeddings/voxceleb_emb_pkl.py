import torch
import torchaudio
import numpy as np
from speechbrain.inference import SpeakerRecognition

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import pickle


# IMPORTANT:
# + this is an example script, paths need to be adapted!
# + this script is for SA-Anonymized VoxCeleb data, for the original data or other 
#   anonymisations minor changes might be needed


def create_embeddings():
    input_dir = "/path/data/anon_sa_1069"
    output_dir = "/path/embeddings/anon_sa_1069"
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/ecapa",
        run_opts={"device": device}
    )

    # process one clip
    def process_clip(clip, in_dir):
        if not clip.endswith(".wav"):
            return None

        clip_path = os.path.join(in_dir, clip)
        clip_name = os.path.splitext(clip)[0]
        parts = in_dir.split(os.sep)
        speaker_id = parts[-2]
        utterance_id = parts[-1]
        out_path = os.path.join(output_dir, speaker_id, utterance_id, f"{clip_name}.npy")
        
        if os.path.exists(out_path):
            return True
        try:
            signal, fs = torchaudio.load(clip_path)
            if fs != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
                signal = resampler(signal)

            # Move to correct device
            signal = signal.to(device)

            # Run model
            emb = model.encode_batch(signal).squeeze().detach().cpu().numpy()

            # Save to .npy
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.save(out_path, emb)
            return True

        except Exception as e:
            print(f"[ERROR] Failed on {clip}: {e}")
            return False
        
    # get all .wav files
    all_clips = []
    # "/path/data/anon_sa_1069/<speaker_id>/<utterance_id>/<clip>.wav"
    for speaker_id in os.listdir(input_dir):
        if not speaker_id.startswith("id"):
            continue
        for utterance_id in os.listdir(os.path.join(input_dir, speaker_id)):
            in_dir = os.path.join(input_dir, speaker_id, utterance_id)
            for clip in os.listdir(in_dir):
                if clip.endswith(".wav"):
                    all_clips.append((clip, in_dir))
    print(f"Found {len(all_clips)} .wav files.", flush=True)

    success_count = 0
    processed = 0
    total = len(all_clips)

    # process parallel (I/O-bound, so threads are safe)
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(process_clip, toupl[0], toupl[1]): toupl for toupl in all_clips}
        for future in as_completed(futures):
            result = future.result()
            processed += 1
            if result:
                success_count += 1
            if processed % 1000 == 0 or processed == total:
                print(f"Progress: {processed}/{total} processed, {success_count} successful", flush=True)

    print(f"\nsuccessfully processed {success_count}/{len(all_clips)} WAV files.")
    print(f"embeddings saved to: {output_dir}", flush=True)

def create_split_pkls():
    label_file = "/path/set_splits/vox2_speaker_sets.csv"
    embedding_dir = "/path/embeddings/anon_sa_1069"
    output_dir_gender = "/path/embeddings/anon_sa_1069_split_pkls/gender"
    output_dir_nationality = "/path/embeddings/anon_sa_1069_split_pkls/nationality"
    output_dir_profession = "/path/embeddings/anon_sa_1069_split_pkls/profession"

    splits_gender = {
        "train": {"features": [], "gender": [], "utt_ids": []},
        "val": {"features": [], "gender": [], "utt_ids": []},
        "test": {"features": [], "gender": [], "utt_ids": []},
    }

    splits_nationality = {
        "train": {"features": [], "nationality": [], "utt_ids": []},
        "val": {"features": [], "nationality": [], "utt_ids": []},
        "test": {"features": [], "nationality": [], "utt_ids": []},
    }

    splits_profession = {
        "train": {"features": [], "profession": [], "utt_ids": []},
        "val": {"features": [], "profession": [], "utt_ids": []},
        "test": {"features": [], "profession": [], "utt_ids": []},
    }

    label_dict = {}  # speaker_id --> (label_g, split_g,

    # load label csv
    with open(label_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            speaker_id = row["speaker_id"].strip()
            label_class_gender = row["gender"].strip()
            split_gender = row["gender_set"].strip().lower()
            label_class_nationality = row["nationality"].strip()
            split_nationality = row["nationality_set"].strip().lower()
            label_class_profession = row["profession"].strip()
            split_profession = row["profession_set"].strip().lower()
            if split_gender in ["train", "val", "test"]:
                label_dict[speaker_id] = (label_class_gender,
                                          split_gender,
                                          label_class_nationality,
                                          split_nationality,
                                          label_class_profession,
                                          split_profession)
            else:
                print(f"Skipping segment {speaker_id} with unknown split: '{split}'")

    print(f"Loaded {len(label_dict)} labeled entries from {label_file}", flush=True)

    # load embeddings
    num_found = 0
    for speaker_id in os.listdir(embedding_dir):
        if not speaker_id.startswith("id"):
            continue
        for utterance_id in os.listdir(os.path.join(embedding_dir, speaker_id)):
            in_dir = os.path.join(embedding_dir, speaker_id, utterance_id)
            for fname in os.listdir(in_dir):
                if fname.endswith(".npy"):
                    if speaker_id in label_dict:
                        try:
                            emb = np.load(os.path.join(in_dir, fname))
                        except Exception as e:
                            print(f"[ERROR] Loading embedding failed: {fname}, error: {e}")
                            continue
                        label_class_gender,split_gender,label_class_nationality,split_nationality,label_class_profession,split_profession = label_dict[speaker_id]
                        utt_id = f"{speaker_id}/{utterance_id}/{fname.replace('.npy', '')}"
                        # gender
                        splits_gender[split_gender]["features"].append(emb)
                        splits_gender[split_gender]["gender"].append(label_class_gender)
                        splits_gender[split_gender]["utt_ids"].append(utt_id)
                        # nationality
                        splits_nationality[split_nationality]["features"].append(emb)
                        splits_nationality[split_nationality]["nationality"].append(label_class_nationality)
                        splits_nationality[split_nationality]["utt_ids"].append(utt_id)
                        # profession
                        splits_profession[split_profession]["features"].append(emb)
                        splits_profession[split_profession]["profession"].append(label_class_profession)
                        splits_profession[split_profession]["utt_ids"].append(utt_id)
                        num_found += 1
                        if num_found % 10000 == 0:
                            print(f"Processed {num_found} embeddings", flush=True)

    print(f"Loaded {num_found} embeddings with labels, save them now", flush=True)

    # save
    os.makedirs(output_dir_gender, exist_ok=True)
    os.makedirs(output_dir_nationality, exist_ok=True)
    os.makedirs(output_dir_profession, exist_ok=True)
    for split in ["train", "val", "test"]:
        # gender
        out_path_gender = os.path.join(output_dir_gender, f"{split}.pkl")
        with open(out_path_gender, "wb") as f:
            pickle.dump(splits_gender[split], f)
        print(f"Saved {len(splits_gender[split]['features'])} items to {out_path_gender}", flush=True)
        # nationality
        out_path_nationality = os.path.join(output_dir_nationality, f"{split}.pkl")
        with open(out_path_nationality, "wb") as f:
            pickle.dump(splits_nationality[split], f)
        print(f"Saved {len(splits_nationality[split]['features'])} items to {out_path_nationality}", flush=True)
        # gender
        out_path_profession = os.path.join(output_dir_profession, f"{split}.pkl")
        with open(out_path_profession, "wb") as f:
            pickle.dump(splits_profession[split], f)
        print(f"Saved {len(splits_profession[split]['features'])} items to {out_path_profession}", flush=True)

    total = sum(len(s["features"]) for s in splits_gender.values())
    print(f"Total included samples in all splits: {total}", flush=True)

if __name__ == "__main__":
    # STEP 1: Create embeddings
    create_embeddings()
    # STEP2: Create pkl files from embeddings and labels for all 3 attributes
    create_split_pkls()
