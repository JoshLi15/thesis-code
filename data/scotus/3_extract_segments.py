import os
import json
import pickle
import datetime
import argparse
from pydub import AudioSegment
from tqdm import tqdm
import csv


# Attribution: https://github.com/hechmik/voxceleb_enrichment_age_gender/blob/main/README.md


def load_dobs(dob_path):
    with open(dob_path, 'rb') as f:
        return pickle.load(f)


def load_transcripts(transcript_folder):
    return sorted([os.path.join(transcript_folder, f) for f in os.listdir(transcript_folder) if f.endswith('.json')])


def segments_overlap(seg1, seg2):
    return not (seg1['stop'] <= seg2['start'] or seg2['stop'] <= seg1['start'])


def remove_overlapping_segments(utts):
    utts = sorted(utts, key=lambda x: x['start'])
    clean_utts = []
    last_stop = -1
    for u in utts:
        if u['start'] >= last_stop:
            clean_utts.append(u)
            last_stop = u['stop']
    return clean_utts


def split_long_segments(utts, min_len=4.0, max_len=10.0):
    new_utts = []
    for utt in utts:
        dur = utt['stop'] - utt['start']
        if dur <= max_len:
            if dur >= min_len:
                new_utts.append(utt)
        else:
            num_full = int(dur // max_len)
            for i in range(num_full):
                new_utts.append({
                    'start': utt['start'] + i * max_len,
                    'stop': utt['start'] + (i + 1) * max_len,
                    'speaker_id': utt['speaker_id']
                })
            remaining_start = utt['start'] + num_full * max_len
            if utt['stop'] - remaining_start >= min_len:
                new_utts.append({
                    'start': remaining_start,
                    'stop': utt['stop'],
                    'speaker_id': utt['speaker_id']
                })
    return new_utts


def calculate_age_days(dob, rec_date):
    if isinstance(dob, int):
        dob = datetime.datetime.fromtimestamp(dob)
    return abs((rec_date - dob).days)


def extract_segments(args):
    dobs = load_dobs(os.path.join(args.base_outfolder, 'dob.p'))
    transcripts = load_transcripts(
        os.path.join(args.base_outfolder, 'transcripts'))
    audio_folder = os.path.join(args.base_outfolder, 'audio')
    output_audio_folder = os.path.join(args.base_outfolder, 'ecapa_segments')
    os.makedirs(output_audio_folder, exist_ok=True)
    output_csv = os.path.join(args.base_outfolder, 'ecapa_segments.csv')

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['segment_id', 'speaker_id', 'age_days', 'wav_path'])

        for transcript_file in tqdm(transcripts):
            with open(transcript_file, 'r', encoding='utf-8') as f:
                js = json.load(f)
            utts = js['utts']
            utts = [u for u in utts if u['speaker_id']
                    in dobs and dobs[u['speaker_id']] is not None]
            utts = sorted(utts, key=lambda x: x['start'])
            utts = remove_overlapping_segments(utts)
            utts = split_long_segments(utts)

            rec_name = os.path.splitext(os.path.basename(transcript_file))[0]
            case_date = datetime.datetime.strptime(js['case_date'], '%Y/%m/%d')
            audio_path = os.path.join(audio_folder, f'{rec_name}.mp3')
            if not os.path.isfile(audio_path):
                print(f"Missing audio file: {audio_path}")
                continue
            audio = AudioSegment.from_file(audio_path)

            for i, utt in enumerate(utts):
                segment_id = f"{rec_name}_{i:04d}"
                start_ms = int(utt['start'] * 1000)
                stop_ms = int(utt['stop'] * 1000)
                segment_audio = audio[start_ms:stop_ms]
                segment_path = os.path.join(
                    output_audio_folder, f"{segment_id}.wav")
                segment_audio.export(segment_path, format='wav')

                speaker_id = utt['speaker_id']
                age_days = calculate_age_days(dobs[speaker_id], case_date)

                writer.writerow(
                    [segment_id, speaker_id, age_days, segment_path])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_outfolder', type=str, required=True,
                        help='Base folder with audio/, transcripts/, dob.p')
    args = parser.parse_args()
    extract_segments(args)
