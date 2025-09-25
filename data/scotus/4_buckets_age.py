import pandas as pd


INPUT_FILE = "/path/metadata_segments.csv"


def get_age_bucket_from_days(age_days):
    age_years = age_days / 365.25

    if age_years <= 39:
        return "young"
    elif age_years <= 65:
        return 'mid_age'
    else:
        return "old"


def process_csv(file_path):
    df = pd.read_csv(file_path)
    df["age_bucket"] = df["age_days"].apply(get_age_bucket_from_days)
    return df


def plot_age_bucket_distribution(df):
    counts = df["age_bucket"].value_counts().sort_index()

    speaker_counts = df.groupby("age_bucket")[
        "speaker_id"].nunique().sort_index()

    print("Verteilung der Age Buckets:")
    for bucket, count in counts.items():
        print(f"{bucket}: {count}")

    print("\nVerteilung der Age Buckets (Unique Speakers):")
    for bucket, count in speaker_counts.items():
        print(f"{bucket}: {count} speakers")


if __name__ == "__main__":
    df = process_csv(INPUT_FILE)
    plot_age_bucket_distribution(df)

    column_order = ["segment_id", "speaker_id",
                    "age_bucket", "age_days", "wav_path"]
    df = df[column_order]

    df.to_csv("output_with_buckets.csv", index=False)
