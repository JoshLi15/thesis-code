import pandas as pd


input_file = "vox2_professions_reordered.csv"
output_file = "vox2_professions_sub_actor.csv"
sep = "\t"

subclass_priority = [
    ("actor", "comedian"),
    ("television actor", "comedian"),
    ("film actor", "comedian"),
    ("actor", "singer"),
    ("television actor", "singer"),
    ("film actor", "singer"),
    ("actor", "writer"),
    ("television actor", "writer"),
    ("film actor", "writer"), 
    ("actor", "director"),
    ("television actor", "director"),
    ("film actor", "director"),
    ("actor", "screenwriter"),
    ("television actor", "screenwriter"),
    ("film actor", "screenwriter"),
    ("actor", "producer"),
    ("television actor", "producer"),
    ("film actor", "producer"),
    ("actor", "film producer"),
    ("television actor", "film producer"),
    ("film actor", "film producer"),
    ("actor", "television producer"),
    ("television actor", "television producer"),
    ("film actor", "television producer"),
    ("actor", "model"),
    ("television actor", "model"),
    ("film actor", "model"),
]

subclass_labels = {
    ("actor", "comedian"): "actor-comedian",
    ("television actor", "comedian"): "actor-comedian",
    ("film actor", "comedian"): "actor-comedian",
    ("actor", "singer"): "actor-singer",
    ("television actor", "singer"): "actor-singer",
    ("film actor", "singer"): "actor-singer",
    ("actor", "writer"): "actor-writer",
    ("television actor", "writer"): "actor-writer",
    ("film actor", "writer"): "actor-writer",
    ("actor", "director"): "actor-director",
    ("television actor", "director"): "actor-director",
    ("film actor", "director"): "actor-director",
    ("actor", "screenwriter"): "actor-screenwriter",
    ("television actor", "screenwriter"): "actor-screenwriter",
    ("film actor", "screenwriter"): "actor-screenwriter",
    ("actor", "producer"): "actor-producer",
    ("television actor", "producer"): "actor-producer",
    ("film actor", "producer"): "actor-producer",
    ("actor", "film producer"): "actor-producer",
    ("television actor", "film producer"): "actor-producer",
    ("film actor", "film producer"): "actor-producer",
    ("actor", "television producer"): "actor-producer",
    ("television actor", "television producer"): "actor-producer",
    ("film actor", "television producer"): "actor-producer",
    ("actor", "model"): "actor-model",
    ("television actor", "model"): "actor-model",
    ("film actor", "model"): "actor-model",
}

def apply_actor_subclass(prof_string):
    professions = [p.strip() for p in prof_string.split(",")]
    prof_set = set(p.lower() for p in professions)
    for combo in subclass_priority:
        if all(term in prof_set for term in combo):
            new_label = subclass_labels[combo]
            if new_label not in professions:
                return new_label
    # only return first profession if no subclass matches
    first_prof = professions[0].lower()
    return first_prof

def process_file(input_path, output_path, sep="\t"):
    df = pd.read_csv(input_path, sep=sep)

    df["Professions"] = df["Professions"].apply(apply_actor_subclass)
    df.to_csv(output_path, sep=sep, index=False)
    print(f"saved: {output_path}")

if __name__ == "__main__":
    process_file(input_file, output_file, sep)
