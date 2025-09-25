import pandas as pd

input_file = "vox1_professions_sub_actor.csv"
output_file = "vox1_professions_merged.csv"
sep = "\t"

MERGE_MAP = {
    # Action and enterainment
    "actor": "actor",
    "film actor": "actor",
    "television actor": "actor",
    "voice actor": "actor",
    "character actor": "actor",
    "child actor": "actor",
    "dub actor": "actor",
    "pornographic actor": "actor",
    "musical theatre actor": "actor-singer",
    "stage actor": "stage actor",
    "actor-comedian": "actor-comedian",
    "comedian": "actor-comedian",
    "entertainer": "actor-comedian",
    "actor-singer": "actor-singer",
    "singer": "singer",
    "playback singer": "singer",
    "opera singer": "singer",
    "pop singer": "singer",
    "jazz singer": "singer",
    "singer-songwriter": "singer",
    "rapper": "singer",
    "fado singer": "singer",
    "actor-model": "actor-model",
    "model": "model",
    "supermodel": "model",
    "beauty pageant contestant": "model",
    "art model": "model",
    "playboy playmate": "model",
    "actor-writer": "actor-writer",
    "writer": "writer",
    "author": "writer",
    "poet": "writer",
    "autobiographer": "writer",
    "playwright": "actor-writer",
    "actor-director": "director",
    "film director": "director",
    "director": "director",
    "television director": "director",
    "documentary filmmaker": "director",
    "actor-producer": "actor-producer",
    "film producer": "actor-producer",
    "television producer": "actor-producer",
    "actor-screenwriter": "actor-screenwriter",
    "screenwriter": "actor-screenwriter",

    # Music
    "musician": "musician",
    "academic musician": "musician",
    "conductor": "musician",
    "bassist": "musician",
    "guitarist": "musician",
    "drummer": "musician",
    "violinist": "musician",
    "jazz musician": "musician",
    "classical composer": "musician",
    "academic musician": "musician",
    "classical pianist": "musician",
    "composer": "audio_producer",
    "disc jockey": "audio_producer",
    "dj": "audio_producer",
    "record producer": "audio_producer",
    "songwriter": "audio_producer",
    "schlager singer": "singer",

    # Politics and Public Affairs
    "politician": "politician",
    "mayor": "politician",
    "local politician": "politician",
    "chancellor of germany": "politician",
    "statesperson": "politician",
    "diplomat": "politician",
    "vice president": "politician",
    "lawyer": "legal_professional",
    "judge": "legal_professional",
    "solicitor": "legal_professional",
    "jurist": "legal_professional",
    "journalist": "media_professional",
    "news presenter": "media_professional",
    "television presenter": "media_professional",
    "radio personality": "media_professional",
    "correspondent": "media_professional",
    "columnist": "media_professional",
    "weather presenter": "media_professional",
    "war correspondent": "media_professional",
    "television personality": "media_professional",
    "spokesperson": "media_professional",

    # Sports
    "association football player": "football_player",
    "gridiron football player": "football_player",
    "association football manager": "football_manager",

    "basketball player": "basketball",
    "basketball": "basketball",
    "basketball coach": "basketball",

    "baseball player": "baseball",
    "professional baseball player": "baseball",
    "baseball coach": "baseball",
    "rugby union player": "rugby",
    "rugby league player": "rugby",

    "cricketer": "cricket",
    "tennis player": "tennis",
    "ice hockey player": "ice_hockey",
    "ice hockey coach": "ice_hockey",

    "american football player": "american_football",
    "american football coach": "american_football",

    "boxer": "combat_sport",
    "judoka": "combat_sport",
    "mma fighter": "combat_sport",
    "wrestler": "combat_sport",
    "amateur wrestler": "combat_sport",
    "professional wrestler": "combat_sport",
    "thai boxer": "combat_sport",
    "mixed martial arts fighter": "combat_sport",
    "kickboxer": "combat_sport",
    "karateka": "combat_sport",
    "jeet kune do": "combat_sport",

    "athlete": "other_athlete",
    "sprinter": "other_athlete",
    "swimmer": "other_athlete",
    "triathlete": "other_athlete",
    "cyclist": "other_athlete",
    "golfer": "other_athlete",
    "gymnast": "other_athlete",
    "archer": "other_athlete",
    "snooker player": "other_athlete",
    "military athlete": "other_athlete",
    "freestyle skier": "other_athlete",
    "squash player": "other_athlete",
    "volleyball player": "other_athlete",
    "table tennis player": "other_athlete",
    "sport shooter": "other_athlete",
    "sport cyclist": "other_athlete",
    "surfer": "other_athlete",
    "para athletics competitor": "other_athlete",
    "para swimmer": "other_athlete",
    "fencer": "other_athlete",
    "field hockey player": "other_athlete",
    "darts player": "other_athlete",
    "badminton player": "other_athlete",
    "dancer": "other_athlete",
    "ballet dancer": "other_athlete",
    "netballer": "other_athlete",
    "middle-distance runner": "other_athlete",
    "long-distance runner": "other_athlete",
    "handball player": "other_athlete",
    "cue sports player": "other_athlete",
    "dressage rider": "other_athlete",
    "canoeist": "other_athlete",
    "bull rider": "other_athlete",
    "bodybuilder": "other_athlete",
    "australian rules football player": "other_athlete",
    "athletics competitor": "other_athlete",
    "artistic gymnast": "other_athlete",
    "weightlifter": "other_athlete",
    "rower": "other_athlete",
    "rider": "other_athlete",
    "professional golfer": "other_athlete",
    "skateboarder": "other_athlete",
    "strongman": "other_athlete",
    "orienteer": "other_athlete",
    "hurdler": "other_athlete",


    "skeleton racer": "winter_athlete",
    "snowboarder": "winter_athlete",
    "speed skater": "winter_athlete",
    "skier": "winter_athlete",
    "ski jumper": "winter_athlete",
    "alpine skier": "winter_athlete",
    "skeleton racer": "winter_athlete",
    "short-track speed skater": "winter_athlete",
    "cross-country skier": "winter_athlete",
    "figure skater": "winter_athlete",
    "curler": "winter_athlete",
    "biathlete": "winter_athlete",

    "sports official": "sports_professional",
    "sports executive": "sports_professional",
    "sports commentator": "sports_professional",
    "sports analyst": "sports_professional",
    "sports agent": "sports_professional",
    "sporting director": "sports_professional",
    "official": "sports_professional",
    "general manager": "sports_professional",
    "coach": "sports_professional",
    "baseball manager": "sports_professional",
    "association football referee": "sports_professional",
    "head coach": "sports_professional",
    "horse trainer": "sports_professional",
    "announcer": "sports_professional",


    "racing automobile driver": "racing_driver",
    "co-driver": "racing_driver",
    "formula one driver": "racing_driver",
    "racing driver": "racing_driver",
    "motorcycle racer": "racing_driver",
    "rally driver": "racing_driver",
    "racing automobile driver": "racing_driver",
    "skipper": "racing_driver",
    "sailor": "racing_driver",
    "jockey": "racing_driver",



    # Knowledge & Academia
    "researcher": "academic",
    "scientist": "academic",
    "astrophysicist": "academic",
    "botanist": "academic",
    "mathematician": "academic",
    "historian": "academic",
    "professor": "academic",
    "literary scholar": "academic",
    "computer scientist": "academic",
    "theologian": "academic",
    "teacher": "academic",
    "research assistant": "academic",
    "rabbi": "academic",
    "pundit": "academic",
    "professor of mathematics": "academic",
    "priest": "academic",
    "physician writer": "academic",
    "pastor": "academic",
    "meteorologist": "academic",
    "librarian": "academic",
    "economist": "academic",
    "catholic priest": "academic",
    "bishop": "academic",
    "archbishop": "academic",
    "astronaut": "academic",
    "naturalist": "academic",


    # Medical & Technical
    "engineer": "medical_technical_professional",
    "train driver": "medical_technical_professional",
    "carpenter": "medical_technical_professional",
    "railway worker": "medical_technical_professional",
    "physician": "medical_technical_professional",
    "psychiatrist": "medical_technical_professional",
    "pediatrician": "medical_technical_professional",

    # Business & Entrepreneurship
    "businessperson": "business",
    "entrepreneur": "business",
    "chief executive officer": "business",
    "manager": "business",
    "business executive": "business",
    "real estate entrepreneur": "business",
    "trader": "business",
    "merchant": "business",
    "banker": "business",
    "business manager": "business",
    "financier": "business",
    "executive board": "business",

    # "chef": "culinary_professional",
    # "pastry chef": "culinary_professional",
    # "restaurateur": "culinary_professional",
    # "celebrity chef": "culinary_professional",
    # "cook": "culinary_professional",
    # "fashion designer": "designer",
    # "interior designer": "designer",
    # "creative director": "designer",
    # "costume designer": "designer",
    # "designer": "designer",
    # "fashion": "designer",
}

unmapped_professions = set()

def merge_professions(prof_string: str) -> str:
    professions = [p.strip().lower() for p in prof_string.split(",")]
    merged = set()
    for prof in professions:
        merged_prof = MERGE_MAP.get(prof, "other")
        if merged_prof == "other":
            unmapped_professions.add(prof)
        merged.add(merged_prof)
    return ",".join(sorted(merged))

def process_file(input_path, output_path, sep="\t"):
    df = pd.read_csv(input_path, sep=sep)

    df["Professions"] = df["Professions"].apply(merge_professions)
    df.to_csv(output_path, sep=sep, index=False)
    print(f"Saved: {output_path}")

    if unmapped_professions:
        print("\nUnmapped Professions:")
        for prof in sorted(unmapped_professions):
            print(f"- {prof}")
    else:
        print("Everything mapped!")

if __name__ == "__main__":
    process_file(input_file, output_file, sep)
