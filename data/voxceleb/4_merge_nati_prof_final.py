import pandas as pd
import os

profession_mapping = {
    # 1. performance_artist
    "actor": "performance_artist",
    "actor-model": "performance_artist",
    "actor-producer": "performance_artist",
    "actor-screenwriter": "performance_artist",
    "actor-writer": "performance_artist",
    "model": "performance_artist",

    # 2. musical_artist
    "singer": "musical_artist",
    "actor-singer": "musical_artist",
    "musician": "musical_artist",

    # 3. sports
    "football_player": "sports",
    "basketball": "sports",
    "combat_sport": "sports",
    "tennis": "sports",
    "cricket": "sports",
    "rugby": "sports",
    "racing_driver": "sports",
    "american_football": "sports",
    "baseball": "sports",
    "winter_athlete": "sports",
    "ice_hockey": "sports",
    "sports_professional": "sports",
    "other_athlete": "sports",
    "football_manager": "sports",

    # 4. politics
    "politician": "politics",

    # 5. media_writer
    "writer": "media_writer",
    "media_professional": "media_writer",
    "audio_producer": "media_writer",
    "director": "media_writer",

    # 6. professionals
    "academic": "professionals",
    "legal_professional": "professionals",
    "medical_technical_professional": "professionals",
    "business": "professionals",

    # 7. other
    "other": "other",
    "actor-comedian": "other",
    "stage actor": "other",
}

country_mapping = {
    # usa_canada
    "USA": "usa_canada",
    "Canada": "usa_canada",

    # uk_similar
    "UK": "uk_similar",
    "Australia": "uk_similar",
    "Ireland": "uk_similar",
    "New Zealand": "uk_similar",
    "Scotland": "uk_similar",
    "England": "uk_similar",

    # india
    "India": "india",

    # south_eu
    "France": "south_eu",
    "Italy": "south_eu",
    "Spain": "south_eu",
    "Portugal": "south_eu",
    "Malta": "south_eu",
    "Monaco": "south_eu",

    # north_eu
    "Sweden": "north_eu",
    "Norway": "north_eu",
    "Denmark": "north_eu",
    "Finland": "north_eu",
    "Iceland": "north_eu",

    # german_similar
    "Germany": "german_similar",
    "Austria": "german_similar",
    "Switzerland": "german_similar",
    "Netherlands": "german_similar",
    "Luxembourg": "german_similar",

    # slavic
    "Russia": "slavic",
    "Ukraine": "slavic",
    "Poland": "slavic",
    "Czech-Republic": "slavic",
    "Slovakia": "slavic",
    "Slovenia": "slavic",
    "Serbia": "slavic",
    "Croatia": "slavic",
    "Romania": "slavic",
    "Bulgaria": "slavic",
    "Belarus-1": "slavic",
    "Bosnia-and-Herzegovina": "slavic",
    "Albania": "slavic",
    "Montenegro": "slavic",
    "Republic-of-North-Macedonia": "slavic",
    "Moldova-1": "slavic",
    "Hungary": "slavic",
    "Georgia-country": "slavic",
    "Armenia": "slavic",

    # english_speaking
    "South Africa": "english_speaking",
    "Nigeria": "english_speaking",
    "Philippines": "english_speaking",
    "Kenya": "english_speaking",
    "Ghana": "english_speaking",
    "Jamaica": "english_speaking",
    "Zimbabwe": "english_speaking",
    "Uganda": "english_speaking",
    "Trinidad and Tobago": "english_speaking",
    "Barbados": "english_speaking",
    "Singapore": "english_speaking",
    "Malaysia": "english_speaking",
    "Liberia": "english_speaking",
    "British-Colonial-Rule": "english_speaking",
    "The-Bahamas": "english_speaking",
    "Pakistan": "english_speaking",

    # south_america
    "Mexico": "south_america",
    "Brazil": "south_america",
    "Argentina": "south_america",
    "Colombia": "south_america",
    "Chile": "south_america",
    "Peru": "south_america",
    "Venezuela": "south_america",
    "Uruguay": "south_america",
    "Paraguay": "south_america",
    "Ecuador": "south_america",
    "Bolivia": "south_america",
    "Cuba": "south_america",
    "Dominican-Republic": "south_america",
    "Panama": "south_america",
    "El-Salvador": "south_america",
    "Nicaragua": "south_america",
    "Honduras": "south_america",
    "Guatemala": "south_america",
    "Costa-Rica-1": "south_america",
}

df = pd.read_csv("./final_metadata/vox2_meta_final.csv", sep=",", encoding="utf-8")

df["profession"] = df["profession"].map(profession_mapping).fillna("other")
df["nationality"] = df["nationality"].map(country_mapping).fillna("other")


df.to_csv("./final_metadata/vox2_merged_meta.csv", index=False)
