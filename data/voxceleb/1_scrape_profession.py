import re
import time
import requests
import csv

WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

HEADERS = {
    "User-Agent": "VoxCeleb-Profession-Fetcher/1.0 (your_email@example.com)"
}


def add_man_vox1():
    d = {
        "A.J. Buckley": ['television actor', 'actor'],
        "A.R. Rahman": ['composer', 'singer'],
        "Arden Cho": ['singer-songwriter', 'actor', 'model', 'television actor', 'taekwondo athlete', 'film actor', 'voice actor'],
        "Caitriona Balfe": ['actor', 'model'],
        "Eddie Izzard": ['stand-up comedian', 'actor', 'voice actor', 'comedian', 'politician', 'improviser', 'television actor', 'writer', 'film director'],
        "Eva Longoria": ['actor', 'television actor', 'film actor', 'restaurateur', 'beauty pageant contestant', 'model', 'nightclub owner', 'television director', 'television producer', 'film director'],
        "Hye-kyo Song": ['actor', 'model'],
        "James Cosmo": ['film actor', 'television actor', 'actor'],
        "Jana Kramer": ['actor', 'singer', 'television actor', 'songwriter', 'film actor'],
        "Jesse Eisenberg": ['voice actor', 'film actor', 'television actor', 'stage actor', 'actor', 'screenwriter', 'film director'],
        "Kiernan Shipka": ['television actor', 'film actor', 'actor', 'voice actor', 'model'],
        "Loretta Swit": ['actor', 'television actor', 'stage actor', 'film actor'],
        "Mya": ['singer', 'songwriter', 'dancer', 'actor'],
        "Richard Ayoade": ['comedian', 'film director', 'television presenter', 'screenwriter', 'television actor', 'voice actor'],
        "Sarah Wayne Callies": ['voice actor', 'television director', 'actor'],
        "Sebastian De Souza": ['actor'],
        "Paul Reubens": ['actor', 'television actor', 'film actor', 'voice actor', 'comedian', 'writer', 'producer'],
        "Wendell Pierce": ['actor', 'television actor', 'film actor', 'producer']
    }
    return d


def add_man_vox2():
    d = {
        "Al Bano Carrisi": ['singer'],
        "Alex Aniston": ['actor', 'television actor', 'film actor', 'model', 'producer', 'writer'],
        "Alex OLoughlin": ['actor', 'television actor', 'film actor', 'producer', 'director'],
        "Alexandra Lara": ['actor', 'television actor', 'film actor'],
        "Ali al-Naimi": ['politician', 'economist', 'businessperson'],
        "Almazbek Atambayev": ['politician', 'economist'],
        "Amare Stoudemire": ['basketball player', 'basketball coach'],
        "Anahí de Cardenas": ['actor', 'television actor', 'film actor', 'model', 'singer-songwriter'],
        "Angéique Kidjo": ['singer', 'songwriter', 'activist', 'actor'],
        "Annemarie Warnkross": ['television presenter', 'radio personality', 'model', 'actor'],
        "Arno Del Curto": ['ice hockey coach', 'ice hockey player'],
        "Arya Babbar": ['actor', 'film actor', 'television actor'],
        "Avery Sunshine": ['singer', 'songwriter', 'musician'],
        "B. S. Yeddyurappa": ['politician', 'businessperson', 'economist'],
        "Barbara dUrso": ['television presenter', 'actor', 'television actor', 'journalist', 'writer', 'model'],
        "Bhairvi Goswami": ['actor', 'television actor', 'film actor', 'model'],
        "Bhavana Menon": ['actor', 'film actor', 'television actor', 'model'],
        "Bill Gilman": ['singer', 'songwriter', 'musician'],
        "Bruce Jenner": ['television actor', 'actor', 'athlete', 'reality television participant', 'sports commentator'],
        "Bárbara Palacios": ['television presenter', 'model', 'beauty pageant contestant'],
        "Candace Cameron-Bure": ['actor', 'television actor', 'film actor', 'producer', 'writer', 'author'],
        "Carlos Sainz, Jr.": ['Formula One driver', 'racing driver', 'motorsport competitor'],
        "Carrie Fletcher": ['actor', 'singer', 'author', 'voice actor', 'television actor', 'musician'],
        "Catherine OHara": ['actor', 'television actor', 'film actor', 'voice actor', 'comedian', 'writer', 'improviser', 'producer'],
        "Chael Sonnen": ['mixed martial arts fighter', 'podcaster', 'promoter', 'politician', 'amateur wrestler'],
        "Claude Makélélé": ['association football player', 'association football manager'],
        "Clifton Collins, Jr.": ['actor', 'television actor', 'film actor', 'voice actor'],
        "Conan OBrien": ['television presenter', 'actor', 'writer', 'comedian', 'film producer', 'television producer', 'voice actor'],
        "Czeslaw Mozil": ['singer', 'songwriter', 'actor', 'musician', 'television presenter'],
        "Cătălin Măruţă": ['television presenter', 'actor', 'television actor', 'film actor', 'radio personality'],
        "Daisy DeLa Hoya": ['television actor', 'actor', 'singer', 'reality television participant', 'model'],
        "Dale Earnhardt, Jr.": ['racing driver', 'sports commentator', 'businessperson', 'motorsport competitor'],
        "Dbanj": ['singer', 'songwriter', 'record producer', 'entrepreneur', 'actor', 'television presenter', 'philanthropist'],
        "Divya Khosla": ['actor', 'film actor', 'television actor', 'model', 'director', 'producer', 'screenwriter'],
        "Dmitri Hvorostovsky": ['opera singer'],
        "Dominik García-Lorido": ['film actor', 'television actor'],
        "Donna Leon": ['writer', 'teacher', 'novelist', 'travel guide', 'copywriter', 'docent', 'music interpreter', 'crime fiction writer', 'film director', 'film producer'],
        "Dravid": ['cricketer', 'cricket coach', 'cricket commentator', 'businessperson'],
        "Dylan Riley Snyder": ['actor', 'singer', 'dancer', 'television actor', 'model', 'film actor', 'stage actor'],
        "Dương Triệu Vũ": ['singer'],
        "Edita Vilkevičiūtė": ['model'],
        "Eiza González": ['actor', 'singer', 'songwriter', 'presenter'],
        "Elizabeth Shannon": ['actor', 'television actor', 'film actor', 'voice actor', 'model'],
        "Elyas M’Barek": ['actor', 'television actor', 'film actor'],
        "Eric Maxim Choupo-Moting": ['association football player'],
        "Fanny Lú": ['singer', 'songwriter', 'actor', 'television presenter'],
        "Felipe Gonzales": ['politician', 'lawyer', 'professor', 'economist'],
        "Galilea Montijo": ['actor', 'television actor', 'model', 'film actor', 'exotic dancer'],
        "Geeta Kapoor": ['choreographer', 'television presenter', 'actor'],
        "Genelia DSouza": ['actor', 'film actor', 'television actor', 'model'],
        "Giampiero Ventura": ['association football manager', 'association football player'],
        "Grégory Lemarchal": ['singer'],
        "Guilherme Marchi": ['bull rider', 'rodeo performer', 'rodeo athlete'],
        "Gustavo Dudamel": ['conductor', 'composer', 'concertmaster', 'violinist'],
        "H. P. Baxxter": ['singer', 'musician', 'rapper', 'master of ceremonies'],
        "Haider al-Abadi": ['politician', 'electrical engineer', 'engineer'],
        "Harry Connick, Jr.": ['singer', 'actor', 'television actor', 'film actor', 'television presenter', 'musician', 'composer', 'bandleader'],
        "Harshad Chopra": ['actor', 'television actor', 'film actor'],
        "Humaima Malik": ['actor', 'model', 'television actor', 'film actor'],
        "Ibrahim Lipumba": ['politician'],
        "Ignazio La Russa": ['politician', 'lawyer'],
        "Ines Sainz Gallo": ['journalist', 'television presenter', 'model', 'sports commentator'],
        "J.R. Martinez": ['actor', 'author', 'motivational speaker', 'reality television participant', 'soldier', 'model'],
        "J. B. Smoove": ['comedian', 'actor', 'screenwriter', 'television actor', 'voice actor', 'film actor', 'film producer'],
        "J. J. Redick": ['basketball player', 'basketball coach'],
        "J Alvarez": ['singer', 'songwriter', 'record producer', 'actor'],
        "James DArcy": ['actor', 'film actor', 'television actor', 'voice actor', 'producer'],
        "Jayasurya": ['actor', 'film producer', 'film actor', 'composer'],
        "Jenny Elvers-Elbertzhagen": ['actor', 'television actor', 'film actor', 'model', 'voice actor'],
        "Jerry OConnell": ['actor', 'television actor', 'film actor', 'voice actor', 'producer', 'director'],
        "Jin-hee Ji": ['actor', 'television actor', 'film actor', 'voice actor'],
        "Jodi Lyn OKeefe": ['actor', 'model', 'television actor', 'film actor', 'voice actor', 'producer'],
        "Joeystarr": ['actor', 'film actor', 'rapper', 'composer', 'television actor', 'record producer'],
        "JoséLuis Rodríguez Zapatero": ['politician', 'lawyer', 'economist'],
        "Joya Ahsan": ['actor', 'model', 'television actor', 'film actor'],
        "Juliano Cazarre": ['actor', 'television actor', 'film actor', 'voice actor'],
        "Julie Piétri": ['singer', 'songwriter', 'actor', 'television actor', 'film actor'],
        "Julio César Chávez, Jr.": ['boxer', 'sports commentator', 'businessperson'],
        "Julio Iglesias, Jr.": ['singer', 'actor', 'television actor', 'model', 'television presenter'],
        "Karen Fairchild": ['singer', 'songwriter', 'musician', 'record producer', 'actor'],
        "Karthi": ['actor'],
        "Kate OMara": ['actor', 'television actor', 'film actor', 'voice actor'],
        "Kipton Cronkite": ['photographer', 'artist', 'writer', 'television presenter', 'businessperson'],
        "Koo Ja-Cheol": ['association football player', 'footballer'],
        "Krystle Lina": ['model', 'actor', 'television actor', 'film actor'],
        "Lazaro Hernandez": ['fashion designer', 'businessperson', 'entrepreneur'],
        "Leymah Gbowee": ['politician', 'businessperson', 'economist', 'peace activist', 'activist'],
        "Liana Mendoza": ['actor', 'television actor', 'film actor', 'voice actor', 'model'],
        "Lil Kim": ['rapper', 'singer', 'songwriter', 'actress', 'television actor', 'model', 'record producer'],
        "Luisana Lopilato": ['actor', 'singer', 'model', 'stage actor', 'film actor'],
        "Malaika Arora Khan": ['actor', 'model', 'television actor', 'film actor', 'dancer', 'television presenter'],
        "Malin Åkerman": ['actor', 'singer', 'model', 'television actor', 'film actor', 'voice actor'],
        "Manchu Vishnu": ['actor', 'film actor', 'television actor', 'producer', 'businessperson'],
        "Marcello Melo Jr.": ['actor', 'film actor', 'television actor', 'singer', 'model'],
        "Margaret Blanchard": ['politician', 'lawyer', 'judge', 'educator', 'professor'],
        "Mariángel Ruiz": ['model', 'actor', 'beauty pageant contestant', 'presenter'],
        "Martin OMalley": ['politician', 'lawyer', 'musician', 'author', 'governor'],
        "Martin ONeill": ['association football manager', 'association football player', 'footballer'],
        "Mathew Baynton": ['actor', 'screenwriter', 'stage actor', 'film actor', 'musician', 'television actor', 'singer', 'writer'],
        "Matthew Bellamy": ['singer', 'songwriter', 'musician', 'guitarist', 'pianist', 'record producer', 'film actor'],
        "Meaghan Martin": ['actor', 'singer', 'guitarist', 'pianist', 'stage actor', 'television actor', 'model', 'film actor', 'voice actor', 'dancer'],
        "NadineNjeim": ['actor', 'model', 'television actor', 'film actor', 'beauty pageant contestant'],
        "Nancy DellOlio": ['television presenter', 'model', 'businessperson', 'author', 'fashion designer'],
        "Nancy ODell": ['television presenter', 'journalist', 'television producer', 'author', 'model'],
        "Natalia Wörner": ['actor', 'film actor', 'stage actor', 'television actor', 'screenwriter', 'voice actor', 'model'],
        "Nawal Al Zoghbi": ['singer', 'songwriter', 'actor', 'television actor', 'model'],
        "Nayantara": ['actor', 'model', 'television actor', 'film actor'],
        "Nic Tse": ['actor', 'singer', 'songwriter', 'model', 'television actor', 'film actor', 'record producer'],
        "Nina Watson": ['actor', 'television actor', 'film actor', 'voice actor', 'model'],
        "Nora Danish": ['actor', 'model'],
        "Norah ODonnell": ['journalist', 'television presenter', 'news presenter', 'author', 'producer'],
        "Norbert Blüm": ['politician', 'university teacher', 'writer'],
        "Ol Dirty Bastard": ['rapper', 'singer', 'songwriter', 'record producer', 'actor'],
        "Paul DiAnno": ['singer', 'songwriter', 'musician', 'record producer', 'actor'],
        "Paul DiGiovanni": ['musician', 'guitarist', 'songwriter', 'record producer', 'actor'],
        "Petter Northug": ['cross-country skier', 'poker player', 'sports commentator'],
        "Philipp Rösler": ['politician', 'physician', 'military physician'],
        "Piaa Bajpai": ['actor', 'model', 'television actor', 'film actor'],
        "Prahladananda Swami": ['teacher', 'guru', 'author', 'philosopher', 'yogi'],
        "Princess Haya bint Al Hussein": ['rider'],
        "Qorianka Kilcher": ['actor'],
        "Quincy Brown": ['actor', 'dancer', 'model'],
        "Rambha": ['actor'],
        "Renate Künast": ['politician', 'lawyer', 'environmentalist'],
        "Ricardo Antonio Chavira": ['actor'],
        "Richard Speight, Jr.": ['actor', 'producer'],
        "Robert Pirès": ['association football player', 'sports commentator'],
        "Rodrigo Guirao Díaz": ['actor'],
        "Ronan OGara": ['rugby union player'],
        "Ronnie OSullivan": ['snooker player'],
        "Rosie ODonnell": ['actor', 'comedian', 'television presenter'],
        "Saad Hariri": ['politician', 'businessperson'],
        "Sam Sorbo": ['actor', 'television actor', 'screenwriter', 'film actor', 'film director'],
        "Samuel Etoo": ['association football player'],
        "Saúl Lisazo": ['association football player', 'actor', 'television actor', 'film actor', 'model'],
        "Sean OPry": ['model', 'actor'],
        "Sebastian Edathy": ['politician'],
        "Shin-Soo Choo": ['baseball player'],
        "Shweta Menon": ['actor', 'model', 'television actor', 'film actor'],
        "Soledad OBrien": ['journalist', 'television presenter', 'producer', 'author'],
        "Soundarya R. Ashwin": ['actor', 'model', 'television actor', 'film actor'],
        "Tanaz Tabatabaei": ['actor', 'model', 'television actor', 'film actor'],
        "Tengku Abdullah": ['sports official'],
        "Tito El Bambino": ['singer', 'songwriter'],
        "Titus ONeil": ['professional wrestler', 'actor', 'television actor'],
        "V. K. Singh": ['politician', 'soldier', 'writer'],
        "Vicente Fox": ['politician', 'economist', 'businessperson'],
        "Vikram Pandit": ['banker', 'businessperson'],
        "Will i Am": ['record producer', 'singer', 'songwriter', 'rapper', 'actor', 'entrepreneur'],
        "Wolfgang Bosbach": ['politician', 'lawyer'],
        "Yayi Boni": ['politician', 'economist', 'banker'],
        "Álex Lora": ['composer', 'songwriter', 'singer', 'guitarist'],
        "ÁlvaroGarcía Linera": ['politician', 'sociologist', 'economist'],
        "Éric Ripert": ['chef', 'television presenter', 'author'],
        "Angélique Kidjo": ['singer', 'songwriter', 'activist', 'record producer'],
        "Bobbie Eakes": ['actor', 'singer', 'television actor', 'film actor'],
        "Daisy De La Hoya": ['actor', 'television actor', 'singer', 'model'],
        "José Luis Rodríguez Zapatero": ['politician', 'economist'],
        "Kareena Kapoor Khan": ['actor', 'model', 'television actor', 'film actor'],
        "Nadine Njeim": ['actor', 'model', 'television actor', 'film actor', 'beauty pageant contestant'],
        "Álvaro García Linera": ['politician', 'sociologist', 'economist'],
    }
    return d


class ProfObject:
    def __init__(self, id: str, name: str, professions: list[str]):
        self.id: str = id
        self.name: str = name
        self.professions: list[str] = professions

    def __str__(self):
        return f"ProfObject(id={self.id}, name={self.name}, professions={self.professions})"


def get_profession_from_wikidata(name) -> list[str]:
    query = f"""
    SELECT ?occupationLabel WHERE {{
      ?person ?label "{name}"@en.
      ?person wdt:P106 ?occupation.
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 5
    """

    response = requests.get(
        WIKIDATA_SPARQL_ENDPOINT,
        params={'query': query, 'format': 'json'},
        headers=HEADERS
    )

    # wait 1 sec
    time.sleep(0.2)

    if response.status_code == 200:
        data = response.json()
        bindings = data.get('results', {}).get('bindings', [])
        results = []

        for item in bindings:
            label = item.get('occupationLabel', {}).get('value')
            if label:
                results.append(label.strip())
            else:
                results = ['']
                print(f"[WARN] Kein Label gefunden in: {item}")
        return results
    else:
        print(f"Fehler {response.status_code}: {response.text}")
        return ['']


def get_professions_vox1(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # return list of ProfObject
    # txt of format:
    # id10001 A.J._Buckley
    # id10002 A.R._Rahman
    # ...

    names = []

    # Load already done professions from CSV
    try:
        with open("vox1_professions.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                seen = set()
                profs = []
                for p in row['Professions'].split(','):
                    p = p.strip()
                    if p and p not in seen:
                        seen.add(p)
                        profs.append(p)
                # if wikidata code in professions, dont add it
                ok = True
                for p in profs:
                    if re.fullmatch(r'Q\d+', p):
                        ok = False
                        break
                if ok:
                    names.append(ProfObject(
                        row['ID'], row['Name'], profs))
    except FileNotFoundError:
        print("vox1_professions.csv not found, starting fresh.")
    except Exception as e:
        print(f"Error reading vox1_professions.csv: {e}")
        already_done = set()

    d = add_man_vox1()

    # Load already done professions from dictionary
    already_done = [obj.name for obj in names if not obj.professions[0] == '']

    # Remove empty lines and comments, split by space, replace underscores with spaces
    lines = [line.strip() for line in lines if line.strip()
             and not line.startswith("#")]
    no_prof = []
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 2:
            continue
        id = parts[0]
        name = " ".join(parts[1:]).replace("_", " ")
        # look if name is already in names
        if name in already_done:
            continue
        if name in d:
            professions = d[name]
        else:
            professions = get_profession_from_wikidata(name)
            if len(professions) == 0:
                if not name in d:
                    no_prof.append(name)

        # remove duplicates and empty strings from professions
        seen = set()
        prof_clean = []
        for p in professions:
            p = p.strip()
            if p and p not in seen:
                seen.add(p)
                prof_clean.append(p)
        names.append(ProfObject(id, name, prof_clean))
        print(f"{i+1}/{len(lines)} processed")

    # remove names with no professions
    names = [obj for obj in names if obj.professions and obj.professions[0] != '']

    # remove duplicates
    seen = set()
    names = [obj for obj in names if not (
        obj.name in seen or seen.add(obj.name))]

    # write names to csv file
    with open("vox1_professions.csv", "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(['ID', 'Name', 'Professions'])
        for obj in names:
            writer.writerow([obj.id, obj.name, ", ".join(
                obj.professions) if obj.professions else ""])
    print(
        f"Processed {len(lines)} lines, found {len(names)} names with professions.")
    print(no_prof)
    print(len(no_prof))


def get_professions_vox2(path):
    s = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            name = row['Name '].replace("_", " ").strip()
            id = row['VoxCeleb2 ID '].strip()
            s.append([id, name])

    # return list of ProfObject
    # txt of format:
    # id10001 A.J._Buckley
    # id10002 A.R._Rahman
    # ...

    names = []

    # Load already done professions from CSV
    try:
        with open("vox2_professions.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                seen = set()
                profs = []
                for p in row['Professions'].split(','):
                    p = p.strip()
                    if p and p not in seen:
                        seen.add(p)
                        profs.append(p)
                # if wikidata code in professions, dont add it
                ok = True
                for p in profs:
                    if re.fullmatch(r'Q\d+', p):
                        ok = False
                        break
                if ok:
                    names.append(ProfObject(
                        row['ID'], row['Name'], profs))
    except FileNotFoundError:
        print("vox2_professions.csv not found, starting fresh.")
    except Exception as e:
        print(f"Error reading vox2_professions.csv: {e}")
        already_done = set()

    d = add_man_vox2()

    # Load already done professions from dictionary
    already_done = [obj.name for obj in names if not obj.professions[0] == '']

    # Remove empty lines and comments, split by space, replace underscores with spaces

    no_prof = []
    for i, entry in enumerate(s):
        if len(entry) < 2:
            continue
        id = entry[0]
        name = " ".join(entry[1:]).replace("_", " ")
        # look if name is already in names
        if name in already_done:
            continue
        if name in d:
            professions = d[name]
        else:
            professions = get_profession_from_wikidata(name)
            if len(professions) == 0:
                if not name in d:
                    no_prof.append(name)

         # remove duplicates and empty strings from professions
        seen = set()
        prof_clean = []
        for p in professions:
            p = p.strip()
            if p and p not in seen:
                seen.add(p)
                prof_clean.append(p)
        names.append(ProfObject(id, name, prof_clean))
        print(f"{i+1}/{len(s)} processed")

    # remove names with no professions
    names = [obj for obj in names if obj.professions and obj.professions[0] != '']

    # remove duplicates
    seen = set()
    names = [obj for obj in names if not (
        obj.name in seen or seen.add(obj.name))]

    # write names to csv file
    with open("vox2_professions.csv", "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(['ID', 'Name', 'Professions'])
        for obj in names:
            writer.writerow([obj.id, obj.name, ", ".join(
                obj.professions) if obj.professions else ""])
    print(
        f"Processed {len(s)} entries, found {len(names)} names with professions.")
    print(no_prof)
    print(len(no_prof))


if __name__ == "__main__":
    get_professions_vox1('vox1_identities.txt')

    get_professions_vox2('vox2_meta_names.csv')
