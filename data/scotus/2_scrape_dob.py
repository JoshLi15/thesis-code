import argparse
import datetime
import json
import os
import pickle
import re
import ssl
import string
import sys
import time
import warnings
from collections import OrderedDict

import numpy as np
import requests
import wikipedia
import wptools
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta
from Levenshtein import distance as lev_dist
from tqdm import tqdm
import urllib.parse
from SPARQLWrapper import SPARQLWrapper, JSON

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Attribution: https://github.com/hechmik/voxceleb_enrichment_age_gender/blob/main/README.md


ssl._create_default_https_context = ssl._create_unverified_context

MAN_PARSED = {
    "Ahilan T. Arulanantham": 1974,
    "Alfredo Parrish": 1947,
    "Alyza D. Lewin": 1965,
    "Amir C. Tayrani": 1978,
    "Amy Howe": 1995,
    "Andrew J. Pincus": 1952,
    "Andrew R. Hinkel": 1981,
    "Andrew S. Tulumello": 1971,
    "Ann-Michele G. Higgins": 1962,
    "Ann O'Connell Adams": 1995,
    "Arthur J. Madden, III": 1949,
    "Asher Perlin": 1968,
    "Atmore Baggot": 1944,
    "Beau B. Brindley": 1977,
    "Beth A. Burton": 1967,
    "Bonnie I. Robin-Vergeer": 1965,
    "Brandon E. Beck": 1978,
    "Brent E. Newton": 1967,
    "Brett E. Legner": 1976,
    "Brett John Prendergast": 1980,
    "Brian C. Shaughnessy": 1963,
    "Brian F. Barov": 1957,
    "Bridget C. Asay": 1971,
    "Burck Bailey": 1934,
    "Carl W. Thurman III": 1968,
    "Carla S. Sigler": 1975,
    "Charles W. Wirken": 1950,
    "Christina Rainville": 1974,
    "Christopher G. Michel": 1988,
    "Christopher M. Curran": 1965,
    "Colleen E. Roh Sinzdak": 1984,
    "Connie L. Lensing": 1952,
    "Craig L. Crawford": 1956,
    "Cynthia H. Hyndman": 1956,
    "D. Todd Doss": 1971,
    "Dale Schowengerdt": 1973,
    "Dana C. Hansen Chavis": 1975,
    "Daniel H. Ruttenberg": 1971,
    "Daniel N. Lerman": 1983,
    "Daniel Rogan": 1974,
    "Daniel T. Hansmeier": 1979,
    "David E. Mills": 1963,
    "David L. McColgin": 1962,
    "David M. Neff": 1960,
    "David W. Bowker": 1972,
    "Deborah Jones Merritt": 1955,
    "Deborah K. Brueckheimer": 1955,
    "Dominic F. Perella": 1975,
    "Donald A. Donati": 1950,
    "Donald R. Dunner": 1932,
    "Douglas Hallward-Driemeier": 1994,
    "Douglas Hallward-driemeier": 1994,
    "E. John Blanchard": 1952,
    "Earl N. Mayfield III": 1973,
    "Earle Duncan Getchell Jr": 1949,
    "Emmet J. Bondurant, II": 1937,
    "Emmett D. Queener": 1974,
    "Eric A. Burgess": 1950,
    "Eugene R. Wedoff": 1950,
    "Evan Mark Tager": 1960,
    "F. Paul Bland, Jr.": 1965,
    "Frank W. Heft, Jr.": 1955,
    "Franny A. Forsman": 1951,
    "Frederick R. Yarger": 1980,
    "G. Alan DuBois": 1960,
    "G. Eric Brunstad, Jr.": 1961,
    "Gary K. Smith": 1950,
    "Steve N. Six": 1965,
    "Gj Rod Sullivan Jr.": 1965,
    "Glen P. Gifford": 1960,
    "Gregg H. Levy": 1953,
    "Gregory Andrew Castanias": 1965,
    "Gregory A. Beck": 1972,
    "Griffin S. Dunham": 1978,
    "H. Christopher Bartolomucci": 1967,
    "H. David Blair": 1965,
    "Herald Price Fahringer": 1927,
    "Howard Srebnick": 1960,
    "Howard W. Foster": 1963,
    "Ilana Eisenstein": 1978,
    "J. Bradley O'Connell": 1960,
    "J. David Breemer": 1970,
    "J. Douglas Richards": 1960,
    "J. Gordon Cooney Jr.": 1959,
    "J. Michael Jakes": 1960,
    "James D. Leach": 1945,
    "James F. Blumstein": 1944,
    "James F. Goodhart": 1958,
    "James K. Leven": 1961,
    "James R. Radmore": 1957,
    "James Sterling Lawrence": 1952,
    "James W. Bilderback II": 1965,
    "James W. Dabney": 1958,
    "Jane N. Kirkland": 1948,
    "Jason E. Murtagh": 1975,
    "Jeffrey S. Bucholtz": 1970,
    "Jeffrey S. Gray": 1959,
    "Jeffrey T. Green": 1965,
    "Jere Krakoff": 1942,
    "Jeremy Friedlander": 1954,
    "Jessica E. Mendez-Colberg": 1986,
    "Joan C. Watt": 1967,
    "John G. Jacobs": 1947,
    "John G. Knepper": 1960,
    "John J.P. Howley": 1965,
    "Joseph F. Whalen": 1957,
    "Joseph Michaelangelo Alioto": 1960,
    "Joseph T. Maziarz": 1965,
    "Joshua T. Gillelan II": 1944,
    "Judith H. Mizner": 1951,
    "Julia Doyle Bernhardt": 1959,
    "Juliet L. Clark": 1970,
    "Karl J. Koch": 1942,
    "Kathleen A. Lord": 1963,
    "Kathryn Grill Graeff": 1961,
    "Kathryn Keena": 1965,
    "Kendall Turner": 1985,
    "Kenneth M. Rosenstein": 1960,
    "Kent E. Cattani": 1961,
    "Kent G. Holt": 1962,
    "Kim Martin Lewis": 1962,
    "Lauren A. Moskowitz": 1980,
    "Laurence E. Komp": 1967,
    "Leigh Marc Manasevit": 1950,
    "Lindsay C. Harrison": 1978,
    "Lindsay S. See": 1986,
    "Lisa Call": 1988,
    "Lisa Schiavo Blatt": 1965,
    "Lori Freno": 1975,
    "M. Reed Hopper": 1951,
    "Mark E. Elias": 1969,
    "Mark Irving Levy": 1949,
    "Mark L. Rienzi": 1978,
    "Mark R. Bendure": 1950,
    "Mark R. Freeman": 1978,
    "Mark T. Waggoner": 1987,
    "Mary R. O'Grady": 1962,
    "Masha G. Hansford": 1980,
    "Matthew C. Lawry": 1970,
    "Matthew D. McGill": 1975,
    "Matthew Guarnieri": 1985,
    "Matthew J. Edge": 1975,
    "Matthew S. Freedus": 1972,
    "Matthew W. Sawchak": 1964,
    "Matthew W.H. Wessler": 1980,
    "Michael J. Benza": 1964,
    "Michael W. Hawkins": 1947,
    "Morgan L. Ratner": 1986,
    "Oramel H. Skinner": 1985,
    "Paul A. Castiglione": 1960,
    "Paul J. Beard II": 1978,
    "Paul M. De Marco": 1967,
    "Paul M. Rashkind": 1955,
    "Paul W. Hughes": 1983,
    "Peter Enrich": 1950,
    "Peter Gartlan": 1967,
    "Peter Jon Van Hoek": 1956,
    "Peter Stirba": 1951,
    "Philip W. Savrin": 1960,
    "Priscilla J. Smith": 1966,
    "R. Jonathan Hart": 1952,
    "Ramzi Kassem": 1978,
    "Randall L. Allen": 1961,
    "Randolph H. Barnhouse": 1957,
    "Rebecca E. Woodman": 1951,
    "Rebecca Taibleson": 1985,
    "Richard Guerriero": 1965,
    "Richard M. Summa": 1965,
    "Randall E. Ravitz": 1973,
    "Robert D. Bartels": 1972,
    "Robert E. Salyer": 1978,
    "Robert M. Loeb": 1964,
    "Robert S. Glazier": 1958,
    "Robert T. Fishman": 1965,
    "Robert W. Coykendall": 1954,
    "Ronald J. VanAmberg": 1950,
    "Roy G. Davis": 1948,
    "Roy T. Englert, Jr.": 1958,
    "Roy T. Englert, Jr.": 1958,
    "Rudolph Telscher": 1965,
    "Ruth Botstein": 1974,
    "S. Kyle Duncan": 1972,
    "Samuel Bonderoff": 1977,
    "Samuel H. Heldman": 1968,
    "Sarah Schrup": 1977,
    "Scott A.C. Meisler": 1979,
    "Scott D. Makar": 1959,
    "Scott H. Angstreich": 1974,
    "Scott Michelman": 1979,
    "Sean E. Summers": 1978,
    "Sean Marotta": 1984,
    "Seth M. Galanter": 1967,
    "Shannon P. O'Connor": 1954,
    "Shanta Driver": 1960,
    "Sheri Lynn Johnson": 1961,
    "Sopan Joshi": 1988,
    "Steffen N. Johnson": 1969,
    "Stephen B. McCullough": 1972,
    "Steven B. Loy": 1970,
    "Steven J. Wells": 1960,
    "Stuart A. Raphael": 1964,
    "Stuart B. Lev": 1958,
    "Teddy B. Gordon": 1946,
    "Teresa Ficken Sachs": 1957,
    "Thomas A. Saenz": 1967,
    "Thomas C. Horne": 1945,
    "Thomas F. Jacobs": 1946,
    "Thomas G. Cotter": 1956,
    "Thomas M. Hefferon": 1961,
    "Thomas S. Waldo": 1965,
    "Todd G. Scher": 1967,
    "Todd G. Vare": 1969,
    "Valerie R. Newman": 1967,
    "W. James Young": 1963,
    "William D. Lunn": 1954,
    "William E. Thro": 1965,
    "William H. Hurd": 1955,
    "William J. Young": 1940,
    "William L. Messenger": 1975,
    "William P. Barnette": 1970,
    "William R. Allensworth": 1951,
    "Willis J. Goldsmith": 1947
}

MAN_PARSED_TOO_OLD = {
    "Ann O'Connell": 1982,
    "David O'Neil": 1963,
    "James B. Helmer, Jr.": 1958,
    "James E. Ryan": 1967,
    "James F. Hurst": 1964,
    "John C. Jones": 1960,
    "John F. Manning": 1961,
    "John S. Williams": 1979,
    "Joseph Margulies": 1963,
    "Katherine Burnett": 1960,
    "Mary E. Maguire": 1966,
    "Michael J. Meehan": 1950,
    "Paul Bender": 1933,
    "Paul Stern": 1960,
    "Richard Bourke": 1971,
    "Robert A. Long": 1960,
    "Robert C. Hilliard": 1958,
    "Robert Heim": 1966,
    "Sarah Harrington": 1974,
    "Theodore B. Olson": 1940,
    "Thomas Goldstein": 1970,
    "William S. Hastings": 1972,
    "Stephen M. Crawford": 1970,
    "Jeremiah Collins": 1952,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Reads through speaker_ids.json and tries to webscrape the DOB of each lawyer')
    parser.add_argument('--base-outfolder', type=str,
                        help='Location of the base outfolder')
    parser.add_argument('--overwrite', action='store_true',
                        default=False, help='Overwrite dob pickle (default: False)')
    parser.add_argument('--skip-attempted', action='store_true',
                        default=False, help='Skip names which have been attempted before')
    args = parser.parse_args()
    return args


class LawyerDOBParser:

    def __init__(self, graduation_dob_offset=25, distance_threshold=4, minimum_age=18,
                 enable_wikidata=True):
        self.graduation_dob_offset = graduation_dob_offset
        self.distance_threshold = distance_threshold
        self.scotus_clerks_populated = False
        self.minimum_age = minimum_age
        self.minimum_dob = datetime.datetime(
            2005, 10, 1) - relativedelta(years=minimum_age)

        self.enable_wikidata = enable_wikidata

    def parse_name(self, name):
        '''
        Parse name looking at wikipedia, then SCOTUS clerks, then the JUSTIA website

        Input: name
        Output: datetime object for D.O.B
        '''
        if not self.scotus_clerks_populated:
            self.get_scotus_clerks()

        # remove Mr. and Ms. from name
        name = re.sub(r'^(Mr\.|Ms\.)\s*', '', name)

        print('Searching for DOB of {}....'.format(name))
        # Optional: Wikidata-Search
        wikidata_dob, wikidata_info = None, {
            'info': {'type': 'wikidata', 'error': None, 'name': name}}
        if hasattr(self, "enable_wikidata") and self.enable_wikidata:
            wikidata_dob, wikidata_info = self.parse_wikidata(name)
            if wikidata_dob:
                wikidata_dob = datetime.datetime(wikidata_dob, 7, 2)
                if wikidata_dob <= self.minimum_dob:
                    return wikidata_dob, wikidata_info

        # Search wikipedia for person
        wiki_dob, wiki_info = self.parse_wiki(name)
        if wiki_dob:
            if wiki_dob <= self.minimum_dob:
                return wiki_dob, wiki_info

        # Search through supreme court clerks for person
        scotus_dob, scotus_info = self.search_scotus_clerks(name)
        if scotus_dob:
            scotus_dob = datetime.datetime(scotus_dob, 7, 2)
            if scotus_dob <= self.minimum_dob:
                return scotus_dob, scotus_info

        # Search through JUSTIA website
        justia_dob, justia_info = self.parse_justia(name)
        if justia_dob:
            justia_dob = datetime.datetime(justia_dob, 7, 2)
            if justia_dob <= self.minimum_dob:
                return justia_dob, justia_info

        print("Couldn't find any age for {}".format(name))
        info_list = [wiki_info, scotus_info, justia_info, wikidata_info]
        collated_info = {'info': {'type': None,
                                  'error': 'no info found', 'collated_info': info_list}}
        return None, collated_info

    def parse_wiki(self, name):
        # search = wikipedia.search(name)
        try:
            search = wikipedia.search(name)
        except Exception as e:
            print(f"Error searching Wikipedia for {name}: {e}")
            info = {'info': {'type': 'wiki', 'error': str(e)}}
            return None, info

        if search:
            if self.name_distance(name, search[0]) <= self.distance_threshold:
                name = search[0]

        wpage = wptools.page(name, silent=True)
        info = {'info': {'type': 'wiki', 'error': None, 'name': name}}
        try:
            page = wpage.get_parse()
        except:
            info['info']['error'] = 'page not found'
            return None, info
        try:
            if page.data:
                if 'infobox' in page.data:
                    if 'birth_date' in page.data['infobox']:
                        dob = page.data['infobox']['birth_date'].strip(
                            '{}').split('|')
                        dinfo = []
                        for d in dob:
                            try:
                                dinfo.append(int(d))
                            except:
                                continue
                        if dinfo:
                            if len(dinfo) > 3:
                                dinfo = dinfo[-3:]
                            if dinfo[0] > 1900:  # simple check if 4-digit year recognised
                                prelim_date = [1, 1, 1]
                                for i, d in enumerate(dinfo):
                                    prelim_date[i] = d
                                dob = datetime.datetime(*prelim_date)
                                info['info']['links'] = page.data['iwlinks']
                                return dob, info
            info['info']['error'] = 'page couldnt be parsed'
            return None, info
        except:
            info['info']['error'] = 'page couldnt be parsed'
            return None, info

    def parse_justia(self, name):
        searched_name, distance, justia_url = self.search_justia(name)
        info = {'info': {'type': 'justia', 'searched_name': searched_name,
                         'justia_url': justia_url, 'error': None}}
        if distance <= self.distance_threshold:
            grad_year = self.parse_justia_lawyer(justia_url)
            if grad_year:
                return grad_year - self.graduation_dob_offset, info
            else:
                info['info']['error'] = 'no year found'
                return None, info
        else:
            info['info']['error'] = 'distance threshold not met'
            return None, info

    def search_justia(self, name):
        """
        Input: Name to search, i.e. Anthony A. Yang (str,)
        Output: Matched name, Levenshtein distance to input, JUSTIA url
        """
        base_search_url = 'https://lawyers.justia.com/search?profile-id-field=&practice-id-field=&query={}&location='
        base_name = name.translate(str.maketrans(
            '', '', string.punctuation)).lower()
        name_query = '+'.join(base_name.split())
        search_url = base_search_url.format(name_query)

        search_url = base_search_url.format(name_query)

        # Selenium WebDriver used because normal request gets detected as bot
        options = Options()
        options.add_argument("--headless")  # Kein Fenster
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(
            "--blink-settings=imagesEnabled=false")  # Kein Bildladen
        prefs = {"profile.managed_default_content_settings.images": 2}
        options.add_experimental_option("prefs", prefs)
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/113.0.0.0 Safari/537.36")

        driver = webdriver.Chrome(options=options)
        driver.get(search_url)

        # wait for element, max 4s
        try:
            WebDriverWait(driver, 4).until(
                EC.presence_of_element_located(
                    (By.CLASS_NAME, "lawyer-avatar"))
            )
        except:
            driver.quit()
            return 'None', 100000, 'None'

        soup = BeautifulSoup(driver.page_source, "lxml")
        driver.quit()

        lawyer_avatars = soup.find_all('a', attrs={'class': 'lawyer-avatar'})

        if lawyer_avatars:
            search_names = []
            search_urls = []

            for a in lawyer_avatars:
                search_names.append(a['title'])
                search_urls.append(a['href'])

            search_names = np.array(search_names)
            search_names_base = [n.translate(str.maketrans(
                '', '', string.punctuation)).lower() for n in search_names]

            distances = np.array([self.name_distance(name, n)
                                 for n in search_names])
            search_urls = np.array(search_urls)

            dist_order = np.argsort(distances)
            distances = distances[dist_order]
            search_urls = search_urls[dist_order]
            search_names = search_names[dist_order]

            return search_names[0], distances[0], search_urls[0]
        else:
            return 'None', 100000, 'None'

    @staticmethod
    def parse_justia_lawyer(lawyer_url):
        """
        Input: Justia lawyer page url
        Output: Graduation year
        """

        # Selenium WebDriver used because normal request gets detected as bot
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(
            "--blink-settings=imagesEnabled=false")
        prefs = {"profile.managed_default_content_settings.images": 2}
        options.add_experimental_option("prefs", prefs)
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/113.0.0.0 Safari/537.36")

        driver = webdriver.Chrome(options=options)
        driver.get(lawyer_url)

        # wait for element, max 4s
        try:
            WebDriverWait(driver, 4).until(
                EC.presence_of_element_located((By.ID, "jurisdictions-block"))
            )
        except:
            driver.quit()
            return None

        soup = BeautifulSoup(driver.page_source, 'lxml')
        driver.quit()

        jurisdictions = soup.find('div', attrs={'id': 'jurisdictions-block'})

        if jurisdictions:
            jd_admitted_year = []
            for j in jurisdictions.find_all('time'):
                try:
                    jd_admitted_year.append(int(j['datetime']))
                except:
                    continue
            if jd_admitted_year:
                return min(jd_admitted_year)
            else:
                # look for professional associations if jurisdictions is emtpy
                prof_assoc = None
                education = None
                blocks = soup.find_all('div', attrs={'class': 'block'})
                for block in blocks:
                    subdivs = block.find_all('div')
                    for subdiv in subdivs:
                        if subdiv.text == 'Professional Associations':
                            prof_assoc = block
                            break
                        if subdiv.text == 'Education':
                            education = block
                            break

                if prof_assoc:
                    prof_assoc_year = []
                    professional_associations = prof_assoc.find_all('time')
                    for p in professional_associations:
                        try:
                            prof_assoc_year.append(int(p['datetime']))
                        except:
                            continue
                    if prof_assoc_year:
                        return min(prof_assoc_year)

                if education:
                    education_years = []
                    education_history = education.find_all('dl')
                    for e in education_history:
                        degree_type = e.find('dd').text
                        normalized = degree_type.strip().translate(
                            str.maketrans('', '', string.punctuation)).lower()
                        if normalized in ['jd', 'llb']:
                            # if degree_type.strip().translate(str.maketrans('', '', string.punctuation)).lower() == 'jd':
                            try:
                                return int(e.find('time')['datetime'])
                            except:
                                continue

    def search_scotus_clerks(self, query_name):
        assert self.clerk_dob_dict, 'get_scotus_clerks must be called before this function'
        distances = np.array([self.name_distance(query_name, k)
                             for k in self.scotus_clerks])
        closest_match = np.argmin(distances)
        info = {'info': {'type': 'clerk',
                         'closest_match': closest_match, 'error': None}}
        if distances[closest_match] <= self.distance_threshold:
            return self.clerk_dob_dict[self.scotus_clerks[closest_match]], info
        else:
            info['info']['error'] = 'distance threshold not met'
            return None, info

    def get_scotus_clerks(self):
        """
        Populates self.clerk_dob_dict with dates of birth for SCOTUS clerks
        """
        base_url = 'https://en.wikipedia.org/wiki/List_of_law_clerks_of_the_Supreme_Court_of_the_United_States_({})'
        seats = ['Chief_Justice', 'Seat_1', 'Seat_2',
                 'Seat_3', 'Seat_4', 'Seat_6', 'Seat_8',
                 'Seat_9', 'Seat_10']
        urls = [base_url.format(s) for s in seats]

        self.all_cdicts = []
        self.clerk_dob_dict = OrderedDict({})

        for url in urls:
            mini_clerk_dict = self.parse_clerk_wiki(url)
            self.all_cdicts.append(mini_clerk_dict)

        for cdict in self.all_cdicts:
            self.clerk_dob_dict = {**self.clerk_dob_dict, **cdict}

        self.scotus_clerks = np.array(list(self.clerk_dob_dict.keys()))
        self.scotus_clerks_populated = True

    def parse_clerk_wiki(self, url):
        r = requests.get(url)
        soup = BeautifulSoup(r.content, 'lxml')
        tables = soup.find_all('table', attrs={'class': 'wikitable'})
        clerk_dict = {}
        for table in tables:
            for tr in table.find_all('tr'):
                row_entries = tr.find_all('td')
                if len(row_entries) != 5:
                    continue
                else:
                    name = row_entries[0].text
                    u = row_entries[3].text
                    year_candidates = re.findall(r'\d{4}', u)

                    if year_candidates:
                        year = int(year_candidates[0])
                    else:
                        continue

                    cleaned_name = re.sub(r'\([^)]*\)', '', name)
                    cleaned_name = re.sub(r'\[[^)]*\]',
                                          '', cleaned_name).strip()
                    clerk_dict[cleaned_name] = year - \
                        self.graduation_dob_offset

        return clerk_dict

    def parse_wikidata(self, name):
        endpoint = "https://query.wikidata.org/sparql"
        sparql = SPARQLWrapper(endpoint)
        sparql.setReturnFormat(JSON)

        query = f"""
        SELECT ?person ?dob WHERE {{
          ?person rdfs:label|skos:altLabel "{name}"@en.
          ?person wdt:P569 ?dob.
        }}
        LIMIT 1
        """

        sparql.setQuery(query)
        info = {'info': {'type': 'wikidata', 'error': None, 'name': name}}

        try:
            results = sparql.query().convert()
            for result in results["results"]["bindings"]:
                dob = result["dob"]["value"]
                return int(dob[:4]), info
        except Exception as e:
            info['info']['error'] = str(e)
            return None, info

        return None, info

    @classmethod
    def name_distance(cls, string1, string2, wrong_initial_penalty=5):
        '''
       Levenshtein distance accommodating for:
        - Matching middle initials vs full middle names
        - Hyphenated surnames (e.g. Smith-Jones vs Jones)
        '''

        def normalize(name):
            return name.lower().translate(str.maketrans('', '', string.punctuation))

        name1 = normalize(string1)
        name2 = normalize(string2)

        base_dist = lev_dist(name1, name2)

        if base_dist == 0:
            return 0

        # special case for last names with -
        if '-' in string1.split()[-1] or '-' in string2.split()[-1]:
            s1_perms = cls.hyphenation_perm(string1)
            s2_perms = cls.hyphenation_perm(string2)
            dists = []
            for s1 in s1_perms:
                for s2 in s2_perms:
                    dists.append(cls.name_distance(s1, s2))
            return min(dists)

        name1_split = name1.split()
        name2_split = name2.split()

        # identical first and last names
        if name1_split[0] == name2_split[0] and name1_split[-1] == name2_split[-1]:
            # case query name has only first and last name
            if len(name1_split) == 2:
                return lev_dist(' '.join([name1_split[0], name1_split[-1]]),
                                ' '.join([name2_split[0], name2_split[-1]]))
            # Reduce all middle names to initials

            def reduce_middle(name_parts):
                return ' '.join([n[0] if (1 <= i < len(name_parts) - 1) else n for i, n in enumerate(name_parts)])
            return lev_dist(reduce_middle(name1_split), reduce_middle(name2_split))

        # fallback
        return base_dist + wrong_initial_penalty

    @staticmethod
    def hyphenation_perm(name):
        splitup = name.split()
        lastname = splitup[-1]
        if '-' in lastname:
            lname_candidates = [' '.join(splitup[:-1] + [l])
                                for l in lastname.split('-')]
            return lname_candidates
        else:
            return [name]


if __name__ == "__main__":
    args = parse_args()
    base_outfolder = args.base_outfolder
    assert os.path.isdir(base_outfolder)

    pickle_path = os.path.join(base_outfolder, 'dob.p')
    info_pickle_path = os.path.join(base_outfolder, 'dob_info.p')

    speaker_id_path = os.path.join(base_outfolder, 'speaker_ids.json')
    assert os.path.isfile(speaker_id_path), "Can't find speaker_ids.json"

    speaker_ids = json.load(open(speaker_id_path, encoding='utf-8'),
                            object_pairs_hook=OrderedDict)

    parser = LawyerDOBParser()
    parser.get_scotus_clerks()

    if args.overwrite or not os.path.isfile(pickle_path):
        dobs = OrderedDict({})
        infos = OrderedDict({})
        speakers_to_scrape = sorted(speaker_ids.keys())
    else:
        infos = pickle.load(open(info_pickle_path, 'rb'))
        dobs = pickle.load(open(pickle_path, 'rb'))
        if args.skip_attempted:
            speakers_to_scrape = set(speaker_ids.keys()) - set(dobs.keys())
        else:
            speakers_to_scrape = set(
                speaker_ids.keys()) - set([s for s in dobs if dobs[s]])

        if speakers_to_scrape:
            speakers_to_scrape = sorted(list(speakers_to_scrape))

    for s in tqdm(speakers_to_scrape):
        query_name = speaker_ids[s]['name']
        if query_name in MAN_PARSED:
            parsed_dob = MAN_PARSED[query_name]
            info = {'info': {'type': 'manual', 'error': None, 'name': query_name}}
        elif query_name in MAN_PARSED_TOO_OLD:
            parsed_dob = MAN_PARSED_TOO_OLD[query_name]
            info = {'info': {'type': 'manual', 'error': None, 'name': query_name}}
        else:
            parsed_dob, info = parser.parse_name(query_name)
        dobs[s] = parsed_dob
        infos[s] = info
        pickle.dump(dobs, open(pickle_path, 'wb'))
        pickle.dump(infos, open(info_pickle_path, 'wb'))

    num_dob_speakers = sum([1 for s in dobs if dobs[s]])

    print('Found DoB for {} out of {} speakers'.format(
        num_dob_speakers, len(speaker_ids)))
    print('Done!')
