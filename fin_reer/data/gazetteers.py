import os
from typing import Set

import nltk
import pandas as pd

cwd = os.getcwd()


class Gazetteers:
    LOCATIONS_SET: Set[str] = set(pd.read_csv(
        os.path.join(cwd, 'data/locations_dataset/locations.csv.gz')).location.tolist())
    CITIES = set(pd.read_csv(os.path.join(cwd, "data/locations_dataset/us_cities.csv.gz")).city.tolist())
    STATES = set(pd.read_csv(os.path.join(cwd, "data/locations_dataset/us_states.csv.gz")).state.tolist())
    COUNTRIES = set(pd.read_csv(os.path.join(cwd, "data/locations_dataset/countries.csv.gz")).country.tolist())
    WORLD_CITIES = set(pd.read_csv(os.path.join(cwd, "data/locations_dataset/worldcities.csv.gz")).city.tolist())

    NAMES = set(pd.read_csv(os.path.join(cwd, "data/names_dataset/names.csv.gz")).name.tolist())

    STOP_WORDS = set(nltk.corpus.stopwords.words("english"))
