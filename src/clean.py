from argparse import ArgumentParser
import os.path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../data/hamburg_wohnungen.csv')


def parse_plz(address):
    plzs = re.findall('(\d{5})', str(address))
    if plzs:
        return plzs.pop()

    return None


def parse_parkplatz_type(parkplatz):
    if (not parkplatz) or (parkplatz == np.NaN):
        return None

    parkplatz = str(parkplatz).lower()

    if "tiefgarage" in parkplatz:
        return "tiefgarage"

    if "außenstellpl" in parkplatz:
        return "aussenstellplatz"

    if "stellpl" in parkplatz:
        return "stellplatz"

    if "carport" in parkplatz:
        return "carport"

    if "parkhaus" in parkplatz:
        return "parkhaus"

    if "garage" in parkplatz:
        return "garage"

    if "duplex" in parkplatz:
        return "duplex"

    return parkplatz


def parse_parkplatz_count(parkplatz):
    parkplaetze = re.findall('(\d+)', str(parkplatz))
    if parkplaetze:
        return parkplaetze.pop()

    return 0


def convert_to_bool(val):
    if val or (val == 1):
        return True

    if (not val) or (val == np.NaN):
        return False

    return val


def map_strings_to_integers(df, column):
    labels = df[column].unique().tolist()
    mapping = dict(zip(labels, range(len(labels))))
    df.replace({column: mapping}, inplace=True)

    return df


def clean_data(debug=False):
    df = pd.read_csv(FILE,
                     # Default seperator is ","
                     sep=";",
                     # First column is the index column
                     index_col=0,
                     # True = Internally process the file in chunks, resulting in lower memory use while parsing, but possibly mixed type inference.
                     low_memory=False,
                     # Columns to read
                     usecols=[
                         "_id",
                         "adr",  # 20251 Hamburg Die vollständige Adresse der Imm...
                         "aufzug",  # NaN | 1.0
                         "badezimmer",  # NaN | float
                         "balkon",  # NaN | 1.0
                         "baujahr",  # NaN | int
                         "etage",  # NaN | float
                         "gaeste_wc",  # NaN | 1.0
                         "garagenmiete",  # NaN | float
                         "gesmiete",  # NaN | float
                         "heizart",  # NaN | string
                         "heizung",  # NaN | string
                         "kaltmiete",  # NaN | float
                         "keller",  # NaN | 1.0
                         "kueche",  # NaN | 1.0
                         "last_crawled",  # NaN | string (e.g. "2018-10-02 13:57:41.945")
                         "lat",  # NaN | float
                         "lng",  # NaN | float
                         "nebenkosten",  # NaN | float
                         "parkplatz",  # NaN | string (e.g. "2 Tiefgaragen-Stellplätze")
                         "schlafzimmer",  # NaN | float
                         "stufenlos",  # NaN | 1.0
                         "sqm",  # int
                         "whg_typ",  # NaN | string (e.g. "Etagenwohnung")
                         "zimmer",  # int
                         "zustand",  # NaN | string (e.g. "Erstbezug")
                     ],
                     # Only read 100 lines in debug mode
                     nrows=100 if debug else None,
                     # Read last_crawled as date, not string
                     parse_dates=["last_crawled"],
                     infer_datetime_format=True,
                     converters={
                         "aufzug": lambda x: convert_to_bool(x),
                         "balkon": lambda x: convert_to_bool(x),
                         "gaeste_wc": lambda x: convert_to_bool(x),
                         "keller": lambda x: convert_to_bool(x),
                         "kueche": lambda x: convert_to_bool(x),
                         "stufenlos": lambda x: convert_to_bool(x),
                     }
                     )

    # Parse zip code from address
    df["plz"] = df["adr"].apply(parse_plz)

    # Parse parkplatz
    df["parkplatz_type"] = df["parkplatz"].apply(parse_parkplatz_type)
    df["parkplatz_count"] = df["parkplatz"].apply(parse_parkplatz_count)
    df = df.drop("parkplatz", 1)

    # Get year and month when the item was crawled
    df["year"] = df["last_crawled"].dt.year
    df["month"] = df["last_crawled"].dt.month

    df = df.drop("adr", 1)
    df = df.drop("last_crawled", 1)

    return df


def train_model(df, debug=False):
    x = df.drop("gesmiete", 1)
    x = map_strings_to_integers(x, "heizart")
    x = map_strings_to_integers(x, "heizung")
    x = map_strings_to_integers(x, "parkplatz_type")
    x = map_strings_to_integers(x, "whg_typ")
    x = map_strings_to_integers(x, "zustand")

    y = df["gesmiete"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--debug", help="Don't read complete csv file", action="store_true")
    args = parser.parse_args()

    df = clean_data(debug=args.debug)
    print(df)
    # train_model(df, debug=args.debug)
