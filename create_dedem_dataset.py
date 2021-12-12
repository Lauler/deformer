import pandas as pd

# Read data
df_wiki = pd.read_csv("data/wikimedia_sv.txt.gz", sep="\\n", header=None, engine="python")
df_wiki = df_wiki.rename(columns={0: "text"})
df_wiki["data_source"] = "wikimedia"

df_euro = pd.read_csv("data/europarl_sv.txt.gz", sep="\\n", header=None, engine="python")
df_euro = df_euro.rename(columns={0: "text"})
df_euro["data_source"] = "europarl"

df_jrc = pd.read_csv("data/jrc_sv.raw.gz", sep="\\n", header=None, engine="python")
df_jrc = df_jrc.rename(columns={0: "text"})
df_jrc["data_source"] = "jrc"

# Combine datasets
df = pd.concat([df_wiki, df_euro]).reset_index()
df = df.rename(columns={"index": "sentence_id"})

# Detect de/dem.
# Possible to do in a single regex, but good to have info on prevalence of 'de' vs 'dem'
df["contains_dem"] = df["text"].str.contains("(?<!\w)[Dd]em(?!\w)")
df["contains_de"] = df["text"].str.contains("(?<!\w)[Dd]e(?!\w)")

# Subset de/dem
df = df[(df["contains_dem"] == 1) | (df["contains_de"] == 1)].reset_index(drop=True)

# Keep only sentences <= 500 characters in length
df["nr_chars"] = df["text"].apply(lambda sentence: len(sentence))
df = df[df["nr_chars"] <= 500].reset_index(drop=True)

df.to_feather("data/dedem_corpus.feather")
