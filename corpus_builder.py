#!/usr/bin/env python
# coding: utf-8

# In[1]:


import data_downloader
import tempfile
import os
import sys
from logging import getLogger
logger = getLogger(__name__)


# In[ ]:


class Arguments():
    self.__init__(self, download_path, cache_path, years, dev_ratio=0.1):
        self.download_path = download_path
        self.cache_path = cache_path
        self.years = years
        self.dev_ratio = self.dev_ratio
        
    


# In[ ]:


WMT_IMPORTERS = {
    "15": data_downloader.Importer1516,
    "16": data_downloader.Importer1516,
    "17": data_downloader.Importer17,
    "18": data_downloader.Importer18,
    "19": data_downloader.Importer19,
    "20": data_downloader.Importer20
}


# In[ ]:


download_path = '/home/is/kosuke-t/Downloads/bleurt/bleurt/wmt/data'
cache_path = '/home/is/kosuke-t/Downloads/bleurt/bleurt/wmt/cache'
years = ["15", "16", "17", "18", "19", "20"]
dev_ratio = 0.0
target_filename = 'wmt15-20_DA_MQM_json'

target_path = os.path.join(download_path, target_filename)

args = Arguments(target_path=target_path, 
                 cache_path=cache_path, 
                 years=years, 
                 dev_ratio=dev_ratio)


# In[ ]:


def create_wmt_dataset(target_file, rating_years, target_language):
    """Creates a JSONL file for a given set of years and a target language."""
    logging.info("*** Downloading ratings data from WMT.")
    assert target_file
    assert not os.path.exists(args.target_file), "Target file already exists. Aborting."
    assert rating_years, "No target year detected."
    for year in rating_years:
        assert year in WMT_IMPORTERS, "No importer for year {}.".format(year)
    assert target_language
    assert target_language == "*" or len(target_language) == 2, "target_language must be a two-letter language code or `*`."

    with tempfile.TemporaryDirectory(dir=args.temp_directory) as tmpdir:
        logging.info("Using tmp directory: {}".format(tmpdir))

        n_records_total = 0
        for year in rating_years:
            logging.info("\nProcessing ratings for year {}".format(year))
            tmp_file = os.path.join(tmpdir, "tmp_ratings.json")

            # Builds an importer.
            importer_class = WMT_IMPORTERS[year]
            importer = importer_class(year, tmpdir, tmp_file)
            importer.fetch_files()
            lang_pairs = importer.list_lang_pairs()
            logging.info("Lang pairs found:")
            logging.info(" ".join(lang_pairs))

            for lang_pair in lang_pairs:

                if target_language != "*" and not lang_pair.endswith(target_language):
                    logging.info("Skipping language pair {}".format(lang_pair))
                    continue

                logging.info("Generating records for {} and language pair {}".format(year, lang_pair))
                n_records = importer.generate_records_for_lang(lang_pair)
                logging.info("Imported {} records.".format(str(n_records)))
                n_records_total += n_records

    logging.info("Done processing {} elements".format(n_records_total))
    logging.info("Copying temp file...")
    tf.io.gfile.copy(tmp_file, target_file, overwrite=True)
    logging.info("Done.")


# In[ ]:


def postprocess(target_file, remove_null_refs=True, average_duplicates=True):
    """Postprocesses a JSONL file of ratings downloaded from WMT."""
    logging.info("\n*** Post-processing WMT ratings {}.".format(target_file))
    assert tf.io.gfile.exists(target_file), "WMT ratings file not found!"
    base_file = target_file + "_raw"
    tf.io.gfile.rename(target_file, base_file, overwrite=True)

    logging.info("Reading and processing wmt data...")
    with tf.io.gfile.GFile(base_file, "r") as f:
        ratings_df = pd.read_json(f, lines=True)
    # ratings_df = ratings_df[["lang", "reference", "candidate", "rating"]]
    ratings_df.rename(columns={"rating": "score"}, inplace=True)

    if remove_null_refs:
        ratings_df = ratings_df[ratings_df["reference"].notnull()]
        assert not ratings_df.empty

    if average_duplicates:
        ratings_df = ratings_df.groupby(by=["lang", "source", "candidate", "reference"]).agg({"score": "mean",}).reset_index()

    logging.info("Saving clean file.")
    with tf.io.gfile.GFile(target_file, "w+") as f:
        ratings_df.to_json(f, orient="records", lines=True)
    logging.info("Cleaning up old ratings file.")
    tf.io.gfile.remove(base_file)


# In[ ]:


def _shuffle_no_leak(all_ratings_df, n_train):
    """Splits and shuffles such that there is no train/dev example with the same ref."""

    def is_split_leaky(ix):
        return (all_ratings_df.iloc[ix].reference == all_ratings_df.iloc[ix-1].reference)

    assert 0 < n_train < len(all_ratings_df.index)

    # Clusters the examples by reference sentence.
    sentences = all_ratings_df.reference.sample(frac=1, random_state=555).unique()
    sentence_to_ix = {s: i for i, s in enumerate(sentences)}
    all_ratings_df["__sentence_ix__"] = [sentence_to_ix[s] for s in all_ratings_df.reference]
    all_ratings_df = all_ratings_df.sort_values(by="__sentence_ix__")
    all_ratings_df.drop(columns=["__sentence_ix__"], inplace=True)

    # Moves the split point until there is no leakage.
    split_ix = n_train
    n_dev_sentences = len(all_ratings_df.iloc[split_ix:].reference.unique())
    if n_dev_sentences == 1 and is_split_leaky(split_ix):
        raise ValueError("Failed splitting data--not enough distinct dev sentences to prevent leak.")
    while is_split_leaky(split_ix):
        split_ix += 1
    if n_train != split_ix:
        logging.info("Moved split point from {} to {} to prevent sentence leaking".format(n_train, split_ix))

    # Shuffles the train and dev sets separately.
    train_ratings_df = all_ratings_df.iloc[:split_ix].copy()
    train_ratings_df = train_ratings_df.sample(frac=1, random_state=555)
    dev_ratings_df = all_ratings_df.iloc[split_ix:].copy()
    dev_ratings_df = dev_ratings_df.sample(frac=1, random_state=555)
    assert len(train_ratings_df) + len(dev_ratings_df) == len(all_ratings_df)

    # Checks that there is no leakage.
    train_sentences = train_ratings_df.reference.unique()
    dev_sentences = dev_ratings_df.reference.unique()
    tf.logging.info("Using {} and {} unique sentences for train and dev.".format(len(train_sentences), len(dev_sentences)))
    assert not bool(set(train_sentences) & set(dev_sentences))

    return train_ratings_df, dev_ratings_df


# In[ ]:


def _shuffle_leaky(all_ratings_df, n_train):
    """Shuffles and splits the ratings allowing overlap in the ref sentences."""
    all_ratings_df = all_ratings_df.sample(frac=1, random_state=555)
    all_ratings_df = all_ratings_df.reset_index(drop=True)
    train_ratings_df = all_ratings_df.iloc[:n_train].copy()
    dev_ratings_df = all_ratings_df.iloc[n_train:].copy()
    assert len(train_ratings_df) + len(dev_ratings_df) == len(all_ratings_df)
    return train_ratings_df, dev_ratings_df


# In[ ]:


def shuffle_split(ratings_file,
                  train_file=None,
                  dev_file=None,
                  dev_ratio=.1,
                  prevent_leaks=True):
    """Splits a JSONL WMT ratings file into train/dev."""
    logging.info("\n*** Splitting WMT data in train/dev.")

    assert tf.io.gfile.exists(ratings_file), "WMT ratings file not found!"
    base_file = ratings_file + "_raw"
    tf.io.gfile.rename(ratings_file, base_file, overwrite=True)

    logging.info("Reading wmt data...")
    with tf.io.gfile.GFile(base_file, "r") as f:
    ratings_df = pd.read_json(f, lines=True)

    logging.info("Doing the shuffle / split.")
    n_rows, n_train = len(ratings_df), int((1 - dev_ratio) * len(ratings_df))
    logging.info("Will attempt to set aside {} out of {} rows for dev.".format(n_rows - n_train, n_rows))
    if prevent_leaks:
        train_df, dev_df = _shuffle_no_leak(ratings_df, n_train)
    else:
        train_df, dev_df = _shuffle_leaky(ratings_df, n_train)
    logging.info("Created train and dev files with {} and {} records.".format(len(train_df), len(dev_df)))

    logging.info("Saving clean file.")
    if not train_file:
        train_file = ratings_file + "_train"
    with tf.io.gfile.GFile(train_file, "w+") as f:
        train_df.to_json(f, orient="records", lines=True)
    if not dev_file:
        dev_file = ratings_file + "_dev"
    with tf.io.gfile.GFile(dev_file, "w+") as f:
        dev_df.to_json(f, orient="records", lines=True)

    logging.info("Cleaning up old ratings file.")
    tf.io.gfile.remove(base_file)


# In[ ]:


create_wmt_dataset(args.target_file, args.rating_years, args.target_language)
postprocess(args.target_file, average_duplicates=args.average_duplicates)
if args.dev_ratio:
    shuffle_split(args.target_file, dev_ratio=args.dev_ratio, prevent_leaks=args.prevent_leaks)

