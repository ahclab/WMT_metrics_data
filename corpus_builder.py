#!/usr/bin/env python
# coding: utf-8

# In[11]:


import data_downloader
import tempfile
import os
import sys
from logging import getLogger
import shutil
import pandas as pd
import argparse
logger = data_downloader.logger


# In[ ]:


WMT_IMPORTERS = {
    "15": data_downloader.Importer1516,
    "16": data_downloader.Importer1516,
    "17": data_downloader.Importer17,
    "18": data_downloader.Importer18,
    "19": data_downloader.Importer19,
    "20": data_downloader.Importer20
}


# In[12]:


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {'off', 'false', '0'}
    TRUTHY_STRINGS = {'on', 'true', '1'}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")
        
parser = argparse.ArgumentParser()

# path settings
parser.add_argument('--target_path', type=str, default='/home/is/kosuke-t/scripts/make_data/wmt_metrics_data/data/wmt15-20_DA.json')
parser.add_argument('--cache_path', type=str, default='/home/is/kosuke-t/scripts/make_data/wmt_metrics_data')
parser.add_argument('--downloaded_dir', type=str, default='/home/is/kosuke-t/scripts/make_data/wmt_metrics_data/cache',
                    help='for WMT20 submissions data. Must be specified when targeting WMT20')

# others
parser.add_argument('--years', type=str, default='15,16,17,18,19,20', help='separation must be given by \",\"')
parser.add_argument('--target_language', type=str, default='*', help='if only english, then \"en\"')
parser.add_argument('--include_unreliables', type=bool_flag, default=False,
                    help='WMT20 has some unreliable data. This flag is set when including such data')
parser.add_argument('--onlyMQM', type=bool_flag, default=False, 
                    help='only download and preprocessing MQM data. When both of onlyMQM and onlyPSQM are False, download DA data on WMT20')
parser.add_argument('--onlyPSQM', type=bool_flag, default=False, 
                    help='only download and preprocessing PSQM data. When both of onlyMQM and onlyPSQM are False, download DA data on WMT20')
parser.add_argument('--addMQM', type=bool_flag, default=False, help='build DA and MQM mixed data')
parser.add_argument('--addPSQM', type=bool_flag, default=False, help='build DA and PSQM mixed data')
parser.add_argument('--average_duplicates', type=bool_flag, default=True, 
                    help='Whether to take average of the scores annotated to the same sentences')
parser.add_argument('--prevent_leaks', type=bool_flag, default=True, help='whether to allow for leaks among train and dev')
parser.add_argument('--dev_ratio', type=float, default=0.1, help='development ratio')

args = parser.parse_args()
args.years = args.years.split(',')

if args.addMQM:
    assert (not args.onlyMQM) and (not args.onlyPSQM) and (not args.addPSQM) ,    'addMQM can stand only when other signals are off'
if args.addPSQM:
    assert (not args.onlyMQM) and (not args.onlyPSQM) and (not args.addMQM) ,    'addPSQM can stand only when other signals are off' 

if '20' in args.years:
    assert os.path.isdir(args.downloaded_dir), 'Fetching 20\'s data cannot be completed with this script.\n'    'Download submission data from {} beforhand.\n'    'Then, put the data folder inside the downloaded_dir of the arguments.'.format(data_downloader.WMT_LOCATIONS['20']['submissions'][-1])


# In[15]:


def create_wmt_dataset(target_file, rating_years, target_language):
    """Creates a JSONL file for a given set of years and a target language."""
    logger.info("*** Downloading ratings data from WMT.")
    assert target_file
    assert not os.path.exists(args.target_path), "Target file already exists. Aborting."
    assert rating_years, "No target year detected."
    for year in rating_years:
        assert year in WMT_IMPORTERS, "No importer for year {}.".format(year)
    assert target_language
    assert target_language == "*" or len(target_language) == 2, "target_language must be a two-letter language code or `*`."
    
    with tempfile.TemporaryDirectory(dir=args.cache_path) as tmpdir:
        logger.info("Using tmp directory: {}".format(tmpdir))
        args.cache_path = tmpdir
        n_records_total = 0
        tmp_file = os.path.join(tmpdir, "tmp_ratings.json")
        
        if '20' in rating_years:
            logger.info('copying 20\'s data to tmp directory...')
            before_copy = os.path.join(args.downloaded_dir, data_downloader.WMT_LOCATIONS['20']['submissions'][0])
            after_copy = os.path.join(tmpdir, data_downloader.WMT_LOCATIONS['20']['submissions'][0])
            shutil.copytree(before_copy, after_copy)
            logger.info('Done.')
        
        def fetch_and_generate(n_records_total):
            # Builds an importer.
            importer_class = WMT_IMPORTERS[year]
            if year != '20':
                importer = importer_class(year, tmp_file, tmpdir, args)
            else:
                importer = importer_class(year, tmp_file, tmpdir, args, args.include_unreliables, args.onlyMQM, args.onlyPSQM)
            importer.fetch_files()
            lang_pairs = importer.list_lang_pairs()
            logger.info("Lang pairs found:")
            logger.info(" ".join(lang_pairs))

            for lang_pair in lang_pairs:

                if target_language != "*" and not lang_pair.endswith(target_language):
                    logger.info("Skipping language pair {}".format(lang_pair))
                    continue

                logger.info("Generating records for {} and language pair {}".format(year, lang_pair))
                n_records = importer.generate_records_for_lang(lang_pair)
                n_records_total += n_records
            return n_records_total
        
        for year in rating_years:
            logger.info("\nProcessing ratings for year {}".format(year))
            
            if year == '20' and args.addMQM:
                n_records_total = fetch_and_generate(n_records_total)
                args.onlyMQM = True
                n_records_total = fetch_and_generate(n_records_total)
            elif year == '20' and args.addPSQM:
                n_records_total = fetch_and_generate(n_records_total)
                args.onlyPSQM = True
                n_records_total = fetch_and_generate(n_records_total)
            else:
                n_records_total = fetch_and_generate(n_records_total)

        logger.info("Done processing {} elements".format(n_records_total))
        logger.info("Copying temp file...")
        shutil.copyfile(tmp_file, target_file)
        logger.info("Done.")


# In[16]:


def postprocess(target_file, remove_null_refs=True, average_duplicates=True):
    """Postprocesses a JSONL file of ratings downloaded from WMT."""
    logger.info("\n*** Post-processing WMT ratings {}.".format(target_file))
    base_file = target_file + "_raw"
    if not os.path.isfile(base_file):
        assert os.path.isfile(target_file), "WMT ratings file not found!"
        os.replace(target_file, base_file)

    logger.info("Reading and processing wmt data...")
    with open(base_file, "r") as f:
        ratings_df = pd.read_json(f, lines=True)
    # ratings_df = ratings_df[["lang", "reference", "candidate", "rating"]]
    ratings_df.rename(columns={"rating": "score"}, inplace=True)

    if remove_null_refs:
        ratings_df = ratings_df[ratings_df["reference"].notnull()]
        assert not ratings_df.empty

    if average_duplicates:
        try:
            ratings_df = ratings_df.groupby(by=["lang", "source", "candidate", "reference"]).agg({"score": "mean",}).reset_index()
        except:
            logger.info('No duplicates.')

    logger.info("Saving clean file.")
    with open(target_file, "w+") as f:
        ratings_df.to_json(f, orient="records", lines=True)
    logger.info("Cleaning up old ratings file.")
    os.remove(base_file)


# In[17]:


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
        logger.info("Moved split point from {} to {} to prevent sentence leaking".format(n_train, split_ix))

    # Shuffles the train and dev sets separately.
    train_ratings_df = all_ratings_df.iloc[:split_ix].copy()
    train_ratings_df = train_ratings_df.sample(frac=1, random_state=555)
    dev_ratings_df = all_ratings_df.iloc[split_ix:].copy()
    dev_ratings_df = dev_ratings_df.sample(frac=1, random_state=555)
    assert len(train_ratings_df) + len(dev_ratings_df) == len(all_ratings_df)

    # Checks that there is no leakage.
    train_sentences = train_ratings_df.reference.unique()
    dev_sentences = dev_ratings_df.reference.unique()
    logger.info("Using {} and {} unique sentences for train and dev.".format(len(train_sentences), len(dev_sentences)))
    assert not bool(set(train_sentences) & set(dev_sentences))

    return train_ratings_df, dev_ratings_df


# In[18]:


def _shuffle_leaky(all_ratings_df, n_train):
    """Shuffles and splits the ratings allowing overlap in the ref sentences."""
    all_ratings_df = all_ratings_df.sample(frac=1, random_state=555)
    all_ratings_df = all_ratings_df.reset_index(drop=True)
    train_ratings_df = all_ratings_df.iloc[:n_train].copy()
    dev_ratings_df = all_ratings_df.iloc[n_train:].copy()
    assert len(train_ratings_df) + len(dev_ratings_df) == len(all_ratings_df)
    return train_ratings_df, dev_ratings_df


# In[19]:


def shuffle_split(ratings_file,
                  train_file=None,
                  dev_file=None,
                  dev_ratio=.1,
                  prevent_leaks=True):
    """Splits a JSONL WMT ratings file into train/dev."""
    logger.info("\n*** Splitting WMT data in train/dev.")

    assert os.path.isfile(ratings_file), "WMT ratings file not found!"
    base_file = ratings_file + "_raw"
    os.replace(ratings_file, base_file)

    logger.info("Reading wmt data...")
    with open(base_file, "r") as f:
        ratings_df = pd.read_json(f, lines=True)

    logger.info("Doing the shuffle / split.")
    n_rows, n_train = len(ratings_df), int((1 - dev_ratio) * len(ratings_df))
    logger.info("Will attempt to set aside {} out of {} rows for dev.".format(n_rows - n_train, n_rows))
    if prevent_leaks:
        train_df, dev_df = _shuffle_no_leak(ratings_df, n_train)
    else:
        train_df, dev_df = _shuffle_leaky(ratings_df, n_train)
    logger.info("Created train and dev files with {} and {} records.".format(len(train_df), len(dev_df)))

    logger.info("Saving clean file.")
    if not train_file:
        train_file = ratings_file + "_train"
    with open(train_file, "w+") as f:
        train_df.to_json(f, orient="records", lines=True)
    if not dev_file:
        dev_file = ratings_file + "_dev"
    with open(dev_file, "w+") as f:
        dev_df.to_json(f, orient="records", lines=True)

    logger.info("Cleaning up old ratings file.")
    os.remove(base_file)


# In[20]:


create_wmt_dataset(args.target_path, args.years, args.target_language)
postprocess(args.target_path, average_duplicates=args.average_duplicates)
if args.dev_ratio > 0.0:
    shuffle_split(args.target_path, dev_ratio=args.dev_ratio, prevent_leaks=args.prevent_leaks)


# In[ ]:




