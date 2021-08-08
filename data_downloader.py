#!/usr/bin/env python
# coding: utf-8

# In[1]:


import abc
import glob
import itertools
import json
import os
import re
import shutil
import tarfile
import urllib
import urllib.request
import subprocess
from sklearn import preprocessing
import numpy as np

import six
# import tensorflow.compat.v1 as tf
from distutils.dir_util import copy_tree
from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False


# In[2]:


WMT_LOCATIONS = {'15': {"eval_data": ("DAseg-wmt-newstest2015", 
                                      "DAseg-wmt-newstest2015.tar.gz",
                                      "http://www.computing.dcu.ie/~ygraham/")},
                 '16': {"eval_data": ("DAseg-wmt-newstest2016", 
                                      "DAseg-wmt-newstest2016.tar.gz",
                                      "http://www.computing.dcu.ie/~ygraham/")},
                 '17': {"full_package":("wmt17-metrics-task-no-hybrids", 
                                        "wmt17-metrics-task-package.tgz",
                                        "http://ufallab.ms.mff.cuni.cz/~bojar/")},
                 '18': {"submissions":("wmt18-metrics-task-nohybrids", 
                                       "wmt18-metrics-task-nohybrids.tgz",
                                       "http://ufallab.ms.mff.cuni.cz/~bojar/wmt18/"),
                        "eval_data": ("newstest2018-humaneval", 
                                      "newstest2018-humaneval.tar.gz",
                                      "http://computing.dcu.ie/~ygraham/")},
                 '19': {"submissions": ("wmt19-submitted-data-v3",
                                        "wmt19-submitted-data-v3-txt-minimal.tgz",
                                        "http://ufallab.ms.mff.cuni.cz/~bojar/wmt19/"),
                        "eval_data": ("newstest2019-humaneval", 
                                      "newstest2019-humaneval.tar.gz",
                                      "https://www.computing.dcu.ie/~ygraham/")},
                 '20': {"submissions":("WMT20_data", "", "https://drive.google.com/drive/folders/1n_alr6WFQZfw4dcAmyxow4V8FC67XD8p"), 
                        "eval_data":("wmt20-metrics", "", "https://github.com/WMT-Metrics-task/wmt20-metrics"), 
                        "MQM":("wmt-mqm-human-evaluation", "", "https://github.com/google/wmt-mqm-human-evaluation"), 
                        'PSQM':("wmt-mqm-human-evaluation", "", "https://github.com/google/wmt-mqm-human-evaluation")},
                 '21':{'submissions':("WMT21-data", "", "")}}


# In[3]:


def separate_lang_pair(lang_pair):
    lang_expr = re.compile("([a-z]{2})-([a-z]{2})")
    match = lang_expr.match(lang_pair)
    if match:
        return match.group(1), match.group(2)
    else:
        return None


def postprocess_segment(segment):
    """Various string post-processing necessary to clean the records."""
    # Identifies NULL values.
    if segment == "NO REFERENCE AVAILABLE\n":
        return None
    # Removes trailing \n's.
    segment = segment.strip()
    return segment

def git_clone(url, destination_path):
    return subprocess.check_call(['git', 'clone', url, destination_path])


# In[4]:


@six.add_metaclass(abc.ABCMeta)
class WMTImporter(object):
    """Base class for WMT Importers.

    The aim of WMT importers is to fetch datafiles from the various WMT sources,
    collect information (e.g., list language pairs) and aggregate them into
    one big file.
    """

    def __init__(self, year, target_path, cache_path, args):
        self.year = year
        self.location_info = WMT_LOCATIONS[year]
        self.target_path = target_path
        self.cache_path = cache_path
        self.temp_directory = cache_path
        self.args = args
    
    def open_tar(self, cache_tar_path):
        logger.info("Untaring...")
        tar = tarfile.open(cache_tar_path)
        if self.year == '17':
            self.cache_path = os.path.join(self.cache_path, 'wmt17-metrics-task-package')
            if not os.path.isdir(self.cache_path):
                os.makedirs(self.cache_path)
        tar.extractall(path=self.cache_path)
        tar.close()
        logger.info("Done.")
    
    def fetch_files(self):
        """Downloads raw datafiles from various WMT sources."""
        cache = self.cache_path
        if cache and not os.path.isdir(cache):
            logger.info("Initializing cache {}".format(cache))
            os.makedirs(cache)
        
        for file_type in self.location_info:
            folder_name, archive_name, url_prefix = self.location_info[file_type]
            url = url_prefix + archive_name
            cache_tar_path = os.path.join(cache, archive_name)
            cache_untar_path = os.path.join(cache, archive_name).replace(".tgz", "", 1).replace(".tar.gz", "", 1)
            if cache:
                logger.info("Checking cached tar file {}.".format(cache_tar_path))
                if os.path.exists(cache_untar_path) :
                    logger.info("Cache and untar directory found, skipping")
        #           tf.io.gfile.copy(cache_untar_path, os.path.join(self.temp_directory, os.path.basename(cache_untar_path)), overwrite=True)
                    continue
                if os.path.isfile(cache_tar_path):
                    logger.info("Cache tar file found")
                    self.open_tar(cache_tar_path)

            logger.info("File not found in cache.")
            logger.info("Downloading {} from {}".format(folder_name, url))
            urllib.request.urlretrieve(url, cache_tar_path)
            logger.info("Done.")
            self.open_tar(cache_tar_path)

    def list_lang_pairs(self):
        """List all language pairs included in the WMT files for the target year."""
        pass

    def generate_records_for_lang(self, lang):
        """Consolidates all the files for a given language pair and year."""
        pass

    def cleanup(self):
        """Housekeeping--we want to erase all the temp files created."""
        for file_type in self.location_info:
            folder_name, archive_name, _ = self.location_info[file_type]

            # Removes data folder
            folder_path = os.path.join(self.temp_directory, folder_name)
            logger.info("Removing", folder_path)
            try:
                shutil.rmtree(folder_path)
            except OSError:
                logger.info("OS Error--skipping")

            # Removes downloaded archive
            archive_path = os.path.join(self.temp_directory, archive_name)
            logger.info("Removing", archive_path)
            try:
                os.remove(archive_path)
            except OSError:
                logger.info("OS Error--skipping")


# In[5]:


class Importer1516(WMTImporter):
    """Importer for years 2015 and 2016."""

    @staticmethod
    def to_json(year, lang, source, reference, candidate, rating, seg_id, system):
        """Converts record to JSON."""
        json_dict = {"year": int(year),
                     "lang": lang,
                     "source": postprocess_segment(source),
                     "reference": postprocess_segment(reference),
                     "candidate": postprocess_segment(candidate),
                     "raw_rating": None,
                     "rating": float(rating.strip()),
                     "segment_id": seg_id,
                     "system": system,
                     "n_ratings": None}
        return json.dumps(json_dict)

    @staticmethod
    def parse_file_name(fname):
        wmt_pattern = re.compile(r"^DAseg\.newstest([0-9]+)\.[a-z\-]+\.([a-z\-]+)")
        match = re.match(wmt_pattern, fname)
        if match:
            year, lang_pair = int(match.group(1)), match.group(2)
            return year, lang_pair
        else:
            return None, None

    def get_full_folder_path(self):
        """Returns path of directory with all the extracted files."""
        file_type = "eval_data"
        folder_name, _, _ = self.location_info[file_type]
        folder = os.path.join(self.cache_path, folder_name)
        return folder

    def list_files_for_lang(self, lang):
        """Lists the full paths of all the files for a given language pair."""
        year = "20"+self.year
        source_file = "DAseg.newstest{}.source.{}".format(str(year), lang)
        reference_file = "DAseg.newstest{}.reference.{}".format(str(year), lang)
        candidate_file = "DAseg.newstest{}.mt-system.{}".format(str(year), lang)
        rating_file = "DAseg.newstest{}.human.{}".format(str(year), lang)
        folder = self.get_full_folder_path()
        return {"source": os.path.join(folder, source_file),
                "reference": os.path.join(folder, reference_file),
                "candidate": os.path.join(folder, candidate_file),
                "rating": os.path.join(folder, rating_file)}

    def list_lang_pairs(self):
        folder = self.get_full_folder_path()
        file_names = os.listdir(folder)
        file_data = [Importer1516.parse_file_name(f) for f in file_names]
        lang_pairs = [lang_pair for year, lang_pair in file_data if year and lang_pair]
        return list(set(lang_pairs))

    def generate_records_for_lang(self, lang):
        year = '20'+self.year
        input_files = self.list_files_for_lang(lang)

        # pylint: disable=g-backslash-continuation
        with open(input_files["source"], "r", encoding="utf-8") as source_file,              open(input_files["reference"], "r", encoding="utf-8") as reference_file,              open(input_files["candidate"], "r", encoding="utf-8") as candidate_file,              open(input_files["rating"], "r", encoding="utf-8") as rating_file:
            # pylint: enable=g-backslash-continuation
            n_records = 0
            with open(self.target_path, "a+") as dest_file:
                for source, reference, candidate, rating in itertools.zip_longest(
                    source_file, reference_file, candidate_file, rating_file):
                    example = Importer1516.to_json(year, lang, source, reference, candidate, rating, n_records + 1, None)
                    dest_file.write(example)
                    dest_file.write("\n")
                    n_records += 1
                logger.info("Processed {} records of {}'s {}".format(str(n_records), year, lang))
                return n_records


# In[6]:


class Importer17(WMTImporter):
    """Importer for year 2017."""

    def __init__(self, *args, **kwargs):
        super(Importer17, self).__init__(*args, **kwargs)
        self.lang_pairs = None
        self.temp_directory = os.path.join(self.cache_path, "wmt17-metrics-task-package")

    def get_folder_path(self):
        """Returns path of directory with all the extracted files."""
        return self.temp_directory

    def agg_ratings_path(self):
        return os.path.join(self.temp_directory, "manual-evaluation", "DA-seglevel.csv")

    def segments_path(self, subset="root"):
        """Return the path to the source, reference, and candidate segments.

        Args:
          subset: one if "root", "source", "reference", or "candidate".

        Returns:
          Path to the relevant folder.
        """
        assert subset in ["root", "source", "reference", "candidate"]
        #     root_dir = os.path.join(self.temp_directory, "extracted_wmt_package")
        root_dir = os.path.join(self.temp_directory, "input")
        if subset == "root":
            return root_dir

        root_dir = os.path.join(root_dir, "wmt17-metrics-task-no-hybrids")
        if subset == "source":
            return os.path.join(root_dir, "wmt17-submitted-data", "txt", "sources")
        elif subset == "reference":
            return os.path.join(root_dir, "wmt17-submitted-data", "txt", "references")
        elif subset == "candidate":
            return os.path.join(root_dir, "wmt17-submitted-data", "txt", "system-outputs", "newstest2017")

    def fetch_files(self):
        """Downloads the WMT eval files."""
        # Downloads the main archive.
        super(Importer17, self).fetch_files()
    
        #Unpacks the segments.
        package_path = self.get_folder_path()
        segments_archive = os.path.join(package_path, "input", "wmt17-metrics-task-no-hybrids.tgz")
        with (tarfile.open(segments_archive, "r:gz")) as tar:
            tar.extractall(path=self.segments_path())
        logger.info("Unpacked the segments to {}.".format(self.segments_path()))

        # Gets the language pair names.
        ratings_path = self.agg_ratings_path()
        lang_pairs = set()
        with open(ratings_path, "r") as ratings_file:
            for l in itertools.islice(ratings_file, 1, None):
                lang = l.split(" ")[0]
                assert re.match("[a-z][a-z]-[a-z][a-z]", lang)
                lang_pairs.add(lang)
        self.lang_pairs = list(lang_pairs)
        logger.info("fetching Done")

    def list_lang_pairs(self):
        """List all language pairs included in the WMT files for the target year."""
        if self.lang_pairs == None:
            ratings_path = self.agg_ratings_path()
            lang_pairs = set()
            with open(ratings_path, "r") as ratings_file:
                for l in itertools.islice(ratings_file, 1, None):
                    lang = l.split(" ")[0]
                    assert re.match("[a-z][a-z]-[a-z][a-z]", lang)
                    lang_pairs.add(lang)
            self.lang_pairs = list(lang_pairs)
        return self.lang_pairs

    def get_ref_segments(self, lang):
        """Fetches source and reference translation segments for language pair."""
        src_subfolder = self.segments_path("source")
        ref_subfolder = self.segments_path("reference")
        src_lang, tgt_lang = separate_lang_pair(lang)
        src_file = "newstest2017-{}{}-src.{}".format(src_lang, tgt_lang, src_lang)
        ref_file = "newstest2017-{}{}-ref.{}".format(src_lang, tgt_lang, tgt_lang)
        src_path = os.path.join(src_subfolder, src_file)
        ref_path = os.path.join(ref_subfolder, ref_file)

#         logger.info("Reading data from files {} and {}".format(src_path, ref_path))
        with open(src_path, "r", encoding="utf-8") as f_src:
            src_segments = f_src.readlines()
        with open(ref_path, "r", encoding="utf-8") as f_ref:
            ref_segments = f_ref.readlines()
        src_segments = [postprocess_segment(s) for s in src_segments]
        ref_segments = [postprocess_segment(s) for s in ref_segments]
#         logger.info("Read {} source and {} reference segments.".format(len(src_segments), len(ref_segments)))
        return src_segments, ref_segments

    @staticmethod
    def parse_submission_file_name(fname):
        """Extracts system names from the name of submission files."""
        wmt_pattern = re.compile(r"^newstest2017\.([a-zA-Z0-9\-\.]+\.[0-9]+)\.[a-z]{2}-[a-z]{2}")
        match = re.match(wmt_pattern, fname)
        if match:
            return match.group(1)
        else:
            return None

    def get_sys_segments(self, lang):
        """Builds a dictionary with the generated segments for each system."""
        # Gets all submission file paths.
        root_folder = self.segments_path("candidate")
        folder = os.path.join(root_folder, lang)
        all_files = os.listdir(folder)
#         logger.info("Reading submission files from {}".format(folder))

        # Extracts the generated segments for each submission.
        sys_segments = {}
        for sys_file_name in all_files:
            sys_name = Importer17.parse_submission_file_name(sys_file_name)
            assert sys_name
            sys_path = os.path.join(folder, sys_file_name)
            with open(sys_path, "r", encoding="utf-8") as f_sys:
                sys_lines = f_sys.readlines()
                sys_lines = [postprocess_segment(s) for s in sys_lines]
                sys_segments[sys_name] = sys_lines

#         logger.info("Read submissions from {} systems".format(len(sys_segments.keys())))
        return sys_segments

    def parse_rating(self, line):
        fields = line.split()
        lang = fields[0]
        sys_names = fields[2].split("+")
        seg_id = int(fields[3])
        z_score = float(fields[4])
        raw_score = None
        for sys_name in sys_names:
            yield lang, sys_name, seg_id, raw_score, z_score

    def generate_records_for_lang(self, lang):
        """Consolidates all the files for a given language pair and year."""
        # Loads source, reference and system segments.
        src_segments, ref_segments = self.get_ref_segments(lang)
        sys_segments = self.get_sys_segments(lang)

        # Streams the rating file and performs the join on-the-fly.
        ratings_file_path = self.agg_ratings_path()
#         logger.info("Reading file {}".format(ratings_file_path))
        n_records = 0
        with open(ratings_file_path, "r", encoding="utf-8") as f_ratings:
            with open(self.target_path, "a+") as dest_file:
                for line in itertools.islice(f_ratings, 1, None):
                    for parsed_line in self.parse_rating(line):
                        line_lang, sys_name, seg_id, raw_score, z_score = parsed_line
                        if line_lang != lang:
                            continue
                        # The "-1" is necessary because seg_id starts counting at 1.
                        src_segment = src_segments[seg_id - 1]
                        ref_segment = ref_segments[seg_id - 1]
                        sys_segment = sys_segments[sys_name][seg_id - 1]
                        example = Importer18.to_json('20'+self.year, lang, src_segment,
                                                     ref_segment, sys_segment, raw_score,
                                                     z_score, seg_id, sys_name)
                        dest_file.write(example)
                        dest_file.write("\n")
                        n_records += 1
        logger.info("Processed {} records of {}'s {}".format(str(n_records), self.year, lang))
        return n_records


# In[7]:


class Importer18(WMTImporter):
    """Importer for year 2018."""

    def parse_submission_file_name(self, fname):
        """Extracts system names from the name of submission files."""
        wmt_pattern = re.compile(r"^newstest2018\.([a-zA-Z0-9\-\.]+\.[0-9]+)\.[a-z]{2}-[a-z]{2}")
        match = re.match(wmt_pattern, fname)
        if match:
            return match.group(1)
        else:
            return None

    def parse_eval_file_name(self, fname):
        """Extracts language pairs from the names of human rating files."""
        wmt_pattern = re.compile(r"^ad-seg-scores-([a-z]{2}-[a-z]{2})\.csv")
        match = re.match(wmt_pattern, fname)
        if match:
            return match.group(1)
        else:
            return None

    def list_lang_pairs(self):
        """List all language pairs included in the WMT files for 2018."""
        folder_name, _, _ = self.location_info["eval_data"]
        subfolder = "analysis"
        folder = os.path.join(self.temp_directory, folder_name, subfolder)
        all_files = os.listdir(folder)
        cand_lang_pairs = [self.parse_eval_file_name(fname) for fname in all_files]
        # We need to remove None values in cand_lang_pair:
        lang_pairs = [lang_pair for lang_pair in cand_lang_pairs if lang_pair]
        return list(set(lang_pairs))

    def get_ref_segments(self, lang):
        """Fetches source and reference translation segments for language pair."""
        folder, _, _ = self.location_info["submissions"]
        src_subfolder = os.path.join("sources")
        ref_subfolder = os.path.join("references")
        src_lang, tgt_lang = separate_lang_pair(lang)
        src_file = "newstest2018-{}{}-src.{}".format(src_lang, tgt_lang, src_lang)
        ref_file = "newstest2018-{}{}-ref.{}".format(src_lang, tgt_lang, tgt_lang)
        src_path = os.path.join(self.temp_directory, folder, src_subfolder, src_file)
        ref_path = os.path.join(self.temp_directory, folder, ref_subfolder, ref_file)

#         logger.info("Reading data from files {} and {}".format(src_path, ref_path))
        with open(src_path, "r", encoding="utf-8") as f_src:
            src_segments = f_src.readlines()
        with open(ref_path, "r", encoding="utf-8") as f_ref:
            ref_segments = f_ref.readlines()

        src_segments = [postprocess_segment(s) for s in src_segments]
        ref_segments = [postprocess_segment(s) for s in ref_segments]

        return src_segments, ref_segments

    def get_sys_segments(self, lang):
        """Builds a dictionary with the generated segments for each system."""
        # Gets all submission file paths.
        folder_name, _, _ = self.location_info["submissions"]
        subfolder = os.path.join("system-outputs", "newstest2018")
        folder = os.path.join(self.temp_directory, folder_name, subfolder, lang)
        all_files = os.listdir(folder)
#         logger.info("Reading submission files from {}".format(folder))

        # Extracts the generated segments for each submission.
        sys_segments = {}
        for sys_file_name in all_files:
            if sys_file_name == '.ipynb_checkpoints':
                continue
            sys_name = self.parse_submission_file_name(sys_file_name)
            assert sys_name
            sys_path = os.path.join(folder, sys_file_name)
            with open(sys_path, "r", encoding="utf-8") as f_sys:
                sys_lines = f_sys.readlines()
                sys_lines = [postprocess_segment(s) for s in sys_lines]
                sys_segments[sys_name] = sys_lines

        return sys_segments

    def get_ratings_path(self, lang):
        folder, _, _ = self.location_info["eval_data"]
        subfolder = "analysis"
        file_name = "ad-seg-scores-{}.csv".format(lang)
        return os.path.join(self.temp_directory, folder, subfolder, file_name)

    def parse_rating(self, rating_line):
        rating_tuple = tuple(rating_line.split(" "))
        # I have a feeling that the last field is the number of ratings
        # but I'm not 100% sure .
        sys_name, seg_id, raw_score, z_score, n_ratings = rating_tuple
        seg_id = int(seg_id)
        raw_score = float(raw_score)
        z_score = float(z_score)
        n_ratings = int(n_ratings)
        return sys_name, seg_id, raw_score, z_score, n_ratings

    @staticmethod
    def to_json(year, lang, src_segment, ref_segment, sys_segment,
                raw_score, z_score, seg_id, sys_name, n_ratings=0):
        """Converts record to JSON."""
        json_dict = {"year": year, "lang": lang, "source": src_segment, 
                     "reference": ref_segment, "candidate": sys_segment, "raw_rating": raw_score,
                     "rating": z_score, "segment_id": seg_id, "system": sys_name,
                     "n_ratings": n_ratings}
        return json.dumps(json_dict)

    def generate_records_for_lang(self, lang):
        """Consolidates all the files for a given language pair and year."""

        # Loads source, reference and system segments.
        src_segments, ref_segments = self.get_ref_segments(lang)
        sys_segments = self.get_sys_segments(lang)

        # Streams the rating file and performs the join on-the-fly.
        ratings_file_path = self.get_ratings_path(lang)
#         logger.info("Reading file {}".format(ratings_file_path))
        n_records = 0
        with open(ratings_file_path, "r", encoding="utf-8") as f_ratings:
            with open(self.target_path, "a+") as dest_file:
                for line in itertools.islice(f_ratings, 1, None):
                    line = line.rstrip()
                    parsed_tuple = self.parse_rating(line)
                    sys_name, seg_id, raw_score, z_score, n_ratings = parsed_tuple

                    # Those weird rules come from the WMT 2019 DA2RR script.
                    # Name of the script: seglevel-ken-rr.py, in Metrics results package.
                    if sys_name == "UAlacant_-_NM":
                        sys_name = "UAlacant_-_NMT+RBMT.6722"
                    if sys_name == "HUMAN":
                        continue
                    if sys_name == "RBMT.6722":
                        continue

                    # The following rules were added by me to unblock WMT2019:
                    if sys_name == "Helsinki-NLP.6889":
                        sys_name = "Helsinki_NLP.6889"
                    if sys_name == "Facebook-FAIR.6937":
                        sys_name = "Facebook_FAIR.6937"
                    if sys_name == "Facebook-FAIR.6937":
                        sys_name = "Facebook_FAIR.6937"
                    if sys_name == "DBMS-KU-KKEN.6726":
                        sys_name = "DBMS-KU_KKEN.6726"
                    if sys_name == "Ju-Saarland.6525":
                        sys_name = "Ju_Saarland.6525"
                    if sys_name == "aylien-mt-gu-en-multilingual.6826":
                        sys_name = "aylien_mt_gu-en_multilingual.6826"
                    if sys_name == "rug-kken-morfessor.6677":
                        sys_name = "rug_kken_morfessor.6677"
                    if sys_name == "talp-upc-2019-kken.6657":
                        sys_name = "talp_upc_2019_kken.6657"
                    if sys_name == "Frank-s-MT.6127":
                        sys_name = "Frank_s_MT.6127"

                    if lang == "de-cs" and sys_name == "Unsupervised.6935":
                        sys_name = "Unsupervised.de-cs.6935"
                    if lang == "de-cs" and sys_name == "Unsupervised.6929":
                        sys_name = "Unsupervised.de-cs.6929"

                    # The "-1" is necessary because seg_id starts counting at 1.
                    src_segment = src_segments[seg_id - 1]
                    ref_segment = ref_segments[seg_id - 1]
                    sys_segment = sys_segments[sys_name][seg_id - 1]
                    if not src_segment or not sys_segment:
                        logger.info("* Missing value!")
                        logger.info("* System: {}".format(sys_name))
                        logger.info("* Segment:" + str(seg_id))
                        logger.info("* Source segment:" + src_segment)
                        logger.info("* Sys segment:" + sys_segment)
                        logger.info("* Parsed line:" + line)
                        logger.info("* Lang:" + lang)
                    example = Importer18.to_json(self.year, lang, src_segment, 
                                                 ref_segment, sys_segment, raw_score, 
                                                 z_score, seg_id, sys_name, n_ratings)
                    dest_file.write(example)
                    dest_file.write("\n")
                    n_records += 1
        logger.info("Processed {} records of {}'s {}".format(str(n_records), self.year, lang))
        return n_records


# In[8]:


class Importer19(Importer18):
    """Importer for WMT19 Metrics challenge."""

    def parse_rating(self, rating_line):
        rating_tuple = tuple(rating_line.split(" "))
        # I have a feeling that the last field is the number of ratings
        # but I'm not 100% sure.
        sys_name, seg_id, raw_score, z_score, n_ratings = rating_tuple

        # For some reason, all the systems for pair zh-en have an extra suffix.
        if sys_name.endswith("zh-en"):
            sys_name = sys_name[:-6]

        seg_id = int(seg_id)
        raw_score = float(raw_score)
        z_score = float(z_score)
        n_ratings = int(n_ratings)
        return sys_name, seg_id, raw_score, z_score, n_ratings

    def parse_submission_file_name(self, fname):
        """Extracts system names from the name of submission files."""

        # I added those rules to unblock the pipeline.
        if fname == "newstest2019.Unsupervised.de-cs.6929.de-cs":
            return "Unsupervised.de-cs.6929"
        elif fname == "newstest2019.Unsupervised.de-cs.6935.de-cs":
            return "Unsupervised.de-cs.6935"

        wmt_pattern = re.compile(r"^newstest2019\.([a-zA-Z0-9\-\.\_\+]+\.[0-9]+)\.[a-z]{2}-[a-z]{2}")
        match = re.match(wmt_pattern, fname)
        if match:
            return match.group(1)
        else:
            return None

    def list_lang_pairs(self):
        """List all language pairs included in the WMT files for 2019."""
        folder_name, _, _ = self.location_info["eval_data"]
        folder = os.path.join(self.temp_directory, folder_name, "*", "analysis", "ad-seg-scores-*.csv")
        all_full_paths = glob.glob(folder)
        all_files = [os.path.basename(f) for f in all_full_paths]
        cand_lang_pairs = [self.parse_eval_file_name(fname) for fname in all_files]
        # We need to remove None values in cand_lang_pair:
        lang_pairs = [lang_pair for lang_pair in cand_lang_pairs if lang_pair]
        return list(set(lang_pairs))

    def get_ratings_path(self, lang):
        folder, _, _ = self.location_info["eval_data"]

        # The pair zh-en has two versions in the WMT 2019 human eval folder.
        if lang == "zh-en":
            path = os.path.join(self.temp_directory, folder, 
                                "turkle-sntlevel-humaneval-newstest2019", "analysis", "ad-seg-scores-zh-en.csv")
            return path

        file_name = "ad-seg-scores-{}.csv".format(lang)
        folder = os.path.join(self.temp_directory, folder, "*", "analysis", "ad-seg-scores-*.csv")
        all_files = glob.glob(folder)
        for cand_file in all_files:
            if cand_file.endswith(file_name):
                return cand_file
        raise ValueError("Can't find ratings for lang {}".format(lang))

    def get_ref_segments(self, lang):
        """Fetches source and reference translation segments for language pair."""
        folder, _, _ = self.location_info["submissions"]
        src_subfolder = os.path.join("txt", "sources")
        ref_subfolder = os.path.join("txt", "references")
        src_lang, tgt_lang = separate_lang_pair(lang)
        src_file = "newstest2019-{}{}-src.{}".format(src_lang, tgt_lang, src_lang)
        ref_file = "newstest2019-{}{}-ref.{}".format(src_lang, tgt_lang, tgt_lang)
        src_path = os.path.join(self.temp_directory, folder, src_subfolder, src_file)
        ref_path = os.path.join(self.temp_directory, folder, ref_subfolder, ref_file)

#         logger.info("Reading data from files {} and {}".format(src_path, ref_path))
        with open(src_path, "r", encoding="utf-8") as f_src:
            src_segments = f_src.readlines()
        with open(ref_path, "r", encoding="utf-8") as f_ref:
            ref_segments = f_ref.readlines()

        src_segments = [postprocess_segment(s) for s in src_segments]
        ref_segments = [postprocess_segment(s) for s in ref_segments]

        return src_segments, ref_segments

    def get_sys_segments(self, lang):
        """Builds a dictionary with the generated segments for each system."""
        # Gets all submission file paths.
        folder_name, _, _ = self.location_info["submissions"]
        subfolder = os.path.join("txt", "system-outputs", "newstest2019")
        folder = os.path.join(self.temp_directory, folder_name, subfolder, lang)
        all_files = os.listdir(folder)
#         logger.info("Reading submission files from {}".format(folder))

        # Extracts the generated segments for each submission.
        sys_segments = {}
        for sys_file_name in all_files:
            sys_name = self.parse_submission_file_name(sys_file_name)
            assert sys_name
            sys_path = os.path.join(folder, sys_file_name)
            with open(sys_path, "r", encoding="utf-8") as f_sys:
                sys_lines = f_sys.readlines()
                sys_lines = [postprocess_segment(s) for s in sys_lines]
                sys_segments[sys_name] = sys_lines

        return sys_segments


# In[9]:


class Importer20(Importer18):
    """Importer for WMT20 Metrics challenge."""
    
    def __init__(self, year, target_path, cache_path, args, include_unreliables, onlyMQM=False, onlyPSQM=False):
        super(Importer20, self).__init__(year, target_path, cache_path, args)
        self.include_unreliables = include_unreliables
        self.onlyMQM = onlyMQM
        self.onlyPSQM = onlyPSQM
        self.args = args
        assert not (onlyMQM and onlyPSQM), "only one of onlyMQM or onlyPSQM can stand"
    
    def open_tar(self, tar_path, open_dir):
        logger.info("Untaring...")
        tar = tarfile.open(tar_path)
        tar.extractall(path=open_dir)
        tar.close()
        logger.info("Done.")
    
    def fetch_files(self):
        """Downloads raw datafiles from various WMT sources."""
        cache = self.cache_path
        
        if cache and not os.path.isdir(cache):
            logger.info("Initializing cache {}".format(cache))
            os.makedirs(cache)
        
        for file_type in self.location_info:
            if self.onlyMQM:
                if file_type in ['submissions', 'eval_data', 'PSQM']:
                    continue
            elif self.onlyPSQM:
                if file_type in ['submissions', 'eval_data', 'MQM']:
                    continue
            
            folder_name, _, url = self.location_info[file_type]
            cache_untar_path = os.path.join(cache, folder_name)
            
            if cache:
                logger.info("Checking cached tar file {}.".format(cache_untar_path))
                if os.path.exists(cache_untar_path) :
                    if file_type == 'submissions':
                        tars = os.path.join(cache_untar_path, '*.tar.gz')
                        tar_paths = glob.glob(tars)
                        untar_paths = [path.replace(".tar.gz", "", 1) for path in tar_paths]
                        for tar_path, untar_paths in zip(tar_paths, untar_paths):
                            if not os.path.exists(untar_paths):
                                self.open_tar(tar_path, cache_untar_path)
                            else:
                                logger.info("Cache and untar directory found, skipping")
                    else:
                        logger.info("Cache and untar directory found, skipping")
                    continue
            logger.info("File not found in cache.")
            if file_type == 'submissions':
                logger.info("Cannot download {} with this script. Download from {}".format(folder_name, url))
                exit(-1)
            logger.info("Downloading {} from {}".format(folder_name, url))
            git_clone(url, cache_untar_path)
            logger.info("Done.")  
    
    def parse_rating(self, rating_line, lang):
        rating_tuple = tuple(rating_line.split(" "))
        # I have a feeling that the last field is the number of ratings
        # but I'm not 100% sure.
        sys_name, seg_id, raw_score, z_score, n_ratings = rating_tuple
        
        # en-zh has unknown seg_id probablly tagged with other format name
        if lang in ['en-zh', 'en-ja', 'en-iu', 'en-cs', 'en-ta', 'en-ru', 'en-de', 'en-pl']:
            seg_id = seg_id.split('_')[-1] 
        
        try:
            seg_id = int(seg_id)
            raw_score = float(raw_score)
            z_score = float(z_score)
            n_ratings = int(n_ratings)
        except:
            logger.info(lang)
            logger.info(rating_line)
        return sys_name, seg_id, raw_score, z_score, n_ratings

    def parse_submission_file_name(self, fname, lang):
        """Extracts system names from the name of submission files."""

        # I added those rules to unblock the pipeline.

        sys_name = fname.replace("newstest2020.{}.".format(lang), "", 1).replace(".txt", "", 1)

        return sys_name

    def list_lang_pairs(self):
        """List all language pairs included in the WMT files for 2020."""
        if self.onlyMQM or self.onlyPSQM:
            return ['en-de', 'zh-en']
        
        folder_name, _, _ = self.location_info["eval_data"]
        folder = os.path.join(self.temp_directory, folder_name, "manual-evaluation", "DA", "ad-seg-scores-*.csv")
        all_full_paths = glob.glob(folder)
        all_files = [os.path.basename(f) for f in all_full_paths]
        cand_lang_pairs = [self.parse_eval_file_name(fname) for fname in all_files]
        # We need to remove None values in cand_lang_pair:
        lang_pairs = [lang_pair for lang_pair in cand_lang_pairs if lang_pair]
        return list(set(lang_pairs))

    def get_ratings_path(self, lang):
        folder, _, _ = self.location_info["eval_data"]

        file_name = "ad-seg-scores-{}.csv".format(lang)
        folder = os.path.join(self.temp_directory, folder, "manual-evaluation", "DA", "ad-seg-scores-*.csv")
        all_files = glob.glob(folder)
        for cand_file in all_files:
            if cand_file.endswith(file_name):
                return cand_file
        raise ValueError("Can't find ratings for lang {}".format(lang))

    def get_ref_segments(self, lang):
        """Fetches source and reference translation segments for language pair."""
        folder, _, _ = self.location_info["submissions"]
        src_subfolder = os.path.join("txt", "sources")
        ref_subfolder = os.path.join("txt", "references")
        src_lang, tgt_lang = separate_lang_pair(lang)
        src_file = "newstest2020-{}{}-src.{}.txt".format(src_lang, tgt_lang, src_lang)
        ref_file = "newstest2020-{}{}-ref.{}.txt".format(src_lang, tgt_lang, tgt_lang)
        src_path = os.path.join(self.temp_directory, folder, src_subfolder, src_file)
        ref_path = os.path.join(self.temp_directory, folder, ref_subfolder, ref_file)

#         logger.info("Reading data from files {} and {}".format(src_path, ref_path))
        with open(src_path, "r", encoding="utf-8") as f_src:
            src_segments = f_src.readlines()
        with open(ref_path, "r", encoding="utf-8") as f_ref:
            ref_segments = f_ref.readlines()

        src_segments = [postprocess_segment(s) for s in src_segments]
        ref_segments = [postprocess_segment(s) for s in ref_segments]

        return src_segments, ref_segments

    def get_sys_segments(self, lang):
        """Builds a dictionary with the generated segments for each system."""
        # Gets all submission file paths.
        folder_name, _, _ = self.location_info["submissions"]
        subfolder = os.path.join("txt", "system-outputs")
        folder = os.path.join(self.temp_directory, folder_name, subfolder, lang)
        all_files = os.listdir(folder)
#         logger.info("Reading submission files from {}".format(folder))

        # Extracts the generated segments for each submission.
        sys_segments = {}
        for sys_file_name in all_files:
            sys_name = self.parse_submission_file_name(sys_file_name, lang)
            assert sys_name
            sys_path = os.path.join(folder, sys_file_name)
            with open(sys_path, "r", encoding="utf-8") as f_sys:
                sys_lines = f_sys.readlines()
                sys_lines = [postprocess_segment(s) for s in sys_lines]
                sys_segments[sys_name] = sys_lines

        return sys_segments
    
    def parse_mqm(self, line, lang):
        rating_tuple = tuple(line.split("\t"))
        system, doc, doc_id, seg_id, rater, source, target, category, severity = rating_tuple
        
        score = 0.0
        assert severity in ['Major', 'Minor', 'Neutral', 'no-error'], 'unknown severity:{}'.format(severity)

        if severity == 'Major':
            if category == 'Non-translation!':
                score = -25.0
            else:
                score = -5.0
        elif severity == 'Minor':
            if category == 'Fluency/Punctuation':
                score = -0.1
            else:
                score = -1.0
        
        try:
            doc_id = int(doc_id)
            seg_id = int(seg_id)
        except:
            logger.info(lang)
            logger.info(line)
        return system, doc, doc_id, seg_id, rater, source, target, category, severity, score
    
    def get_mqm_segments(self, lang):
        src_lang, tgt_lang = separate_lang_pair(lang)
        folder_name, _, _ = self.location_info["MQM"]
        file = os.path.join(self.temp_directory, folder_name, "{}{}".format(src_lang, tgt_lang), "mqm_newstest2020_{}{}.tsv".format(src_lang, tgt_lang))
        rater_score = {}
        seg_scores = {}
        with open(file, mode='r', encoding='utf-8') as r:
            for line in itertools.islice(r, 1, None):
                line = line.rstrip()
                system, doc, doc_id, seg_id, rater, source, target, category, severity, score = self.parse_mqm(line, lang)
                if rater not in rater_score:
                    rater_score[rater] = {'score':[score], 
                                          'source':[source.rstrip()],
                                          'target':[target.rstrip()], 
                                          'system':[system], 
                                          'seg_id':[seg_id]}
                else:
                    rater_score[rater]['score'].append(score)
                    rater_score[rater]['source'].append(source.rstrip())
                    rater_score[rater]['target'].append(target.rstrip())
                    rater_score[rater]['system'].append(system)
                    rater_score[rater]['seg_id'].append(seg_id)
        for rater in rater_score.keys():
            rater_score[rater]['z_score'] = list(preprocessing.scale(rater_score[rater]['score']))
        for rater in rater_score.keys():
            for seg_id, src, tgt, score, z_score, system in zip(rater_score[rater]['seg_id'], 
                                                                rater_score[rater]['source'], 
                                                                rater_score[rater]['target'], 
                                                                rater_score[rater]['score'], 
                                                                rater_score[rater]['z_score'], 
                                                                rater_score[rater]['system']):
                sys_id = (system, seg_id)
                if sys_id not in seg_scores:
                    seg_scores[sys_id] = {'rater':[rater],
                                          'score':[score], 
                                          'z_score':[z_score],
                                          'source':[source],
                                          'target':[target]}
                else:
                    seg_scores[sys_id]['rater'].append(rater)
                    seg_scores[sys_id]['score'].append(score)
                    seg_scores[sys_id]['z_score'].append(z_score)
                    seg_scores[sys_id]['source'].append(source)
                    seg_scores[sys_id]['target'].append(target)
        for sys_id in seg_scores.keys():
            seg_scores[sys_id]['z_mean_score'] = np.mean(seg_scores[sys_id]['z_score'])
            
        return rater_score, seg_scores
                
    def get_mqm_avg_segments(self, lang, seg_scores):
        src_lang, tgt_lang = separate_lang_pair(lang)
        folder_name, _, _ = self.location_info["MQM"]
        file = os.path.join(self.temp_directory, folder_name, "{}{}".format(src_lang, tgt_lang), "mqm_newstest2020_{}{}.avg_seg_scores.tsv".format(src_lang, tgt_lang))
        with open(file, mode='r', encoding='utf-8') as r:
            for line in itertools.islice(r, 1, None):
                line = line.rstrip()
                system, mqm_avg_score, seg_id = tuple(line.split(' '))
                sys_id = (system, int(seg_id))
                seg_scores[sys_id]['raw_score'] = mqm_avg_score
        return seg_scores
    
    def generate_mqm_records_for_lang(self, lang):
        rater_score, seg_scores = self.get_mqm_segments(lang)
        src_segments, ref_segments = self.get_ref_segments(lang)
        sys_segments = self.get_sys_segments(lang)
        seg_scores = self.get_mqm_avg_segments(lang, seg_scores)
        
        n_records = 0
        skipped_n_records = 0
        with open(self.target_path, "a+") as dest_file:
            for sys_id in seg_scores.keys():
                sys_name, seg_id = sys_id
                raw_score = seg_scores[sys_id]['raw_score']
                z_score = seg_scores[sys_id]['z_score']
                
                # The "-1" is necessary because seg_id starts counting at 1.
                src_segment = src_segments[seg_id - 1]
                ref_segment = ref_segments[seg_id - 1]
                if sys_name not in sys_segments:
                    print(sys_name, lang)
                else:
                    sys_segment = sys_segments[sys_name][seg_id - 1]

                if not src_segment or not sys_segment:
                    logger.info("* Missing value!")
                    logger.info("* System: {}".format(sys_name))
                    logger.info("* Segment:" + str(seg_id))
                    logger.info("* Source segment:" + src_segment)
                    logger.info("* Sys segment:" + sys_segment)
                    logger.info("* Parsed line:" + line)
                    logger.info("* Lang:" + lang)
                example = self.to_json(self.year, lang, src_segment, 
                                             ref_segment, sys_segment, raw_score, 
                                             z_score, seg_id, sys_name)
                dest_file.write(example)
                dest_file.write("\n")
                n_records += 1
        logger.info("Processed {} records of {}'s {}".format(str(n_records), self.year, lang))
        logger.info("Skipped {} records of {}'s {}".format(str(skipped_n_records), self.year, lang))
        return n_records
    
    def parse_psqm(self, line, lang):
        rating_tuple = tuple(line.split("\t"))
        system, doc, doc_id, seg_id, rater, source, target, score = rating_tuple
        
        score = 0.0
        try:
            doc_id = int(doc_id)
            seg_id = int(seg_id)
        except:
            logger.info(lang)
            logger.info(line)
        return system, doc, doc_id, seg_id, rater, source, target, score
    
    def get_psqm_segments(self, lang):
        src_lang, tgt_lang = separate_lang_pair(lang)
        folder_name, _, _ = self.location_info["PSQM"]
        file = os.path.join(self.temp_directory, folder_name, "{}{}".format(src_lang, tgt_lang), "psqm_newstest2020_{}{}.tsv".format(src_lang, tgt_lang))
        rater_score = {}
        seg_scores = {}
        with open(file, mode='r', encoding='utf-8') as r:
            for line in itertools.islice(r, 1, None):
                line = line.rstrip()
                system, doc, doc_id, seg_id, rater, source, target, score = self.parse_psqm(line, lang)
                if rater not in rater_score:
                    rater_score[rater] = {'score':[score], 
                                          'source':[source.rstrip()],
                                          'target':[target.rstrip()], 
                                          'system':[system], 
                                          'seg_id':[seg_id]}
                else:
                    rater_score[rater]['score'].append(score)
                    rater_score[rater]['source'].append(source.rstrip())
                    rater_score[rater]['target'].append(target.rstrip())
                    rater_score[rater]['system'].append(system)
                    rater_score[rater]['seg_id'].append(seg_id)
        for rater in rater_score.keys():
            rater_score[rater]['z_score'] = list(preprocessing.scale(rater_score[rater]['score']))
        for rater in rater_score.keys():
            for seg_id, src, tgt, score, z_score, system in zip(rater_score[rater]['seg_id'], 
                                                                rater_score[rater]['source'], 
                                                                rater_score[rater]['target'], 
                                                                rater_score[rater]['score'], 
                                                                rater_score[rater]['z_score'], 
                                                                rater_score[rater]['system']):
                sys_id = (system, seg_id)
                if sys_id not in seg_scores:
                    seg_scores[sys_id] = {'rater':[rater],
                                          'score':[score], 
                                          'z_score':[z_score],
                                          'source':[source],
                                          'target':[target]}
                else:
                    seg_scores[sys_id]['rater'].append(rater)
                    seg_scores[sys_id]['score'].append(score)
                    seg_scores[sys_id]['z_score'].append(z_score)
                    seg_scores[sys_id]['source'].append(source)
                    seg_scores[sys_id]['target'].append(target)
        for sys_id in seg_scores.keys():
            seg_scores[sys_id]['z_mean_score'] = np.mean(seg_scores[sys_id]['z_score'])
        
        return rater_score, seg_scores
    
    
    def generate_psqm_records_for_lang(self, lang):
        rater_score, seg_scores = self.get_psqm_segments(lang)
        src_segments, ref_segments = self.get_ref_segments(lang)
        sys_segments = self.get_sys_segments(lang)
        
        n_records = 0
        skipped_n_records = 0
        with open(self.target_path, "a+") as dest_file:
            for sys_id in seg_scores.keys():
                sys_name, seg_id = sys_id
                raw_score = 'n/a'
                z_score = seg_scores[sys_id]['z_score']
                
                # The "-1" is necessary because seg_id starts counting at 1.
                src_segment = src_segments[seg_id - 1]
                ref_segment = ref_segments[seg_id - 1]
                if sys_name not in sys_segments:
                    print(sys_name, lang)
                else:
                    sys_segment = sys_segments[sys_name][seg_id - 1]

                if not src_segment or not sys_segment:
                    logger.info("* Missing value!")
                    logger.info("* System: {}".format(sys_name))
                    logger.info("* Segment:" + str(seg_id))
                    logger.info("* Source segment:" + src_segment)
                    logger.info("* Sys segment:" + sys_segment)
                    logger.info("* Parsed line:" + line)
                    logger.info("* Lang:" + lang)
                example = self.to_json(self.year, lang, src_segment, 
                                             ref_segment, sys_segment, raw_score, 
                                             z_score, seg_id, sys_name)
                dest_file.write(example)
                dest_file.write("\n")
                n_records += 1
        logger.info("Processed {} records of {}'s {}".format(str(n_records), self.year, lang))
        logger.info("Skipped {} records of {}'s {}".format(str(skipped_n_records), self.year, lang))
        
        return n_records
    
    def to_json(self, year, lang, src_segment, ref_segment, sys_segment,
                raw_score, z_score, seg_id, sys_name, n_ratings=0):
        """Converts record to JSON."""
        if self.args.use_avg_seg_scores and (self.args.onlyMQM or self.args.onlyPSQM): 
            json_dict = {"year": year, "lang": lang, "source": src_segment, 
                         "reference": ref_segment, "candidate": sys_segment, "z_rating": z_score,
                         "rating": raw_score, "segment_id": seg_id, "system": sys_name,
                         "n_ratings": n_ratings}
            return json.dumps(json_dict)
        else:
            json_dict = {"year": year, "lang": lang, "source": src_segment, 
                         "reference": ref_segment, "candidate": sys_segment, "raw_rating": raw_score,
                         "rating": z_score, "segment_id": seg_id, "system": sys_name,
                         "n_ratings": n_ratings}
            return json.dumps(json_dict)
    
    def generate_records_for_lang(self, lang):
        """Consolidates all the files for a given language pair and year."""
        
        if self.onlyMQM:
            n_records = self.generate_mqm_records_for_lang(lang)
            return n_records
        elif self.onlyPSQM:
            n_records = self.generate_psqm_records_for_lang(lang)
            return n_records
        
        if self.args.priorMQM == False :
            raise NotImplementedError
        if lang in ['en-de', 'zh-en']:
            _, MQMsegments = self.get_mqm_segments(lang)
        
        
        # Loads source, reference and system segments.
        src_segments, ref_segments = self.get_ref_segments(lang)
        sys_segments = self.get_sys_segments(lang)

        # Streams the rating file and performs the join on-the-fly.
        ratings_file_path = self.get_ratings_path(lang)
#         logger.info("Reading file {}".format(ratings_file_path))
        n_records = 0
        skipped_n_records = 0
        
        if lang in ['en-zh', 'en-ja', 'en-iu', 'en-cs', 'en-ta', 'en-ru', 'en-de', 'en-pl'] and (not self.include_unreliables):
            return 0
        
        with open(ratings_file_path, "r", encoding="utf-8") as f_ratings:
            with open(self.target_path, "a+") as dest_file:
                for line in itertools.islice(f_ratings, 1, None):
                    line = line.rstrip()
                    parsed_tuple = self.parse_rating(line, lang)
                    sys_name, seg_id, raw_score, z_score, n_ratings = parsed_tuple
                    
                    if sys_name == 'HUMAN.0' and lang == 'de-en':
                        skipped_n_records += 1
                        continue
                    if sys_name == 'HUMAN.0' and lang == 'zh-en':
                        skipped_n_records += 1
                        continue
                    if sys_name == 'HUMAN.0' and lang == 'ru-en':
                        skipped_n_records += 1
                        continue
                    if sys_name == 'HUMAN-B':
                        sys_name == 'Human-B.0'
                    if sys_name == 'Huoshan-Translate.1470' and lang == 'ps-en':
                        sys_name = 'Huoshan_Translate.1470'
                    if sys_name == 'Facebook-AI.729' and lang == 'iu-en':
                        sys_name = 'Facebook_AI.729'
                    if sys_name == 'Huawei-TSC.1533' and lang == 'ps-en':
                        sys_name = 'Huawei_TSC.1533'
                    if sys_name == 'UQAM-TanLe.520' and lang == 'iu-en':
                        sys_name = 'UQAM_TanLe.520'
                    if sys_name == 'HUMAN' and lang == 'ps-en':
                        sys_name = 'Human-A.0'
                    if sys_name == 'NICT-Kyoto.1220' and lang == 'iu-en':
                        sys_name = 'NICT_Kyoto.1220'
                    if sys_name == 'HUMAN' and lang == 'iu-en':
                        sys_name = 'Human-A.0'
                    if sys_name == 'Huawei-TSC.1539' and lang == 'km-en':
                        sys_name = 'Huawei_TSC.1539'
                    if sys_name == 'Huoshan-Translate.651' and lang == 'km-en':
                        sys_name = 'Huoshan_Translate.651'
                    if sys_name == 'HUMAN' and lang == 'km-en':
                        sys_name = 'Human-A.0'
                    
                    if self.args.priorMQM and self.args.onlyDA == False and lang in ['en-de', 'zh-en']:
                        if (sys_name, seg_id) in MQMsegments:
                            skipped_n_records += 1
                            continue
                    
                    # The "-1" is necessary because seg_id starts counting at 1.
                    src_segment = src_segments[seg_id - 1]
                    ref_segment = ref_segments[seg_id - 1]
                    sys_segment = sys_segments[sys_name][seg_id - 1]
                    
                    if not src_segment or not sys_segment:
                        logger.info("* Missing value!")
                        logger.info("* System: {}".format(sys_name))
                        logger.info("* Segment:" + str(seg_id))
                        logger.info("* Source segment:" + src_segment)
                        logger.info("* Sys segment:" + sys_segment)
                        logger.info("* Parsed line:" + line)
                        logger.info("* Lang:" + lang)
                    example = Importer18.to_json(self.year, lang, src_segment, 
                                                 ref_segment, sys_segment, raw_score, 
                                                 z_score, seg_id, sys_name, n_ratings)
                    dest_file.write(example)
                    dest_file.write("\n")
                    n_records += 1
        logger.info("Processed {} records of {}'s {}".format(str(n_records), self.year, lang))
        logger.info("Skipped {} records of {}'s {}".format(str(skipped_n_records), self.year, lang))
        return n_records
    
    
    


# In[10]:


class Importer21(Importer18):
    """Importer for WMT20 Metrics challenge."""
    
    def __init__(self, year, target_path, cache_path, args):
        super(Importer21, self).__init__(year, target_path, cache_path, args)
        self.args = args
        self.tasks = ['challengeset', 'florestest2021', 'newstest2021', 'tedtalks']
    
    def open_tar(self, tar_path, open_dir):
        logger.info("Untaring...")
        tar = tarfile.open(tar_path)
        tar.extractall(path=open_dir)
        tar.close()
        logger.info("Done.")
    
    def fetch_files(self):
        """Downloads raw datafiles from various WMT sources."""
        return
#         cache = self.cache_path
        
#         if cache and not os.path.isdir(cache):
#             logger.info("Initializing cache {}".format(cache))
#             os.makedirs(cache)
        
#         for file_type in self.location_info:    
#             folder_name, _, url = self.location_info[file_type]
#             cache_untar_path = os.path.join(cache, folder_name)
            
#             if cache:
#                 logger.info("Checking cached tar file {}.".format(cache_untar_path))
#                 if os.path.exists(cache_untar_path) :
#                     if file_type == 'submissions':
#                         tars = os.path.join(cache_untar_path, '*.tar.gz')
#                         tar_paths = glob.glob(tars)
#                         untar_paths = [path.replace(".tar.gz", "", 1) for path in tar_paths]
#                         for tar_path, untar_paths in zip(tar_paths, untar_paths):
#                             if not os.path.exists(untar_paths):
#                                 self.open_tar(tar_path, cache_untar_path)
#                             else:
#                                 logger.info("Cache and untar directory found, skipping")
#                     else:
#                         logger.info("Cache and untar directory found, skipping")
#                     continue
#             logger.info("File not found in cache.")
#             if file_type == 'submissions':
#                 logger.info("Cannot download {} with this script. Download from {}".format(folder_name, url))
#                 exit(-1)
#             logger.info("Downloading {} from {}".format(folder_name, url))
#             git_clone(url, cache_untar_path)
#             logger.info("Done.") 
    
    def parse_rating(self, rating_line, lang):
        rating_tuple = tuple(rating_line.split(" "))
        # I have a feeling that the last field is the number of ratings
        # but I'm not 100% sure.
        sys_name, seg_id, raw_score, z_score, n_ratings = rating_tuple
        
        # en-zh has unknown seg_id probablly tagged with other format name
#         if lang in ['en-zh', 'en-ja', 'en-iu', 'en-cs', 'en-ta', 'en-ru', 'en-de', 'en-pl']:
#             seg_id = seg_id.split('_')[-1] 
        try:
            seg_id = int(seg_id)
            raw_score = float(raw_score)
            z_score = float(z_score)
            n_ratings = int(n_ratings)
        except:
            logger.info(lang)
            logger.info(rating_line)
        return sys_name, seg_id, raw_score, z_score, n_ratings

    def parse_submission_file_name(self, fname, lang, task):
        """Extracts system names from the name of submission files."""

        # I added those rules to unblock the pipeline.

        sys_name = fname.replace("{}.{}.{}.".format(task, lang, 'hyp'), "", 1).replace(".{}".format(lang.split('-')[-1]), "", 1)

        return sys_name
    
    def parse_source_file_name(self, fname, task):
        lang = fname.replace("{}.".format(task), "")
        return lang[:5]
    
    def list_lang_pairs(self, task):
        """List all language pairs included in the WMT files for 2020."""
        
        folder_name, _, _ = self.location_info["submissions"]
        folder = os.path.join(self.temp_directory, folder_name, 'sources', '*')
        all_full_paths = glob.glob(folder)
        all_files = [os.path.basename(f) for f in all_full_paths if os.path.basename(f).startswith(task)]
        cand_lang_pairs = [self.parse_source_file_name(fname, task) for fname in all_files]
        # We need to remove None values in cand_lang_pair:
        lang_pairs = [lang_pair for lang_pair in cand_lang_pairs if lang_pair]
        return list(set(lang_pairs))

    def get_ratings_path(self, lang, task):
        folder, _, _ = self.location_info["eval_data"]

        file_name = "ad-seg-scores-{}.csv".format(lang)
        folder = os.path.join(self.temp_directory, folder, "manual-evaluation", "DA", "ad-seg-scores-*.csv")
        all_files = glob.glob(folder)
        for cand_file in all_files:
            if cand_file.endswith(file_name):
                return cand_file
        raise ValueError("Can't find ratings for lang {}".format(lang))

    def get_ref_segments(self, lang, task):
        """Fetches source and reference translation segments for language pair."""
        folder, _, _ = self.location_info["submissions"]
        src_lang, tgt_lang = separate_lang_pair(lang)
        src_subfolder = 'sources'
        ref_subfolder = 'references'
        src_file = "{}.{}-{}.src.{}".format(task, src_lang, tgt_lang, src_lang)
        ref_file = "{}.{}-{}.ref.ref-A.{}".format(task, src_lang, tgt_lang, tgt_lang)
        src_path = os.path.join(self.temp_directory, folder, src_subfolder, src_file)
        ref_path = os.path.join(self.temp_directory, folder, ref_subfolder, ref_file)

#         logger.info("Reading data from files {} and {}".format(src_path, ref_path))
        with open(src_path, "r", encoding="utf-8") as f_src:
            src_segments = f_src.readlines()
        with open(ref_path, "r", encoding="utf-8") as f_ref:
            ref_segments = f_ref.readlines()

        src_segments = [postprocess_segment(s) for s in src_segments]
        ref_segments = [postprocess_segment(s) for s in ref_segments]
        return src_segments, ref_segments

    def get_sys_segments(self, lang, task):
        """Builds a dictionary with the generated segments for each system."""
        # Gets all submission file paths.
        folder_name, _, _ = self.location_info["submissions"]
        subfolder = os.path.join("system-outputs", task)
        folder = os.path.join(self.temp_directory, folder_name, subfolder, lang)
        all_files = os.listdir(folder)
#         logger.info("Reading submission files from {}".format(folder))

        # Extracts the generated segments for each submission.
        sys_segments = {}
        for sys_file_name in all_files:
            sys_name = self.parse_submission_file_name(sys_file_name, lang, task)
            assert sys_name
            sys_path = os.path.join(folder, sys_file_name)
            with open(sys_path, "r", encoding="utf-8") as f_sys:
                sys_lines = f_sys.readlines()
                sys_lines = [postprocess_segment(s) for s in sys_lines]
                sys_segments[sys_name] = sys_lines

        return sys_segments
    
    def generate_records_for_lang(self, lang, task):
        """Consolidates all the files for a given language pair and year."""
        # Loads source, reference and system segments.
        src_segments, ref_segments = self.get_ref_segments(lang, task)
        sys_segments = self.get_sys_segments(lang, task)
    
        # Streams the rating file and performs the join on-the-fly.
#         ratings_file_path = self.get_ratings_path(lang)
#         logger.info("Reading file {}".format(ratings_file_path))
        n_records = 0
        skipped_n_records = 0
#         if lang in ['en-zh', 'en-ja', 'en-iu', 'en-cs', 'en-ta', 'en-ru', 'en-de', 'en-pl'] and (not self.include_unreliables):
#             return 0

        with open(self.target_path, "a+") as dest_file:
            for sys_name in sys_segments.keys():
                for seg_id in range(len(src_segments)):
                    src_segment = src_segments[seg_id]
                    ref_segment = ref_segments[seg_id]
                    sys_segment = sys_segments[sys_name][seg_id]
                    example = Importer18.to_json(self.year, lang, src_segment, 
                                                 ref_segment, sys_segment, 0.0, 
                                                 0.0, seg_id, sys_name, 0)
                    dest_file.write(example)
                    dest_file.write("\n")
                    n_records += 1
        logger.info("Processed {} records of {}'s {}".format(str(n_records), self.year, lang))
        logger.info("Skipped {} records of {}'s {}".format(str(skipped_n_records), self.year, lang))
        return n_records
    
    
    


# In[11]:


# importer = Importer21('21', 
#                       '/home/ksudoh/kosuke-t/scripts/make_data/wmt_metrics_data/data/WMT21_test.json', 
#                       '/home/ksudoh/kosuke-t/scripts/make_data/wmt_metrics_data/cache',
#                       args=None)


# In[12]:


# importer.fetch_files()


# In[13]:


# for task in importer.tasks:
#     for lang in importer.list_lang_pairs(task):
#         logger.info('{}, {}'.format(lang, task))
#         importer.generate_records_for_lang(lang, task)
# #         importer.generate_records_for_lang(lang, task)


# In[142]:





# In[ ]:




