import math
import xml.etree.ElementTree as ET
import nltk
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import os
from collections import Counter
import json
from nltk.stem import PorterStemmer

OUTPUT_FILE = "vsm_inverted_index.json"

class InvertedIndexDictionary:

    def __init__(self, path_to_xml_dir):
        self.path_to_xml_dir = path_to_xml_dir
        self.dict = {}
        try:
            self.stop_words = set(stopwords.words("english"))
        except:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words("english"))

        self.stop_words.add("\n")
        self.count_of_docs = 0

    def build_inverted_index(self):
        """
           Builds the complete inverted index from XML files in the specified directory.

           This function:
           1. Processes all XML files in the directory
           2. Extracts term frequencies for each document
           3. Calculates document lengths
           4. Computes IDF values for both TF-IDF and BM25 models
           5. Constructs the final dictionary with all components

           The resulting dictionary contains:
           - TF: Term frequency mapping for each term and document
           - len_by_doc_name: Length of each document
           - normal_IDF: Standard IDF values for TF-IDF calculation
           - BM25_IDF: IDF values for BM25 calculation
        """

        files_list = [file for file in os.listdir(self.path_to_xml_dir) if file.endswith(".xml")]
        term_frequency = {}
        normal_idf = {}
        bm25_idf = {}
        len_by_doc_name = {}
        for file in files_list:
            count_of_docs_in_file, counter_dict_for_file, doc_len_dict_for_file = self.get_inverted_index_of_file(file)
            self.count_of_docs += count_of_docs_in_file
            self.merge_two_dicts(term_frequency, counter_dict_for_file)
            len_by_doc_name.update(doc_len_dict_for_file)

        for word in term_frequency:
            normal_idf[word] = math.log2(self.count_of_docs / len(term_frequency[word]))
            bm25_idf[word] = self.get_bm25_idf(len(term_frequency[word]))

        self.dict = {"TF": term_frequency, "len_by_doc_name": len_by_doc_name, "normal_IDF": normal_idf, "BM25_IDF": bm25_idf}

    def get_inverted_index_of_file(self, file):
        """
           Processes a single XML file to extract document information and term frequencies.

           This function:
           1. Parses the XML file into a tree structure
           2. Processes each record (document) in the file
           3. Extracts and tokenizes text from each record
           4. Calculates term frequencies normalized by the most frequent term
           5. Builds document-specific inverted indices

           Returns:
               tuple: (
                        count_of_docs_in_file: Number of documents in the file,
                        counter_dict_for_file: Dictionary mapping terms to documents with normalized frequencies,
                        doc_len_dict_for_file: Dictionary mapping document IDs to their lengths
                    )
        """

        tree = ET.parse(self.path_to_xml_dir + "/" + file)
        root = tree.getroot()
        counter_dict_for_file = {}
        doc_len_dict_for_file = {}
        count_of_docs_in_file = len(root.findall("RECORD"))

        for record in root:
            counter_dict_for_doc = {}
            record_num = record.find("RECORDNUM")
            if record_num == None:
                continue
            len_of_doc, words = self.get_tokenized_words_from_record(record)
            doc_len_dict_for_file[record_num.text] = len_of_doc
            term_freq_dict_for_doc = Counter(words)
            most_freq_term = max(term_freq_dict_for_doc.values())

            for word, count in term_freq_dict_for_doc.items():
                counter_dict_for_doc[word] = {record_num.text: count/most_freq_term}
            self.merge_two_dicts(counter_dict_for_file, counter_dict_for_doc)

        return count_of_docs_in_file, counter_dict_for_file, doc_len_dict_for_file

    def get_tokenized_words_from_record(self, record):
        """
           Extracts and combines text from different fields of an XML record.

           This function:
           1. Extracts text from title, abstract, and extract fields if present
           2. Concatenates the text with appropriate spacing
           3. Passes the combined text to tokenization and preprocessing

           Returns:
               tuple: Result from get_tokenized_filtered_and_stemmed_words containing
                      document length and list of processed words
        """

        text = ""
        title = record.find("TITLE")
        abstract = record.find("ABSTRACT")
        extract = record.find("EXTRACT")

        if title is not None:
            text += str(title.text)
        if abstract is not None:
            text = " ".join([text, str(abstract.text)])
        if extract is not None:
            text = " ".join([text, str(extract.text)])

        return self.get_tokenized_filtered_and_stemmed_words(text)

    def get_tokenized_filtered_and_stemmed_words(self, text):
        """
           Processes text through tokenization, stopword removal, and stemming.

           This function:
           1. Tokenizes text using regular expression to extract word characters
           2. Converts all text to lowercase for consistency
           3. Removes common stopwords to focus on meaningful terms
           4. Applies Porter stemming to reduce words to their root forms

           Returns:
               tuple: (
                   len(words): Original count of words before filtering,
                   filtered_words: List of stemmed words with stopwords removed
               )
        """

        tokenizer = RegexpTokenizer(r'\w+')
        porter_stemmer = PorterStemmer()

        words = tokenizer.tokenize(text.lower())
        filtered_words = [porter_stemmer.stem(word) for word in words if not word in self.stop_words]

        return len(words), filtered_words

    @staticmethod
    def merge_two_dicts(dict1, dict2):
        for word in dict2:
            if word in dict1:
                dict1[word].update(dict2[word])
            else:
                dict1[word] = dict2[word]

    def save_data_to_files(self, path=""):
        with open(path + OUTPUT_FILE, "w") as outfile:
            json.dump(self.dict, outfile)

    @staticmethod
    def load_data_from_files(path=""):
        with open(path + OUTPUT_FILE) as json_file:
            return json.load(json_file)

    def get_bm25_idf(self, n_word):  # n_word is the number of documents in which "word" appears
        N = self.count_of_docs
        return math.log(((N - n_word + 0.5) / (n_word + 0.5)) + 1)
