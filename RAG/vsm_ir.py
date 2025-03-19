import sys
import os
from InformationRetrievalGivenQuery import ret_info
from InvertedIndexDictionary import InvertedIndexDictionary


def create_index():
    """
       Creates and saves an inverted index from the corpus documents.

       This function:
       1. Validates command-line arguments
       2. Constructs the dictionary from corpus documents
       3. Builds the inverted index with term frequencies
       4. Saves the index to disk for later retrieval
    """

    if len(sys.argv) < 3:
        print("Usage: python vsm_ir.py create_index <path_to_corpus>")
        return

    input_files_path = sys.argv[2] if sys.argv[2].endswith("/") else sys.argv[2] + "/"
    inverted_index = InvertedIndexDictionary(input_files_path)
    inverted_index.build_inverted_index()
    inverted_index.save_data_to_files()
    # Reference the constant from the module instead of as a class attribute
    from InvertedIndexDictionary import OUTPUT_FILE
    print(f"Index created successfully and saved to {OUTPUT_FILE}")


def query():
    """
       Processes a search query using the specified retrieval method.

       This function:
       1. Validates command-line arguments
       2. Loads the inverted index from disk
       3. Calls ret_info to execute retrieval and generation
       4. Handles exceptions with informative error messages
    """

    if len(sys.argv) < 5:
        print("Usage: python vsm_ir.py query <method> <index_path> \"<query_text>\" [<corpus_path>]")
        print("  method: either 'tfidf' or 'bm25'")
        print("  index_path: path to the index file")
        print("  query_text: the search query in quotes")
        print("  corpus_path: (optional) path to the corpus directory")
        return

    try:
        from InvertedIndexDictionary import OUTPUT_FILE
        dic = InvertedIndexDictionary.load_data_from_files(path="" if sys.argv[3] == OUTPUT_FILE else sys.argv[3])
        ret_info(dic)
    except Exception as e:
        print(f"Error during query processing: {str(e)}")


def show_help():
    print("Vector Space Model Information Retrieval with RAG")
    print("------------------------------------------------")
    print("Commands:")
    print("  create_index <path_to_corpus>")
    print("    Creates and saves an inverted index from the documents in the specified directory")
    print()
    print("  query <method> <index_path> \"<query_text>\" [<corpus_path>]")
    print("    Performs a search query and generates a response using RAG")
    print("    method: either 'tfidf' or 'bm25'")
    print("    index_path: path to the index file")
    print("    query_text: the search query in quotes")
    print("    corpus_path: (optional) path to the corpus directory")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        show_help()
    elif sys.argv[1] == "create_index":
        create_index()
    elif sys.argv[1] == "query":
        query()
    else:
        show_help()