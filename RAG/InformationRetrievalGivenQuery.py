import sys
import json
import math
import os
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from operator import itemgetter

# Import the OpenAI client from the new SDK
from openai import OpenAI

api_key = ""  # Replace with actual API key

q_dict_tfidf = {}
q_dict_bm25 = {}
k = 4
b = 0.0001
CORPUS_DIRECTORY = 'cfc-xml_corrected'  # Default corpus directory
MODEL_NAME = "gpt-4o-mini"  # Used the more affordable model


def simplify_q_input():
    """
       Processes query text for search operations.

       Steps:
       1. Converts input to lowercase
       2. Removes stopwords
       3. Tokenizes into words
       4. Stems words to root forms

       Returns:
           list: Stemmed words from query with stopwords removed
       """

    input_text = sys.argv[4].lower()
    stop_words = set(stopwords.words("english"))
    tokenizer = RegexpTokenizer(r'\w+')
    ps = PorterStemmer()
    question = tokenizer.tokenize(input_text)
    filtered_question = [word for word in question if word not in stop_words]
    return [ps.stem(word) for word in filtered_question]


def tf_idf(q, inverted_index, normal_IDF, all_docs_len):
    """
       Calculates TF-IDF similarity scores between query and documents.

       Implements vector space model with cosine similarity:
       1. Calculates document weights using TF-IDF scoring
       2. Computes query term weights
       3. Determines cosine similarity between query and each document

        Returns:
        list: Ranked document IDs and scores sorted by relevance
    """

    Wq = {}
    cossim = {}
    for w in inverted_index:
        if q_dict_tfidf.get(w) is None:
            q_dict_tfidf[w] = {key: inverted_index[w][key] * normal_IDF[w] for key in inverted_index[w]}

    max_word = max((q.count(W_i) for W_i in q), default=0)  # Handle empty query case

    for W_i in q:
        count = q.count(W_i)
        if W_i in normal_IDF:
            Wq[W_i] = (count / max_word) * normal_IDF[W_i]

    for key in all_docs_len:
        numer, sum_1, sum_2 = 0, 0, 0
        for W_i in inverted_index:
            if W_i in q_dict_tfidf and key in q_dict_tfidf[W_i]:
                sum_1 += q_dict_tfidf[W_i].get(key, 0) ** 2
        for W_i in q:
            if W_i in q_dict_tfidf and key in q_dict_tfidf.get(W_i, {}):
                numer += q_dict_tfidf.get(W_i, {}).get(key, 0) * Wq.get(W_i, 0)
                sum_2 += Wq.get(W_i, 0) ** 2

        denom = math.sqrt(sum_1 * sum_2) if sum_1 * sum_2 > 0 else 1
        cossim[key] = numer / denom if denom != 0 else 0

    return sorted(cossim.items(), key=itemgetter(1), reverse=True)


def compute_avgdl(all_docs_len):
    """
       Calculates the average document length.

       Returns:
           float: Average document length, or 0 if no documents
    """

    return sum(all_docs_len.values()) / len(all_docs_len) if all_docs_len else 0


def q_bm25(q, inverted_index, all_docs_len, bm25_IDF, avgdl):
    """
       Calculates BM25 scores for documents based on query terms.

       Implements the BM25 ranking algorithm:
       1. For each document, computes term score contributions
       2. Uses term frequency, document length, and IDF values
       3. Applies k and b parameters for term frequency scaling and length normalization

       Returns:
           list: Documents ranked by BM25 score in descending order
       """

    for key in all_docs_len:
        q_dict_bm25[key] = 0
        for W_i in q:
            try:
                if W_i in inverted_index and key in inverted_index[W_i]:
                    numer = inverted_index[W_i][key] * (k + 1)
                    denom = inverted_index[W_i][key] + k * (1 - b + b * (all_docs_len[key] / avgdl))
                    q_dict_bm25[key] += bm25_IDF[W_i] * (numer / denom)
            except Exception as e:
                continue
    return sorted(q_dict_bm25.items(), key=itemgetter(1), reverse=True)


def get_document_text_from_xml(doc_id, corpus_dir):
    """
       Extracts document text from XML files in the corpus directory.

       This function:
       1. Searches through all XML files in the specified directory
       2. Locates the record with the matching document ID
       3. Extracts text from title, abstract and extract fields
       4. Combines them into a formatted document text

       Returns:
           str: Formatted document text or error message if not found
    """

    import xml.etree.ElementTree as ET
    import os

    # Find which XML file contains the document
    for file in os.listdir(corpus_dir):
        if not file.endswith(".xml"):
            continue

        try:
            tree = ET.parse(os.path.join(corpus_dir, file))
            root = tree.getroot()

            for record in root.findall("RECORD"):
                record_num = record.find("RECORDNUM")
                if record_num is not None and record_num.text == doc_id:
                    # Extract the text from various fields
                    text_parts = []

                    title = record.find("TITLE")
                    if title is not None and title.text:
                        text_parts.append(f"Title: {title.text}")

                    abstract = record.find("ABSTRACT")
                    if abstract is not None and abstract.text:
                        text_parts.append(f"Abstract: {abstract.text}")

                    extract = record.find("EXTRACT")
                    if extract is not None and extract.text:
                        text_parts.append(f"Extract: {extract.text}")

                    return "\n\n".join(text_parts)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    return f"Document {doc_id} not found"


def generate_response_with_openai(query, documents):
    """
       Generates a response to the query using OpenAI's model.

       This function:
       1. Formats retrieved documents as context for the model
       2. Creates a system prompt instructing the model's behavior
       3. Makes an API call to OpenAI with appropriate parameters
       4. Falls back to extractive sum if the API call fails

       Returns:
           str: Generated response or extractive summary if API fails
       """

    if not documents:
        return "No relevant documents found for your query."

    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        # Format the context from documents
        context = "\n\n".join([f"Document {i + 1}:\n{doc}" for i, doc in enumerate(documents)])

        # Create the prompt for the model
        prompt = (
            "You are a helpful research assistant. Answer the question based only on the provided documents. "
            "If the documents don't contain the relevant information, say so rather than making up an answer."
        )

        # Call the OpenAI API
        response = client.chat.completions.create(
            model=MODEL_NAME,  # Using GPT-4o-mini
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Context documents:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"}
            ],
            temperature=0.3,  # Lower temperature for more focused responses
            max_tokens=300  # Limit token usage to reduce costs
        )

        # Extract the generated response
        return response.choices[0].message.content

    except Exception as e:
        print(f"OpenAI API error: {str(e)}")
        print("Falling back to extractive summary...")
        return generate_extractive_response(query, documents)


def generate_extractive_response(query, documents):
    """
       Creates a simple extractive summary from documents when API generation fails.

       This function:
       1. Extracts and normalizes sentences from all documents
       2. Ranks sentences by relevance to query keywords
       3. Selects top relevant sentences or defaults to first sentences
       4. Formats them into a coherent response

       Returns:
           str: Extractive summary based on the most relevant sentences
       """

    sentences = []
    for doc in documents:
        doc_sentences = doc.replace("\n", " ").split(". ")
        sentences.extend([s.strip() for s in doc_sentences if s.strip()])

    # Simple keyword matching to find relevant sentences
    query_keywords = query.lower().split()
    relevant_sentences = []

    for sentence in sentences:
        # Count keyword matches
        match_count = sum(1 for keyword in query_keywords if keyword.lower() in sentence.lower())
        if match_count > 0:
            relevant_sentences.append((sentence, match_count))

    # Sort by relevance and take top sentences
    relevant_sentences.sort(key=lambda x: x[1], reverse=True)
    top_sentences = [s[0] for s in relevant_sentences[:3]]

    if not top_sentences:
        # If no matching sentences, just take the first few from the documents
        top_sentences = sentences[:3] if len(sentences) >= 3 else sentences

    response = f"Based on the retrieved documents:\n\n"
    response += ". ".join(top_sentences)
    if not response.endswith("."):
        response += "."

    return response

def ret_info(dict):
    """
       Main function that executes the RAG pipeline.

       This function:
       1. Retrieves and ranks documents using either TF-IDF or BM25
       2. Extracts the full text of top documents
       3. Generates a response using OpenAI model or fallback method
       4. Outputs results to console and files
    """

    inverted_index = dict["TF"]
    all_docs_len = dict["len_by_doc_name"]
    normal_IDF = dict["normal_IDF"]
    bm25_IDF = dict["BM25_IDF"]
    avgdl = compute_avgdl(all_docs_len)

    q = simplify_q_input()
    if not q:
        print("Error in query question")
        return

    retrieval_method = sys.argv[2]
    if retrieval_method == "tfidf":
        relevant_docs = tf_idf(q, inverted_index, normal_IDF, all_docs_len)
    elif retrieval_method == "bm25":
        relevant_docs = q_bm25(q, inverted_index, all_docs_len, bm25_IDF, avgdl)
    else:
        print("Invalid retrieval method")
        return

    # Save top documents to file
    with open("ranked_query_docs.txt", "w") as f:
        for doc in relevant_docs[:10]:  # Save top 10 documents
            f.write(f"{doc[0]}\n")

    top_docs = [doc[0] for doc in relevant_docs[:3]]  # Get top 3 relevant docs

    # Get corpus directory from command line if provided
    corpus_dir = CORPUS_DIRECTORY
    if len(sys.argv) > 5:
        corpus_dir = sys.argv[5]

    # Get the actual document texts
    document_texts = [get_document_text_from_xml(doc_id, corpus_dir) for doc_id in top_docs]

    # Print the top retrieved documents
    print("\n=== Top Retrieved Documents ===")
    for i, (doc_id, text) in enumerate(zip(top_docs, document_texts)):
        print(f"\nDocument {i + 1} (ID: {doc_id})")
        print(f"Relevance Score: {relevant_docs[i][1]:.4f}")
        print("-" * 40)
        # Print a snippet of the text
        snippet = text[:300] + "..." if len(text) > 300 else text
        print(snippet)

    # Generate response using OpenAI GPT-4o-mini
    original_query = sys.argv[4]
    print(f"\n=== Generating RAG response with {MODEL_NAME} ===")
    rag_response = generate_response_with_openai(original_query, document_texts)

    # Print and save results
    print("\n=== RAG-Generated Response ===\n")
    print(rag_response)
    with open("rag_generated_response.txt", "w") as f:
        f.write(rag_response)