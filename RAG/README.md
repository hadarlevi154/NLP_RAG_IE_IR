Hi! I'm glad you are here after reading my email!
I really hope you will be impressed :)

# Retrieval-Augmented Generation (RAG) System

A comprehensive information retrieval system enhanced with generative AI capabilities to provide more relevant and natural responses to user queries.
The corpus consists of research documents and medical papers about cystic fibrosis, including studies on its genetic factors, treatment approaches, etc.

## Overview

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline that:

1. **Retrieves** relevant documents using state-of-the-art information retrieval methods
2. **Augments** a large language model with the retrieved context
3. **Generates** comprehensive, accurate responses based on the documents

The system supports two retrieval methods:
- TF-IDF (Term Frequency-Inverse Document Frequency)
- BM25 (Best Matching 25)

## Features

- Create inverted indices from XML document collections
- Query the index using either TF-IDF or BM25 ranking algorithms
- Generate responses using OpenAI's GPT-4o-mini (or fall back to extractive sum)
- Command-line interface for index creation and querying
- Detailed console output showing retrieved documents and scores
- Results saved to files for further analysis

## Installation

### Prerequisites

- Python 3.x
- NLTK
- OpenAI Python library (v1.0.0+)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/rag-system.git
   cd rag-system
   ```

2. Install required packages:
   ```
   pip install nltk openai
   ```

3. Download NLTK data:
   ```python
   import nltk
   nltk.download('stopwords')
   ```

4. Set your OpenAI API key:
   Edit the `api_key` variable in `InformationRetrievalGivenQuery.py` or set it as an environment variable.

## Usage

### Create an Index

```
python vsm_ir.py create_index path/to/corpus
```

Where `path/to/corpus` is a directory containing XML files with documents in the expected format.

### Execute a Query

```
python vsm_ir.py query tfidf vsm_inverted_index.json "your search query" [path/to/corpus]
```

or 

```
python vsm_ir.py query bm25 vsm_inverted_index.json "your search query" [path/to/corpus]
```

Where:
- `tfidf` or `bm25` specifies the retrieval method
- `vsm_inverted_index.json` is the path to the index file
- `"your search query"` is the search query in quotes
- `[path/to/corpus]` is the optional path to the corpus (needed to extract full document text)

Possible queries:
- Is salt (sodium and/or chloride) transport/permeability abnormal in CF?
- What abnormalities of insulin secretion or insulin metabolism occur in CF patients?
- Can CF be diagnosed prenatally? 

## How It Works

### Indexing (create_index)

1. Reads XML documents from the corpus directory
2. Tokenizes document text, removes stopwords, and applies stemming
3. Builds an inverted index with term frequencies
4. Calculates IDF values for terms
5. Saves the index to a JSON file

### Retrieval (query)

1. Processes the query using the same tokenization, stopword removal, and stemming
2. Calculates document scores using either TF-IDF or BM25
3. Ranks documents by relevance score
4. Extracts the full text of the top documents from the XML files
5. Saves the top document IDs to a file

### Generation

1. Formats the retrieved documents as context
2. Sends the query and context to OpenAI's GPT-4o-mini model
3. If API call fails, falls back to a rule-based extractive summary
4. Prints and saves the generated response

## File Descriptions

- `vsm_ir.py`: Main entry point with command-line interface
- `InformationRetrievalGivenQuery.py`: Core RAG implementation
- `InvertedIndexDictionary.py`: Index creation and storage

## Output Files

- `vsm_inverted_index.json`: The created index file
- `ranked_query_docs.txt`: Top document IDs for the query
- `rag_generated_response.txt`: The generated response

## Document Format

The system expects XML files with the following structure:

```xml
<ROOT>
  <RECORD>
    <RECORDNUM>document_id</RECORDNUM>
    <TITLE>document title</TITLE>
    <ABSTRACT>document abstract</ABSTRACT>
    <EXTRACT>document extract</EXTRACT>
    <!-- other optional fields -->
  </RECORD>
  <!-- more records -->
</ROOT>
```

## Customization

- In `InformationRetrievalGivenQuery.py`:
  - Change `MODEL_NAME` to use a different OpenAI model
  - Adjust `max_tokens` to control response length
  - Modify the prompt template to change the response style

## Troubleshooting

- **OpenAI API Errors**: Check your API key and network connection
- **Document Not Found**: Verify corpus path and XML structure
- **Missing Fields**: Ensure XML documents contain the expected fields

## Future Improvements

- Add embedding-based retrieval methods
- Implement semantic search capabilities
- Add a web interface for easier interaction
- Support more document formats
- Include relevance feedback mechanisms
