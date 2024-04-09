# various processing for training data

import csv
from bs4 import BeautifulSoup
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SimpleDirectoryReader
from transformers import GPT2Tokenizer
import ebooklib
from ebooklib import epub
from llama_index.llms.openai import OpenAI

embed_model = OpenAIEmbedding()

def extract_text_from_epub(epub_file):
    print('---- Extracting text...')
    output_file = 'txt_from_epub.txt'
    book = epub.read_epub(epub_file)
    text_content = []

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:

            # Decode the content from bytes to string using UTF-8 encoding
            content = item.get_content().decode('utf-8')
            soup = BeautifulSoup(content, 'html.parser')

            text_content.append(soup.get_text().replace('\n','').replace('\t',''))
    result = '\n'.join(text_content)
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        file.write(result)

def split_text_into_chunks(input_file, output_file, chunk_size=500):
    print('---- Chunking text...')
    token_counts = []
    openai_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    documents = SimpleDirectoryReader(input_files=[input_file]).load_data()
    # decrease breakpoint as much as possible without getting single outlier sentences
    splitter = SemanticSplitterNodeParser(
        buffer_size=1, breakpoint_percentile_threshold=93, embed_model=embed_model
    )

    chunks = splitter.get_nodes_from_documents(documents)
    column_names = ['text']

    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(column_names)

        for chunk in chunks:
            chunk_text = chunk.get_content()
            if(len(chunk_text) < 100): continue

            #count tokens
            nr_tokens = len(openai_tokenizer.tokenize(chunk_text))
            token_counts.append(nr_tokens)

            writer.writerow([chunk_text])
        print('Max chunk is of (tokens): ', max(token_counts) , '. Max for gpt 3.5turbo is 4096 total.')

def gen_synthetic_queries(csvfile):
    print('---- Generating synthetic queries...')
    import re
    import json
    llm = OpenAI(model='gpt-3.5-turbo', response_format={ "type": "json_object" }
)

    with open(csvfile, 'r') as file:
        csv_reader = csv.reader(file)

        prompt_template = """\
          Context information is below.

          ---------------------
          {context_str}
          ---------------------

          Based on the context information.
          generate technical {num_questions_per_chunk} questions referring to a company's reporting \
          and answer each of them using the techniques described in the context. Pretend you are analyzing a company. Be specific about that company. \
          Do not be general in your questions, be specific. \
          Generate JSON for each question / answer pair. Separate pairs with ---.\
          Restrict the questions and answers to the context information provided."
          """

        queries = {}
        relevant_docs = {}
        with open('training_data/output.csv', 'a') as file_w:
            writer = csv.writer(file_w)
            column_names = ['text']
            writer.writerow(column_names)

            for ix, text in enumerate(csv_reader):
                text = text[0]
                if(text=='text'): continue
                #if(ix == 6): break # for sampling TODO: remove in prod

                # generate question with LLM
                query = prompt_template.format(context_str=text, num_questions_per_chunk=4)
                response = llm.complete(query)
                pairs = str(response).strip().split("---")

                pairs = [
                    re.sub(r"^\d+[\).\s]", "", question).strip() for question in pairs
                ]
                pairs = [question for question in pairs if len(question) > 0]

                for pair in pairs:
                    try:
                        pair = json.loads(pair)
                    except:
                        print('LLM JSON generation / parsing error! Skipping ...')
                        continue;

                    # rename keys
                    pair['human'] = pair['question']
                    del pair['question']

                    pair['bot'] = pair['answer']
                    del pair['answer']

                    # convert dict to string
                    pair_str = 'human: ' + pair['human'] + ' \n bot: ' + pair['bot']

                    writer.writerow([pair_str])
        return queries, relevant_docs

def filter_by_blacklist(csv_file):
    print('---- Filtering training data...')
    BLACKLIST = ['regression', 'r-squared']
    outfile = csv_file.split('.')[0] + '_filtered.csv'

    with open(csv_file, 'r', newline='') as csv_in, \
         open(outfile, 'w', newline='') as csv_out:
        reader = csv.reader(csv_in)
        writer = csv.writer(csv_out)

        for row in reader:
            # Check if the word to exclude is in any of the columns in the row
            if not any(word in ' '.join(row) for word in BLACKLIST):
                writer.writerow(row)

"""
    Main flow for generating synthetic training data from documents, for LLM QA training (huggingface autotune csv format):
    - run extract_text_from_epub('epub.epub'). This creates txt_from_epub.txt.
    - run split_text_into_chunks(input_file, output_file). This creates output.csv
    - run gen_synthetic_queries('output.csv'). This creates trainingdata/output.csv
    - filter_by_blacklist('training_data/output.csv'). This creates trainingdata/output_filtered.csv
    
    then, scan training data to remove opaque calculations and work out words: regression, r squared
    This is all done by gen_training_data:
"""

def gen_training_data(doc_filename):
    SUPPORTED_INPUT_EXT = ['epub']
    extension = doc_filename.split('.')
    extension = extension[len(extension)-1]
    if(extension not in 'epub'): raise Exception('Input file type not supported. Supported types are: ', SUPPORTED_INPUT_EXT)

    # saves to txt_from_epub.txt
    extract_text_from_epub('epub.epub')
    # saves to output.csv. You should filter out intros and other fluff here
    split_text_into_chunks('txt_from_epub.txt', 'output.csv')
    # saves to trainingdata/output.csv
    gen_synthetic_queries('output.csv')
    # saves to trainingdata/output_filtered.csv
    filter_by_blacklist('training_data/output.csv')

#gen_training_data('epub.epub')