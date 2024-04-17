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
import json
import re

embed_model = OpenAIEmbedding()


def extract_text_with_tables(html_data):
    """this parses the text but leaves tables as HTML for structure"""

    soup = BeautifulSoup(html_data, 'html.parser')

    # Initialize variables to store extracted text and HTML table content
    extracted_text = []
    extracted_tables = []

    # Process each element in the BeautifulSoup object
    for element in soup.descendants:
        # Check if the element is a table
        if element.name == 'table':
            # Append the HTML representation of the table
            extracted_tables.append(str(element))
        else:
            # If the element is not a table, extract its text content
            text = element.get_text(strip=True)
            if text:  # Append non-empty text content
                extracted_text.append(text)

    # Join the extracted text elements into a single string
    extracted_text = ' '.join(extracted_text)

    # Return the extracted text and list of HTML table representations
    return extracted_text, extracted_tables

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace and parse the JSON
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    return data

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

def extract_text_from_pdf(pdf_file):
    import fitz  # Import PyMuPDF

    texts = []
    try:
        pdf_document = fitz.open(pdf_file)

        for page_number in range(len(pdf_document)):
            page = pdf_document[page_number]

            page_text = page.get_text()
            texts.append(page_text.replace('\n',''))

        pdf_document.close()

    except Exception as e:
        print(f"PDF read occurred: {e}")

    output_file = 'txt_from_epub.txt'
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        file.write('\n'.join(texts))

def split_text_into_chunks(input_file, output_file, chunk_size=500):
    print('---- Chunking text...')
    token_counts = []
    openai_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    documents = SimpleDirectoryReader(input_files=[input_file]).load_data()
    # decrease breakpoint as much as possible without getting single outlier sentences
    # less threshold = more chunks
    splitter = SemanticSplitterNodeParser(
        buffer_size=4, breakpoint_percentile_threshold=90, embed_model=embed_model
    )

    chunks = splitter.get_nodes_from_documents(documents)
    column_names = ['text']

    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(column_names)

        for chunk in chunks:
            chunk_text = chunk.get_content()
            if(len(chunk_text) < 250): continue

            #count tokens
            nr_tokens = len(openai_tokenizer.tokenize(chunk_text))
            token_counts.append(nr_tokens)

            writer.writerow([chunk_text])
        print('Max chunk is of (tokens): ', max(token_counts) , '. Max for gpt 3.5turbo is 4096 total.')

def gen_synthetic_queries(csvfile):
    print('---- Generating synthetic queries...')
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
        with open('training_data/output.csv', 'w') as file_w:
            writer = csv.writer(file_w)
            column_names = ['text']
            writer.writerow(column_names)

            for ix, text in enumerate(csv_reader):
                text = text[0]
                if(text=='text'): continue
                #if(ix == 6): break # for sampling TODO: remove in prod

                # generate question with LLM
                query = prompt_template.format(context_str=text, num_questions_per_chunk=2)
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
    SUPPORTED_INPUT_EXT = ['epub', 'pdf']
    extension = doc_filename.split('.')
    extension = extension[len(extension)-1]
    if(extension not in SUPPORTED_INPUT_EXT): raise Exception('Input file type not supported. Supported types are: ', SUPPORTED_INPUT_EXT)

    # saves to txt_from_epub.txt
    if extension == 'epub':
        extract_text_from_epub(doc_filename)
    if extension == 'pdf':
        extract_text_from_pdf(doc_filename)

    # saves to output.csv. You should filter out intros and other fluff here
    split_text_into_chunks('txt_from_epub.txt', 'output.csv')
    # saves to trainingdata/output.csv
    gen_synthetic_queries('output.csv')
    # saves to trainingdata/output_filtered.csv
    filter_by_blacklist('training_data/output.csv')

def contains_blacklisted_phrase(text, blacklist):
    # Create a regular expression pattern that matches any phrase in the blacklist
    pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, blacklist)) + r')\b', flags=re.IGNORECASE)
    match = pattern.search(text)
    return match is not None

def csv_to_bge_format(csv_file):
    #writes conversion to out2.jsonl (and adds html data too)
    jsonl_data = read_jsonl('bge_finetuning_data.jsonl')
    BLACKLIST = ['Registrant', 'Table of contents', 'Table of Contents', 'COVID', 'compliance', 'Exhibit', 'SEC', 'registration', 'bookkeeping', 'accounting', 'GAAP', 'financial reporting', 'exchange act'
                 , 'law', 'environmental' ,'climate change', 'equity and inclusion', 'harassment', '2021 to 2023'
                 ]
    with open('out.jsonl', 'w', newline='') as file:
        for obj in jsonl_data:
            pos = json.loads(obj['pos'][0])
            label = pos['label']
            data = pos['data']

            try:
                p_data = extract_text_with_tables(data, 'html.parser').get_text()
            except:
                p_data = data

            if contains_blacklisted_phrase(p_data+' '+obj['query']+' '+label, BLACKLIST):
                continue

            newObj = {}
            newObj['query'] = obj['query']
            newObj['pos'] = [{"label":label, "data":p_data}]
            newObj['neg'] = []
            file.writelines([json.dumps(newObj)+'\n'])

        with open(csv_file, 'r') as in_file:
            csv_reader = csv.reader(in_file)
            for jsonl_elem in csv_reader:
                jsonl_text = jsonl_elem[0]
                if(jsonl_text == 'text'): continue

                split_text = jsonl_text.split('bot: ')
                human = split_text[0]
                bot = split_text[1]

                obj = {}
                obj['query'] = human.replace('human', '')
                obj['pos'] = [{'data':bot}]

                file.writelines([json.dumps(obj) + '\n'])
    check_add_neg('out.jsonl')

def check_add_neg(jsonl_fname):
    jsonl_data = read_jsonl(jsonl_fname)
    new_data = []
    for obj in jsonl_data:
        newobj = {}
        newobj['query'] = obj['query']
        newobj['pos'] = [obj['pos'][0]['data']]
        newobj['neg'] = ['']

        new_data.append(newobj)

    with open('out2.jsonl', 'w', newline='') as file:
        for oo in new_data:
            file.writelines([json.dumps(oo)+'\n'])

csv_to_bge_format('training_data/dataset_1.csv')