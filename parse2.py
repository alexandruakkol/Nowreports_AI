#!/usr/bin/env python
import re
from bs4 import BeautifulSoup as bs, Tag
from unstructured.partition.html import partition_html
from unstructured.staging.base import elements_to_json, convert_to_dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fuzzywuzzy import fuzz
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import SimpleDirectoryReader, Document

from ai import openai_embed_model

################### ------------- PREPROCESSING PARAMS -------------###################
TABLE_OVERLAP_PCT = 0.3
tableparse_model_name = "yolox"

TEXT_LIMIT_CHARS = 100

# paragraphs shorter than this limit get discarded (anti stubs)
TABLE_DISCARD_LIMIT_CHARS = 150

WINDOWS = {
   0: {
       "CHUNKSIZE": 1100,
       "SPLITPOINT": 800
   },
   1: {
       "CHUNKSIZE": 2300,
       "SPLITPOINT": 2100
   }
}

TABLE_WINDOWS = {
   0: {
       "CHUNKSIZE": 2500, # max for split tables, chars
       "SPLITPOINT": 2000 # min for concat text, chars
   },
   1: {
       "CHUNKSIZE": 5000,
       "SPLITPOINT": 4000
   }
}

def parse_10k_filing(raw_10k):
    doc_start_pattern = re.compile(r'<DOCUMENT>')
    doc_end_pattern = re.compile(r'</DOCUMENT>')
    type_pattern = re.compile(r'<TYPE>[^\n]+')

    doc_start_is = [x.end() for x in doc_start_pattern.finditer(raw_10k)]
    doc_end_is = [x.start() for x in doc_end_pattern.finditer(raw_10k)]

    doc_types = [x[len('<TYPE>'):] for x in type_pattern.findall(raw_10k)]
    document = {}

    for doc_type, doc_start, doc_end in zip(doc_types, doc_start_is, doc_end_is):
        if doc_type == '10-K':
           document[doc_type] = raw_10k[doc_start:doc_end]

    def remove_attributes(soup_element):
       attr_blacklist = ['style','id','class','contextref','decimals','format','name','scale']
       if isinstance(soup_element, Tag):
           soup_element.attrs = {key: value for key, value in soup_element.attrs.items() if key not in attr_blacklist}
           for child in soup_element.contents: # recursively go through children
               remove_attributes(child)
       return soup_element

    def processElements(elem):
       if not hasattr(elem ,'body'): return []
       processed_elements = process_tables(str(elem.body))

       return processed_elements

    processed_items = []

    bs_html = bs(document['10-K'], 'lxml')
    ee = processElements(bs_html.html)

    processed_items.extend(ee)

    # flatten
    processed_texts = processed_items

    return processed_texts

####################################### TABLE PARSING ###################################################


def remove_similar(items, similarity_threshold=98):
    unique_items = []

    for item in items:
        if not any(fuzz.ratio(item, existing_item) > similarity_threshold for existing_item in unique_items):
            unique_items.append(item)

    return unique_items

def make_embed_json_string(label="", data=""):
        return '{"label":"' + label + '","data":"' + data + '"}'

def process_tables(htmltext):
    elements = partition_html(  # url="http://localhost:8000/lastreport/320193",
            text=htmltext,
            model_name = tableparse_model_name,
            strategy = 'hi_res',
            infer_table_structure = True
    )
    #
    # with open('logs/queryresults.txt', 'w') as file:
    #     for element in elements:
    #         file.write('\n\n')
    #         file.write(str(element))
    #
    # quit()

    data = convert_to_dict(elements)
    extracted_elements = []

    #chunking split into two windows
    for window_number in TABLE_WINDOWS:
        cache = ''
        current_title = ''
        for ix, entry in enumerate(data):
            entry_type = entry["type"]
            entry_text = entry["text"]
            if not ix == len(data):
                next_entry_type = ''
            else: next_entry_type = data[ix+1]["type"]

            ################ ----  text in table processing ---- ################
            # text before tables gets handled in table processing
            if entry_type != 'Table' and next_entry_type != "Table":

                if len(cache) > WINDOWS[window_number]["SPLITPOINT"]:
                    json_string = make_embed_json_string(current_title, cache)
                    extracted_elements.append(json_string)
                    cache = ''

                if entry_type == 'Title':
                    current_title = entry_text

                # if the next element is also text, try to cache it
                if next_entry_type in ["NarrativeText", "Title", "UncategorizedText"]:
                    cache = cache + ' ' + entry_text
                    continue

                if (len(cache) + len(entry_text)) > TEXT_LIMIT_CHARS:
                    cache = cache + ' ' + entry_text
                    # json_string = make_embed_json_string(current_title, cache + ' ' + entry_text)
                    # extracted_elements.append(json_string)
                    continue


            ################ ----  table processing ---- ################
            if entry_type == "Table":
                # split the tables by character
                chunksize = WINDOWS[window_number]["CHUNKSIZE"]
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunksize, chunk_overlap=TABLE_OVERLAP_PCT * chunksize)
                docs = text_splitter.create_documents([entry["metadata"]["text_as_html"]])
                chunks = text_splitter.split_documents(docs)
                for chunk in chunks:
                    chunk_text = getattr(chunk, "page_content")
                    if len(chunk_text) <= TABLE_DISCARD_LIMIT_CHARS: continue
                    # description limited to x chars
                    json_string = make_embed_json_string(data[ix-1]["text"][0:300], chunk_text)
                    chunk = json_string
                    extracted_elements.append(chunk)

    filtered_extracted_elements = remove_similar(extracted_elements) #takes 2 sec per 1000

    return filtered_extracted_elements

def semantic_string_split(in_str):
    documents = [Document(text=in_str)]
    # decrease breakpoint as much as possible without getting single outlier sentences
    splitter = SemanticSplitterNodeParser(
        buffer_size=4, breakpoint_percentile_threshold=90, embed_model=openai_embed_model
    )

    chunks = splitter.get_nodes_from_documents(documents)
    texts = [chunk.get_content() for chunk in chunks]
    filtered_by_size = [text for text in texts if len(text) > 100]
    return filtered_by_size


def cleanHTML(obj):
    return obj