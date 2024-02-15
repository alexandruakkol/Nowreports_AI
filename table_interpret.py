from unstructured.partition.html import partition_html
from unstructured.staging.base import elements_to_json, convert_to_dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fuzzywuzzy import fuzz

model_name = "yolox"

################### ------------- PREPROCESSING PARAMS -------------###################
OVERLAP_PCT = 0.3

# paragraphs shorter than this limit get discarded (anti stubs)
TABLE_DISCARD_LIMIT_CHARS = 150
TEXT_DISCARD_LIMIT_CHARS = 60

WINDOWS = {
   0: {
       "CHUNKSIZE": 256, # max for split tables, chars
       "SPLITPOINT": 512 # min for concat text, chars
   },
   1: {
       "CHUNKSIZE": 1664,
       "SPLITPOINT": 1024
   }
}

def remove_similar(items, similarity_threshold=95):
    unique_items = []

    for item in items:
        if not any(fuzz.ratio(item, existing_item) > similarity_threshold for existing_item in unique_items):
            unique_items.append(item)

    return unique_items

def process_tables(htmltext):
    elements = partition_html(  # url="http://localhost:8000/lastreport/320193",
            text=htmltext,
            model_name = model_name,
            strategy = 'hi_res',
            infer_table_structure = True
    )

    data = convert_to_dict(elements)
    extracted_elements = []

    for window_number in WINDOWS:
        cache = ''
        current_title = ''
        for ix, entry in enumerate(data):
            entry_type = entry["type"]
            entry_text = entry["text"]

            if not ix == len(data):
                next_entry_type = ''
            else: next_entry_type = data[ix+1]["type"]

            ################ ----  text processing ---- ################
            # text before tables gets handles in table processing
            if entry_type != 'Table' and next_entry_type != "Table":

                if len(cache) > WINDOWS[window_number]["SPLITPOINT"]:
                    extracted_elements.append(current_title + ' ' + cache)
                    cache = ''

                if entry_type == 'Title':
                    current_title = '[' + entry_text + ']'
                    continue

                # if the next element is also text, try to cache it
                if next_entry_type == "NarrativeText" :
                    cache = cache + ' ' + entry_text
                    continue

                if len(entry_text) > TEXT_DISCARD_LIMIT_CHARS: extracted_elements.append(current_title + ' ' + entry_text)
                continue

            ################ ----  table processing ---- ################
            if entry_type == "Table":
                # split the tables by character
                chunksize = WINDOWS[window_number]["CHUNKSIZE"]
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunksize, chunk_overlap=OVERLAP_PCT * chunksize)
                docs = text_splitter.create_documents([entry["metadata"]["text_as_html"]])
                chunks = text_splitter.split_documents(docs)
                for chunk in chunks:
                    chunk_text = getattr(chunk, "page_content")
                    if len(chunk_text) <= TABLE_DISCARD_LIMIT_CHARS: continue
                    chunk = '[' + data[ix-1]["text"][0:200] + '] ' + chunk_text
                    extracted_elements.append(chunk)

    filtered_extracted_elements = remove_similar(extracted_elements) #2 sec per 1000

    return filtered_extracted_elements


