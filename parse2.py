#!/usr/bin/env python
import re
from bs4 import BeautifulSoup as bs, Tag
import os
from table_interpret import process_tables

WINDOWS = {
   0: {
       "CHUNKSIZE": 128,
       "SPLITPOINT": 256
   },
   1: {
       "CHUNKSIZE": 1536,
       "SPLITPOINT": 1024
   }
}


SOUND_FILEPATH = 'ding.mp3'


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

   def processElements(elem, typ):
       if not hasattr(elem ,'body'): return []
       processed_elements = process_tables(str(elem.body))

       return processed_elements

   processed_items = []

   print('=== starting to process ')
   bs_html = bs(document['10-K'], 'lxml')
   ee = processElements(bs_html.html, '1')
   print('=== processing done')
   processed_items.extend(ee)

   os.system(f"afplay {SOUND_FILEPATH}")

   # flatten
   processed_texts = processed_items

   return processed_texts



