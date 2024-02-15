#!/usr/bin/python3

#--------------------------------------------------------------#
# --------------- WORKING SMART MAKES THIS WORK ---------------#
#--------------------------------------------------------------#

from parse2 import parse_10k_filing
from bs4 import BeautifulSoup as bs
from ai import calc_embeddings
from db import (mv_insert_data, mv_search_and_query, pg_pullToEmbed, pg_update_chunks_nr,
                mv_pg_crosscheck_chunks,pg_reset_chunks_nr, pg_update_modified,mv_delete_filingid, pg_chunks_proc_inprogress)
import requests
import traceback
import time


#SINGLE_REPORT_MODE = False //TODO this
TIMER_SWITCH = True
API_SERVER = 'http://localhost:8000/'
def debug_print_processed_texts(processed_texts):
    with open('queryresults.txt', 'w') as file:
        for paragraph in processed_texts:
            file.write('\n\n ===================================================')
            file.write(paragraph)
    print('Printed parsed paragraphs in queryresults.txt!')

def getReport(cik):
  url = API_SERVER + 'lastreport/' + str(cik)
  res = requests.get(url, params={"full":True})
  return res.text

def cleanHTML(text):
    try:
        res = bs(text, 'html.parser').get_text()
    except Exception as e:
        res =''
        print('cleanHTML error', e)
    return res

##---------------------------------------------------------------------##
##########################----- EXECUTION -----##########################
##_____________________________________________________________________##
def start_pull():
    toEmbed = pg_pullToEmbed()

    def erase_rollback_insert(filing_id):
        pg_reset_chunks_nr(filing_id)
        mv_delete_filingid(filing_id)

    for reportRow in toEmbed:
        if TIMER_SWITCH: tic = time.perf_counter()

        filing_id = reportRow[0]
        cik = reportRow[2]
        fullreport = getReport(cik)
        filing_id_str = str(filing_id)

        #if filing_id != 4: continue
        pg_chunks_proc_inprogress(filing_id)
        print('Processing filingID '+filing_id_str)
        # data processing
        try:
            processed_texts = parse_10k_filing(fullreport)

            if False:   #TODO: debug mech
                debug_print_processed_texts(processed_texts)
                quit()

            processed_texts_no = len(processed_texts)
            if (processed_texts_no is None) or (processed_texts_no == 0) : raise ValueError('zero processed_texts_no')
        except Exception as e:
            print('Error on processing filingID: ' + str(filing_id), e)
            traceback.print_exc()
            continue

        print('---- Parsing OK ' + filing_id_str)
        filingIDs=[]
        embeddings=[]
        counters=[]

        for ix, paragraph in enumerate(processed_texts):
            embeddings.append( calc_embeddings(paragraph)[0] )
            counters.append(str(ix))
            filingIDs.append(filing_id)
        print('inserting filingID: ' + str(filing_id))

        # update chunk no to DB
        try:
            pg_update_chunks_nr(filing_id, processed_texts_no)
        except Exception as e:
            print('Error on processing filingID. -------- REVERTING --------: ' + str(filing_id), e)
            erase_rollback_insert(filing_id)
            continue

        #dd = mv_insert_data([counters, filingIDs, processed_texts[:4], embeddings])
        try:
            mv_insert_data([counters, filingIDs, processed_texts, embeddings])
        except Exception as e:
            print('Error inserting filingID ' + str(filing_id) + ' into vector DB.', e)
            erase_rollback_insert(filing_id)
            continue

        # crosscheck
        crosscheck = mv_pg_crosscheck_chunks(filing_id, processed_texts_no)
        if not crosscheck["ok"]:
            print("Chunk count crosscheck failed (mv/pg). -------- (NOT) REVERTING --------:", crosscheck)
            #erase_rollback_insert(filing_id)
            continue
        else: print('-Crosscheck OK')

        pg_update_modified(filing_id)
        if TIMER_SWITCH: toc = time.perf_counter()
        if TIMER_SWITCH: timing_sentence = f" in {toc - tic:0.4f}s"
        else: timing_sentence = 'UNTIMED'

        print(f"---PROCESSING SUCCESS {filing_id} {timing_sentence}")

start_pull()
#similarities = get_similarities('how much is the dividend?', 37)
#print(similarities)


