#!/usr/bin/python3
import json

#--------------------------------------------------------------#
# --------------- WORKING SMART MAKES THIS WORK ---------------#
#--------------------------------------------------------------#

from parse2 import parse_10k_filing, semantic_string_split
from bs4 import BeautifulSoup as bs
from ai import calc_embeddings, label_earnings_message
from db import (mv_insert_data, mv_search_and_query, pg_pullToEmbed, pg_update_chunks_nr,
                mv_pg_crosscheck_chunks,pg_reset_chunks_nr, pg_update_modified,mv_delete_filingid, pg_chunks_proc_inprogress, pg_pullTranscripts,
                pg_pullTranscriptsByCik,
                pg_pullTranscriptMessages, pg_setCallInsertStatus)
from db import mv_reset_test_collection, mv_query_by_filingid, mv_delete_transcript_embeddings
import requests
import traceback
import time
import os

#SINGLE_REPORT_MODE = False //TODO this
TIMER_SWITCH = True
API_SERVER = 'https://nowreports.com/api/'
TEST_MODE = False # clears test collection and only pulls AAPL. for testing only!

def debug_print_processed_texts(processed_texts):
    with open('logs/queryresults.txt', 'w') as file:
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
        # make this not concatenate all text
        res = bs(text, 'html.parser')
        res = ' '.join(res.stripped_strings)

    except Exception as e:
        res =''
        print('cleanHTML error', e)
    return res

##---------------------------------------------------------------------##
##########################----- EXECUTION -----##########################
##_____________________________________________________________________##

def start_transcript_add_pipeline(cik, filing_id):

    if not (cik and filing_id):
        return print('No cik/filingID start_transcript_add_pipeline. Very unexpected. To be checked')

    data = pg_pullTranscriptsByCik(cik)
    callid = data[0][0]
    last_filing_id = data[0][2]


    if not callid: return
    print(f'Found new transcripts for {cik}')

    # first, delete all transcript embeddings from the current filingid (as precaution)
    mv_delete_transcript_embeddings(filing_id)

    transcript_messages = pg_pullTranscriptMessages(callid)
    transcripts_to_embed = []

    # get rid of operator
    transcript_messages = [message for message in transcript_messages if message[1] != 'Operator']

    print('transcript_messages filtered', len(transcript_messages))

    # process each message. classify [isIrrelevant, isCompany] by LLM and filter out
    for transcript_message in transcript_messages:
        text = transcript_message[0]
        agent = transcript_message[1]

        label_output = label_earnings_message(agent, text)
        try:
            label_json = json.loads(label_output)
            if label_json['isIrrelevant'] or not label_json['isCompany']:
                continue

            # split into semantic chunks because some replies are very long
            split_texts = semantic_string_split(text)

            max=0 # for length debugging reasons
            for text in split_texts:
                if len(text) > max:
                    max = len(text)

                message_obj = {"label": label_json['question_subject_summary'], "data": f"{agent}: {text}"}
                transcripts_to_embed.append(json.dumps(message_obj))

            #print('--- Max transcript chunk: ', max)

        except Exception as _:
            continue

    transcripts_embeddings = calc_embeddings(transcripts_to_embed)

    trues = []
    filingIDs = []
    for _ in transcripts_to_embed:
        trues.append(True)
        filingIDs.append(last_filing_id)

    #print('LENN', len(transcripts_to_embed))
    mv_insert_data([filingIDs, transcripts_to_embed, transcripts_embeddings["sparse"], transcripts_embeddings["dense"], trues])
    pg_setCallInsertStatus(callid, 'added')


def start_filing_proc_pipeline(toEmbed):
    def erase_rollback_insert(filing_id):
        pg_reset_chunks_nr(filing_id)
        mv_delete_filingid(filing_id)

    def process_row(reportRow):
        if TIMER_SWITCH: tic = time.perf_counter()

        filing_id = reportRow[0]
        cik = reportRow[2]
        fullreport = getReport(cik)
        filing_id_str = str(filing_id)

        # TEXT PROCESSING
        pg_chunks_proc_inprogress(filing_id)
        print('Processing filingID ' + filing_id_str)


        # processed_texts contain HTML tables (for source)
        processed_texts = parse_10k_filing(fullreport)

        if False:  # TODO: debug mech for source (HTML) texts
            debug_print_processed_texts(processed_texts)
            quit()

        processed_texts_no = len(processed_texts)
        if (processed_texts_no is None) or (processed_texts_no == 0): raise ValueError('zero processed_texts_no')

        # build matching texts (without HTML tags)
        matching_texts = [cleanHTML(obj) for obj in processed_texts]

        if False:  # TODO: debug mech for matching texts
            debug_print_processed_texts(matching_texts)
            quit()

        print('---- Parsing OK ' + filing_id_str)
        filingIDs = []
        falses = []  # list of 0s for isTranscript

        # CALC EMBEDDINGS PROCESSING (of MATCHING version - without HTML)
        embeddings = calc_embeddings(matching_texts)

        for _ in enumerate(processed_texts):
            filingIDs.append(filing_id)
            falses.append(False)

        # UPDATE CHUNKS IN DB
        try:
            if not TEST_MODE:
                pg_update_chunks_nr(filing_id, processed_texts_no)
        except Exception as e:
            if not TEST_MODE:
                erase_rollback_insert(filing_id)
            raise Exception('Error on processing filingID. -------- REVERTING --------: ' + str(filing_id), e)

        # MILVUS INSERT
        try:
            print('inserting filingID into vectorDB: ' + str(filing_id))
            # inserting SOURCE TEXTS WITH HTML into vectorDB
            mv_insert_data([filingIDs, processed_texts, embeddings["sparse"], embeddings["dense"], falses])
        except Exception as e:
            erase_rollback_insert(filing_id)
            raise Exception('Error inserting filingID ' + str(filing_id) + ' into vector DB.', e)


        # crosscheck
        # crosscheck = mv_pg_crosscheck_chunks(filing_id, processed_texts_no)
        # if not crosscheck["ok"]:
        #     print("Chunk count crosscheck failed (mv/pg). -------- (NOT) REVERTING --------:", crosscheck)
        #     # erase_rollback_insert(filing_id)
        #     continue
        # else:
        #     print('-Crosscheck OK')

        pg_update_modified(filing_id)

        # -----------------------  TRANSCRIPTS ADD TO EMBED  ------------------------

        start_transcript_add_pipeline(cik, filing_id)

        if TIMER_SWITCH: toc = time.perf_counter()
        if TIMER_SWITCH:
            timing_sentence = f" in {toc - tic:0.4f}s"
        else:
            timing_sentence = 'UNTIMED'

        print(f"---PROCESSING SUCCESS {filing_id} {timing_sentence}")

    # this is where the processing is called
    for reportRow in toEmbed:
        try:
            process_row(reportRow)
        except Exception as e:
            print('Error on processing. Skipped ' + str(reportRow[0]), e)
            traceback.print_exc() #print the stack trace
            continue


def start_regular_pull():
    toEmbed = pg_pullToEmbed()
    print(toEmbed)
    start_filing_proc_pipeline(toEmbed)

def start_test_pull():
    #start_transcript_add_pipeline('789019', 20)
    start_filing_proc_pipeline([[4654,'789019/000095017023035122/0000950170-23-035122.txt','789019']])

def update_embeddings_with_transcripts():

    #cycle through new transcripts
    transcripts_data = pg_pullTranscripts()

    for transcript in transcripts_data:
        last_filing_id = transcript[2]
        cik=transcript[3]
        start_transcript_add_pipeline(cik, last_filing_id)
        print('---- ADDED TRANSCRIPTS TO ' + last_filing_id)


# mv_query_by_filingid(4653, ["source", "isTranscript"])  # outputs to file all mv sources
# quit()

if TEST_MODE:
    print('TEST PULL MODE (just AAPL)')
    start_test_pull()
    #mv_delete_transcript_embeddings(4654)
    #mv_query_by_filingid(4653, ["source"])  # outputs to file all mv sources
    #update_embeddings_with_transcripts()
    #quit()
    #
    # start_transcript_add_pipeline('789019', 4654)
    #
    # mv_query_by_filingid(4654, ["source", "isTranscript"])  # outputs to file all mv sources
    #
    # mv_reset_test_collection()
    # mv_query_by_filingid(4654, ["isTranscript"])  # outputs to file all mv sources

    SOUND_FILEPATH = 'ding.mp3'
    os.system(f"afplay {SOUND_FILEPATH}")
else:

    #update_embeddings_with_transcripts()
    start_regular_pull()

#similarities = get_similarities('how much is the dividend?', 37)
#print(similarities)


