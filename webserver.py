from ai import calc_embeddings, qa
from db import mv_search_and_query, print_file, pg_get_injections
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
import nltk #todo: centralize nltk tasks with parse2
import re

# injection setup: what is needed for calculations
injections = pg_get_injections()
financial_terms = injections["financial_terms"]
formulas = injections["formulas"]
terms_texts = injections["embedding_text"]

def detect_financial_terms(sentence):
    sentence_lower = sentence.lower()
    terms_found = []

    for term, keywords in financial_terms.items():
        if any(re.search(r'\b' + re.escape(keyword) + r'\b', sentence_lower) for keyword in keywords):
            terms_found.append(term)

    return terms_found

def logAnwsering(question, context):
    with open('anws.log', 'w') as logfile:
        logfile.write('\n\n/////////////////////////////////////// Question: ' + question)
        for ix, el in enumerate(context):
            logfile.write('\n CC' + str(ix) + ' =============== ' + el)

def undupe_context_arr(context_arr):
    # filter out duplicate sentences (from the window method)
    res = []
    sentences_cache = []
    for hit_ix, hit in enumerate(context_arr):
        unduped_hit = ''
        sentence_separated_hit = nltk.tokenize.sent_tokenize(hit)
        for sentence in sentence_separated_hit:
            #if sentence has been processed before, discard
            if sentence not in sentences_cache:
                sentences_cache.append(sentence)
                unduped_hit = unduped_hit + ' ' + sentence
        if len(unduped_hit) != 0:
            res.append(unduped_hit)
    return res

def preQueryProc(question, filingID):
    finterms = detect_financial_terms(question)
    print('detected finterms: ', finterms)
    finterm_values = []
    for finterm in finterms:
        formula_elements = formulas[finterm].split(',')
        for formula_element in formula_elements:
            question = f'How much is the {formula_element}.  (look in tables)'
            if 'ask the user' in question:
                question = formula_element
            print('---formula ', question)
            finterm_value = get_similarities(question, filingID, 3)
            finterm_values.append(finterm_value)
    return [finterm_values, finterms]

def postQueryProc(postquery, finterms):
    for finterm in finterms:
        formula_text = terms_texts[finterm]
        postquery += f'. {formula_text} '
    print('postquery ', postquery)
    return postquery

def get_similarities(question, filingID, limit=13):
    q_embed = calc_embeddings(question)[0]
    hits = mv_search_and_query([q_embed], expr="filingID == " + str(filingID), limit=limit)[0]

    #print('HITS:',hits) #TODO: eliminate unsure anwsers by distance

    if True: # debug for distance optimization
        print('Distances ' + str(hits.distances))

    context_arr = [hit.entity.get('source') for hit in hits]
    print_file(context_arr)
    unduped_context_arr = undupe_context_arr(context_arr)
    logAnwsering(question, unduped_context_arr)
    print('\n------anws', ', '.join(unduped_context_arr))
    return ', '.join(unduped_context_arr)

def answer_question(messages, filingID):
    question = messages[-1]["content"]
    prequery_results = preQueryProc(question, filingID) # finds data for formula requirements
    finterm_values = prequery_results[0]
    finterms = prequery_results[1]

    if len(finterms) > 0:
        limit = 3
        finterm_values.append(' Use these figures to calculate the metric the user is asking for.')
    else: limit = 13

    context = get_similarities(question, filingID, limit)
    for finterm_value in finterm_values:
        context += f' {finterm_value} '
    context = postQueryProc(context, finterms)
    messages[-1]["content"] = messages[-1]["content"] + ' [CONTEXT]: ' + context
    for stream_msgs in qa(messages):
        if stream_msgs and len(stream_msgs) > 0:
            yield f"{stream_msgs}\n\n".encode('utf-8')


app = Flask(__name__)
app.debug = True

CORS(app, origins=["http://localhost:3000", "http://localhost:3001"])

@app.route("/completion", methods=["POST"])
def handle_completion():
    if not request.is_json: return jsonify({'error': 'Request does not contain JSON data'}), 400

    try:
        data = request.get_json()
        messages = json.loads(data.get('messages')) # this should also include prev convos
        filingID = data.get('filingID')
        # TODO: use opensource AI summarization model for appending prev convos #

        return Response(answer_question(messages, int(filingID)), mimetype='text/event-stream')
    except Exception as e:
        print(e)
        return jsonify({'error': 'Invalid JSON data'}), 400


