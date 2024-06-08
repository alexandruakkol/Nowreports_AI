from ai import calc_embeddings, qa_mixtral, label_earnings_message, bedrock_qa
from db import mv_search_and_query, print_file, pg_get_injections, mv_check_filingID, mv_query_by_filingid
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
    with open('logs/query_results.log', 'a') as logfile:
        logfile.write('\n\n/////////////////////////////////////// Question: ' + question)
        for ix, el in enumerate(context):
            logfile.write('\n CC' + str(ix) + ' =============== ' + str(el))

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
    # look for financial terms to pre-fetch them (and calculate if necessary)
    finterms = detect_financial_terms(question)
    print('finterms:', finterms)
    finterm_values = []
    for finterm in finterms:
        formula_elements = formulas[finterm].split(',')
        for formula_element in formula_elements:
            question = f'How much is the {formula_element}?  (data is in tables)'
            if 'ask the user' in question:
                question = formula_element
            #print('---formula ', question)
            finterm_value = get_similarities(question, filingID, 3)
            finterm_values.append(finterm_value)
    return [finterm_values, finterms]

def postQueryProc(postquery, finterms):
    for finterm in finterms:
        formula_text = terms_texts[finterm]
        postquery += f'. {formula_text} '
    #print('postquery ', postquery)
    return postquery

def get_similarities(question, filingID, limit=13):
    query_embeddings = calc_embeddings([question])
    hits = mv_search_and_query(query_embeddings, expr="filingID == " + str(filingID), limit=limit)

    #if True: # debug for distance optimization
        #print('Distances ' + str(hits.distances))

    # DISTANCE FILTERING
    #this changes based on reranker k param
    DISTANCE_THRESHOLD = 0.1
    context_arr = [{"source":hit.fields["source"], "distance":hit.distance} for hit in hits[0] if hit.distance > DISTANCE_THRESHOLD]

    #unduped_context_arr = undupe_context_arr(context_arr)
    #unduped_context_arr = context_arr
    # reranking rewrites distance (top 2 is 0.1612...)
    logAnwsering(question, context_arr)

    hit_texts = [hit['source'] for hit in context_arr]

    #print('\n------anws', ', '.join(unduped_context_arr))
    return ', '.join(hit_texts)

def answer_question(messages, filingID, isHigherLimit=False):
    print('got question')
    question = messages[-1]["content"]
    prequery_results = preQueryProc(question, filingID) # finds data for formula requirements
    finterm_values = prequery_results[0]
    finterms = prequery_results[1]

    if len(finterms) > 0:
        limit = 7 # to account for the extra 3 finterm matches
        print('Found finterm')
        finterm_values.append(' Use these figures to calculate the metric the user is asking for.')
    else: limit = 7  #if too large, does not fit into context size

    if isHigherLimit: limit = 11 # this is set for the AI scan report questions

    context = get_similarities(question, filingID, limit)
    for finterm_value in finterm_values:
        context += f' {finterm_value} '
    context = postQueryProc(context, finterms)
    messages[-1]["content"] = '[QUESTION]: ' + messages[-1]["content"] + ' [CONTEXT]: ' + context
    for stream_msgs in bedrock_qa(messages):
        if stream_msgs and len(stream_msgs) > 0:
            if False:
                print(f"{stream_msgs}".encode('utf-8'))
            yield f"{stream_msgs}[ss]".encode('utf-8')


app = Flask(__name__)
app.debug = True

CORS(app, origins=["http://localhost:3000", "http://localhost:3001"])

@app.route("/completion", methods=["POST"])
def handle_completion():
    if not request.is_json:
        return jsonify({'error': 'Request does not contain JSON data'}), 400

    try:
        data = request.get_json()
        messages = json.loads(data.get('messages')) # this includes gross prev convos
        filingID = data.get('filingID')
        isAIReport = data.get('isAIReport')

        answer = answer_question(messages, int(filingID), isHigherLimit=isAIReport)

        return Response(answer, mimetype='text/event-stream')
    except Exception as e:
        print(e)
        return jsonify({'error': 'Invalid JSON data'}), 500

@app.route("/test", methods=["GET"]) # this endpoint tests this API and Milvus (not openAI)
def handle_apitest():
    try:
        is_mv_check_ok = type(mv_query_by_filingid(4657, ["sparse_vector"])[0]['pk']) is str
        if is_mv_check_ok:
            return '', 200
        else:
            return jsonify({'testing endpoint error': 'Vector DB check failed'}), 500
    except Exception as e:
        print('testing fault: ', e)
        return jsonify({'testing endpoint error': 'Vector DB error'}), 500

@app.route('/label_earnings_message', methods=["POST"])
def handle_label_earnings_message():
    try:
        if not request.is_json: return jsonify({'error': 'Request does not contain JSON data'}), 400

        data = request.get_json()
        text = data.get('text')
        agent = data.get('agent')

        label_res = label_earnings_message(agent, text)
        return label_res, 200

    except Exception as e:
        return jsonify({'error': e}), 400

