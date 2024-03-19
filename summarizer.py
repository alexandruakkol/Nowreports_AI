# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from db import pg_write_convo_summary
import json

tokenizer = AutoTokenizer.from_pretrained("kabita-choudhary/finetuned-bart-for-conversation-summary")
model = AutoModelForSeq2SeqLM.from_pretrained("kabita-choudhary/finetuned-bart-for-conversation-summary")

def stringify_agent_json(_json):
    _json = _json.replace("'",'"')
    result = ''

    #check if json
    try:
        _json =  json.loads(_json.replace("'",'"'))
    except:
        return _json

    for message in _json:
        result = result + ' ' + message['role'] + ': ' + message['content']
    return result

def summarize(_json):
    #structure of JSON input is: [{'role': 'user', 'content': '{[TEXT]}'},[{}],...]
    agent_string = stringify_agent_json(_json)

    print("SUMMARIZE input: ", agent_string)

    inputs = tokenizer([agent_string], return_tensors="pt")

    summary_ids = model.generate(inputs["input_ids"], num_beams=4)
    res = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # inputs = tokenizer(text, return_tensors="pt").input_ids
    # outputs = model.generate(inputs)
    # res = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print('SUMMARIZER OUTPUT:', res)
    return res

def summarize_and_writedb(raw_convo, convo_id):
    summary = summarize(raw_convo)
    pg_write_convo_summary(convo_id, summary)