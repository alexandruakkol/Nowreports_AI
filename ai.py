from dotenv import load_dotenv #pip install python-dotenv
from openai import OpenAI
from transformers import GPT2Tokenizer, AutoModel, AutoTokenizer
from db import print_file
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import os
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from llama_index.embeddings.openai import OpenAIEmbedding

api_key = os.environ["MISTRAL_API_KEY"]
mistral_model = "open-mixtral-8x22b"
mistral_client = MistralClient(api_key=api_key)

SYSTEM_PROMPT = {"role": "system",
                 "content": 'You are an AI tool called NowReports that has information about a business, provided in context. answers user questions accurately, based on data from a financial report. The user is a potential investor in the company, so he would want to know the important information, both good and bad, before buying it. Structure your responses into brief, easy to understand points whenever you can. Do not give long answers if not asked to. Never generate tags like [CONTEXT] or [AI] into your response. If asked for a financial metric, or to calculate something, use chain of thought: first find the formula, secondly look into the report for all the necessary data, and then perform the calculation yourself, until you get to the result. Pay attention so that all your calculations are correct and make sense. '''}
system_prompt_file = open('system_prompt.txt', 'r')
SYSTEM_PROMPT = {"role": "system", "content": system_prompt_file.read().replace('\n', '')}
print(SYSTEM_PROMPT)

openai_embed_model = OpenAIEmbedding()


def qa_mixtral(json_messages):
    messages = [SYSTEM_PROMPT]
    for message in json_messages:
        messages.append(ChatMessage(role=message["role"], content=message["content"]))

    if True: # actual prompt logging
        for message in messages:
            print_file(message, 'actual_prompt.txt', 'a')

    chat_response = mistral_client.chat_stream(
        model=mistral_model,
        messages=messages,
        temperature=0.25,
        max_tokens=700
    )

    for message in chat_response:
        # finish reason debug
        if True and message.choices[0].finish_reason is not None:
            print(message.choices[0].finish_reason)
        yield message.choices[0].delta.content

    #print(chat_response.choices[0].message.content)


# model = INSTRUCTOR('hkunlp/instructor-large')
# instruction = "Represent the financial report section for retrieving supporting sections: "
tokenizer = AutoTokenizer.from_pretrained('alexakkol/BAAI-bge-base-en-nowr-1-2')

#for GPT tokenization
openai_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

load_dotenv()
llm_client = OpenAI()

# model = SentenceTransformer('alexakkol/BAAI-bge-base-en-nowr-1-2')

# thenlper/gte-large is the best of all these
# nickmuchi/setfit-finetuned-financial-text-classification meh, didnt work with apple. is also old
# sentencetransformer all-mpnet-base-v2 is wrong, supposed to be the best of sentencetransformers..
# sentencetransformer all-MiniLM-L6-v2 still wrong..
# M2 bert 80M no RAM

ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu", model_name='BAAI/bge-m3')

def calc_embeddings(data):
    return ef(data)
    # return model.encode([data], normalize_embeddings=True)

def qa(messages):
    messages.insert(0, SYSTEM_PROMPT)

    if True: # actual prompt logging
        for message in messages:
            print_file(message, 'actual_prompt.txt', 'a')

    stream = llm_client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        max_tokens=300,
        stream=True,
        temperature=0.3
        # frequency_penalty=0.5,
        # presence_penalty=0.5
    )
    for message in stream:
        yield message.choices[0].delta.content

    #return completion.
def openai_count_tokens(text: str) -> int:
    return len(openai_tokenizer.tokenize(text))

def label_earnings_message(agent, message):
    LABEL_EARNINGS_SYSPROMPT = ('The message below is from an earnings call. Your task is to output a JSON with the properties: isCompany (true or false), question_subject_summary and isIrrelevant (true or false): '
                             '\n isCompany = true if the message is coming from the company (usually not a question)'
                             '\n isIrrelevant = true if the message is just an introduction, smalltalk, or anything that\'s irrelevant to business analysis'
                             '\n question_subject_summary = a summary of the subject of the current question.'
    )

    messages = [
        ChatMessage(role="system", content=LABEL_EARNINGS_SYSPROMPT),
        ChatMessage(role="user", content=f"{agent}: ${message}")
    ]

    if False:  # actual prompt logging
        for message in messages:
            print_file(message, 'actual_prompt.txt', 'a')

    chat_response = llm_client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        response_format={"type": "json_object"},
    )

    return chat_response.choices[0].message.content
