from dotenv import load_dotenv #pip install python-dotenv
from openai import OpenAI
from InstructorEmbedding import INSTRUCTOR
from transformers import GPT2Tokenizer, AutoModel, AutoTokenizer
from db import print_file

model = INSTRUCTOR('hkunlp/instructor-large')
instruction = "Represent the financial report section for retrieving supporting sections: "
tokenizer = AutoTokenizer.from_pretrained('hkunlp/instructor-large')

#for GPT tokenization
openai_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

load_dotenv()
llm_client = OpenAI()
#model = SentenceTransformer('thenlper/gte-large')

# thenlper/gte-large is the best of all these
# nickmuchi/setfit-finetuned-financial-text-classification meh, didnt work with apple. is also old
# sentencetransformer all-mpnet-base-v2 is wrong, supposed to be the best of sentencetransformers..
# sentencetransformer all-MiniLM-L6-v2 still wrong..
# M2 bert 80M no RAM

def calc_embeddings(data):
    return model.encode([data, instruction], convert_to_tensor=False)

def qa(messages):
    SYSTEM_PROMPT = {"role": "system", "content": 'You are an AI tool called NowReports that briefly answers user questions based on a financial report. The user is a potential investor in the company. You are given the context from that document. Briefly summarize and explain the info that you get. The user is referring to the domain of finance and business, and to the financial document that is currently given in your context. Do not give long answers if not asked to. Never generate tags like [CONTEXT] or [AI] into your response. If asked to calculate something, look into the report for the necessary data and always do the calculation yourself. '''}
    messages.insert(0, SYSTEM_PROMPT)

    if True: # actual prompt logging
        for message in messages:
            print_file(message, 'actual_prompt.txt', 'a')

    stream = llm_client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        max_tokens=300,
        stream=True,
        frequency_penalty=0.5,
        presence_penalty=0.5
    )
    for message in stream:
        yield message.choices[0].delta.content

    #return completion.
def openai_count_tokens(text: str) -> int:
    return len(openai_tokenizer.tokenize(text))

