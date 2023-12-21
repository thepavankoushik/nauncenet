import together
import os
from rag_util import create_prompt_from_RAG
together.api_key = os.environ['TOGETHER_API_KEY']









def extract_classification_and_justification(input_string):
    # Split the input string into lines
    lines = input_string.strip().split('\n')

    # Extract classification (assuming it's the first line)
    classification = lines[0].strip()

    # Extract justification (assuming it's the rest of the lines)
    justification = "\n".join(lines[1:]).strip()
    return classification, justification


def llama2_70b(user_input):
  prompt = create_prompt_from_RAG(user_input)
  output = together.Complete.create(prompt, model="togethercomputer/llama-2-70b-chat",
                                    max_tokens = 100,
                                    temperature=0.7
                                    )
  response = output['output']['choices'][0]['text']
  classification, response = extract_classification_and_justification(response)
  return classification

def falcon_7b(user_input):
  prompt = create_prompt_from_RAG(user_input)
  output = together.Complete.create(prompt, model="togethercomputer/falcon-7b-instruct",
                                    max_tokens = 100,
                                    temperature=0.7
                                    )
  response = output['output']['choices'][0]['text']
  classification, response = extract_classification_and_justification(response)
  return classification

def mistral_7b(user_input):
  prompt = create_prompt_from_RAG(user_input)
  output = together.Complete.create(prompt, model="mistralai/Mistral-7B-Instruct-v0.1",
                                    max_tokens = 100,
                                    temperature=0.7
                                    )
  response = output['output']['choices'][0]['text']
  classification, response = extract_classification_and_justification(response)
  return classification

def llm_classify_tweet(model, user_input):
  response = model(user_input)
  return response