from langsmith import Client
from langsmith.schemas import Run, Example
from langsmith.evaluation import evaluate
import openai
from langsmith.wrappers import wrap_openai
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
import os
from langchain_community.llms import AI21

LANGCHAIN_API_KEY="ls__7d989adf26cb48409e9f9ec86b929110"

langclient = Client(api_key=LANGCHAIN_API_KEY)

# Define dataset: these are your test cases
dataset_name = "Assorted AI Dataset25"
dataset = langclient.create_dataset(dataset_name, description="Assorted personal ai prompts3")
langclient.create_examples(
    inputs=[
        {"question": "There are two paths with two men, one path leading to life and the other to death. One man will always lie while the other will always tell the truth. You may only ask one question total to figure out which path will lead to life. What is the question that leads to the answer?"},
        {"question": "what is the 152th element called? please provide a reference"},
        {"question": "where can i buy the incredibles 1 lego set"},
        {"question": "does curious george have a tail"},
        {"question": "If five cats can catch five mice in five minutes, how long will it take one cat to catch one mouse?"}
    ],
    outputs=[
        {"must_mention": ["other person"]},
        {"must_mention": ["Ununennium"]},
        {"must_mention": ["website", "store", "can find"]},
        {"must_mention": ["no"]},
        {"must_mention": ["one minute", "one mouse"]}
    ],
    dataset_id=dataset.id,
)

"""# Define AI system
openai_client = wrap_openai(HuggingFaceHub)"""

HUGGING_FACE_API_TOKEN = "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"
AI21_API_KEY = "LufKXBwrXWaFzo4JBXuvmJvsvSoqQmHk"

flan_t5_model = HuggingFaceEndpoint(
    repo_id="google/flan-t5-xxl",
    temperature=1e-1,
    huggingfacehub_api_token=HUGGING_FACE_API_TOKEN,
    model_kwargs={}
)

AI21_LLM = AI21(ai21_api_key=AI21_API_KEY)

def predict_openai(inputs: dict) -> dict:
    messages = [{"role": "user", "content": inputs["question"]}]
    response = openai_client.chat.completions.create(messages=messages, model="gpt-3.5-turbo")
    return {"output": response}

def predict_hugging_face(inputs: dict) -> dict:
    generate = flan_t5_model.generate([inputs["question"]])
    print(generate)

# Define evaluators
def must_mention(run: Run, example: Example) -> dict:
    prediction = run.outputs.get("output") or ""
    required = example.outputs.get("must_mention") or []
    score = all(phrase in prediction for phrase in required)
    return {"key":"must_mention", "score": score}

keyClient = Client(api_key=LANGCHAIN_API_KEY)

experiment_results = evaluate(
    predict_hugging_face, # Your AI system
    data=dataset_name, # The data to predict and grade over
    evaluators=[must_mention], # The evaluators to score the results
    experiment_prefix="assorted-huggingface-generator", # A prefix for your experiment names to easily identify them
    metadata={
      "version": "1.0.0",
    },
    client=keyClient
)