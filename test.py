# importing OpenAI Wrapper from LangChain
#from langchain_community.llms import OpenAI
from langchain_openai import OpenAI
import os

# provide your API KEY here
os.environ["OPENAI_API_KEY"] = "sk-proj-SOS13R5DUeUcYisVfexLT3BlbkFJpv6BmpYOLZJsRCDipYzx"
# initializing OpenAI LLM
llm = OpenAI(model_name="gpt-3.5-turbo")

# query
query = 'Tell me a joke'

# model output
print(llm(query))