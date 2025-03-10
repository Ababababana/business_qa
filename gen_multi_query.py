from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def gen_multi_query(llm):
    template = """You are an AI language model assistant. 

    Your task is to generate five different versions of the given user question \n
    to retrieve relevant documents from a vector database.

    By generating multiple perspectives on the user question, \n
    your goal is to help the user overcome some of the limitations of the distance-based similarity search. 

    Provide these alternative questions separated by newlines. Original question: 

    {question}"""
    prompt_perspectives = ChatPromptTemplate.from_template(template)

    generate_queries = (
        prompt_perspectives 
        | llm 
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )
    return generate_queries