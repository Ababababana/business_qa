from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults



#web_search_tool = TavilySearchResults()


class CRAG():
    def __init__(self,llm):
        self.llm = llm

    def retrieval_grader(self,):
        prompt = PromptTemplate(
            template="""You are a teacher grading a quiz. You will be given: 
            1/ a QUESTION
            2/ A FACT provided by the student
            
            You are grading RELEVANCE RECALL:
            A score of 1 means that ANY of the statements in the FACT are relevant to the QUESTION. 
            A score of 0 means that NONE of the statements in the FACT are relevant to the QUESTION. 
            1 is the highest (best) score. 0 is the lowest score you can give. 
            
            Explain your reasoning in a step-by-step manner. Ensure your reasoning and conclusion are correct. 
            
            Avoid simply stating the correct answer at the outset.
            
            Question: {question} \n
            Fact: \n\n {documents} \n\n
            
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        
            Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
            """,
            input_variables=["question", "documents"],
        )

        retrieval_grader = prompt | self.llm | JsonOutputParser()
        return retrieval_grader


    def generator(self,):
        prompt = PromptTemplate(
            template="""You are an assistant for question-answering tasks. \n
            
            Use the following documents to answer the question. \n
            
            If you don't know the answer, just say that you don't know. \n
            
            Use three sentences maximum and keep the answer concise: \n
            Question: {question} 
            Documents: {documents} 
            Answer: 
            """,
            input_variables=["question", "documents"],
        )

        # Chain
        rag_chain = prompt | self.llm | StrOutputParser()
        return rag_chain
    
    def load_web_tool(self,k=3):
        #import pdb;pdb.set_trace()
        web_search_tool = TavilySearchResults(k)
        return web_search_tool
