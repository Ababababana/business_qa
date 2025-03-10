import os

#######   这三个API KEY 换成你自己的 ######
os.environ['OPENAI_API_KEY']="sk-rmJifPa28392FEndYdyhyQDuZgWOFALTHqCfgiEIeuosgGkM"
os.environ['OPENAI_API_BASE']="https://chatapi.littlewheat.com/v1"
os.environ['TAVILY_API_KEY'] = "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"



from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from sentence_transformers import SentenceTransformer
from Raptor import setup_Raptor_retriever
from utils import predict_from_crag, load_example_docs, setup_text_splitter
from Raptor import setup_Raptor_retriever
from gen_multi_query import gen_multi_query
from rerank import reciprocal_rank_fusion
from retrieval import CRAG
from typing import List, TypedDict
from IPython.display import Image, display
from langchain.schema import Document
from langgraph.graph import START, END, StateGraph
from langchain_community.tools.tavily_search import TavilySearchResults
#from firecrawl import FirecrawlApp

#app = FirecrawlApp(api_key="fc-4951f0ac599740bea79b9fe986723503")



########### 搜索网页在这里加
# example url documents (some langchain website tutorial)
url_list = [
    #"https://python.langchain.com/docs/concepts/lcel/",
    #"https://python.langchain.com/docs/how_to/output_parser_structured/",
    #"https://python.langchain.com/docs/how_to/self_query/",
    "https://www.gelonghui.com/live/"
    ]

########### k是 网页搜索深度 
web_search_tool = TavilySearchResults(k=30)


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
model_name = "BAAI/bge-base-en-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embedding_model = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

text_splitter = setup_text_splitter(chunk_size=2000, chunk_overlap=200)

docs = load_example_docs(url_list)

retriever = setup_Raptor_retriever(docs,text_splitter,llm,embedding_model,embed_size=768)

corrective_retrieval = CRAG(llm)


class GraphState(TypedDict):
    
    question: str     #question
    generation: str      #LLM generation
    search: str     #whether to add search
    documents: List[str]     #list of documents
    steps: List[str]


def retrieve(state):

    #Retrieve documents
    #import pdb;pdb.set_trace()
    question = state["question"]
    documents = retriever.invoke(question)
    steps = state["steps"]
    steps.append("retrieve_documents")
    
    return {"documents": documents, "question": question, "steps": steps}


def generate(state):
    
    #Generate answer
    question = state["question"]
    documents = state["documents"]
    generation = corrective_retrieval.generator().invoke({"documents": documents, "question": question})
    steps = state["steps"]
    steps.append("generate_answer")
    
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "steps": steps,
    }


def grade_documents(state,relevant_doc_required):

    #Determines whether the retrieved documents are relevant to the question.
    #import pdb;pdb.set_trace()
    question = state["question"]
    documents = state["documents"]
    steps = state["steps"]
    steps.append("grade_document_retrieval")
    filtered_docs = []
    search = "No"
    relevant_doc_counter=0
    
    for d in documents:
        score = corrective_retrieval.retrieval_grader().invoke(
            {"question": question, "documents": d.page_content}
        )
        grade = score["score"]
        if grade == "yes":
            filtered_docs.append(d)
            relevant_doc_counter+=1
        
    if relevant_doc_counter <relevant_doc_required:
        search = "Yes"
    
    return {
        "documents": filtered_docs,
        "question": question,
        "search": search,
        "steps": steps,
    }


def web_search(state):
    
    #Web search based on the re-phrased question.
    question = state["question"]
    documents = state.get("documents", [])
    steps = state["steps"]
    steps.append("web_search")
    web_results = web_search_tool.invoke({"query": question})
    #import pdb;pdb.set_trace()
    documents.extend(
        [
            Document(page_content=d["content"], metadata={"url": d["url"]})
            for d in web_results
        ]
    )
    
    return {"documents": documents, "question": question, "steps": steps}


def decide_to_generate(state):
    
    #Determines whether to generate an answer, or re-generate a question.

    search = state["search"]
    if search == "Yes":
        print("\n\n Not enought infos in local database, searching from internet\n")
        return "search"
    else:
        print("\n\n Gnerate from relevant local datas")
        return "generate"
    

# Graph
workflow = StateGraph(GraphState)

# Define the nodes

workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node(
    "grade_documents",
    lambda state:
    grade_documents(state,relevant_doc_required=1)
    )  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("web_search", web_search)  # web search

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "search": "web_search",
        "generate": "generate",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)
#import pdb;pdb.set_trace()

custom_graph = workflow.compile()





###### 公司名字 在这里加
company_names=["领益智造","立讯精密","蓝思科技","精研科技","歌尔股份","瑞声科技","长盈精密","伯恩光学"]



for company_name in company_names:
    #example question(not specific and some mistakes)
    inputs= f"""搜集{company_name}这家公司在2025年2月以来的，新技术、新业务、新产品发布、新外部合作相关的新闻
    按照下列格式来分别对每一家公司详细输出答案，每个方面最少200字，最好能列出相关的数据，如没有在细分字段上抓取到信息则表述为略：

    {company_name}:

    （1）新技术：
    （2）新业务：
    （3）新产品：
    （4）新合作：

    """

    response = predict_from_crag(inputs,custom_graph)
    #import pdb;pdb.set_trace()
    res="\n\n"+response['response']


    ######  在这里设定要 让生成的结果写入 哪个 文件（建议txt文件）
    with open('./record_cp.txt',"a") as f:
        f.write(res)
    print(res)


