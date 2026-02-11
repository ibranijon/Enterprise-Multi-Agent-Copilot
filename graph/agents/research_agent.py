from typing import Any, Dict
from graph.state import GraphState
from retrival.doc_retriver import retriever



def retrieve_node(state: GraphState) -> Dict[str,Any]:
    print("---RETRIVE DATA FROM VECTORSTORE---")

    question = state["question"]
    documents = retriever.invoke(question)

    return {'question':question,'documents':documents}



