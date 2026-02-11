from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

from graph.state import GraphState
from graph.consts import PLAN, RESEARCH, VERIFY, WRITE
from graph.nodes import planner_node, research_node, verifier_node, writer_node

load_dotenv()

# Workflow: User Query -> Planner -> Research -> Draft(Writer) -> Verify -> End
workflow = StateGraph(GraphState)

# Nodes
workflow.add_node(PLAN, planner_node)
workflow.add_node(RESEARCH, research_node)
workflow.add_node(WRITE, writer_node)
workflow.add_node(VERIFY, verifier_node)

# Edges
workflow.set_entry_point(PLAN)
workflow.add_edge(PLAN, RESEARCH)
workflow.add_edge(RESEARCH, WRITE)
workflow.add_edge(WRITE, VERIFY)
workflow.add_edge(VERIFY, END)

# Compile
app = workflow.compile()
