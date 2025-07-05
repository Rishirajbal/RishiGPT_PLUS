from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, add_messages, START, END


class RishiGPTPLUS(TypedDict):
    messages: Annotated[list, add_messages]
    model: str


def choose_model(state: RishiGPTPLUS) -> str:
    return "groq"

def model_groq(state: RishiGPTPLUS) -> RishiGPTPLUS:
    return {"messages": state["messages"] + [" Using GROQ"], "model": "groq"}

def model_cohere(state: RishiGPTPLUS) -> RishiGPTPLUS:
    return {"messages": state["messages"] + [" Using Cohere"], "model": "cohere"}

def model_openai(state: RishiGPTPLUS) -> RishiGPTPLUS:
    return {"messages": state["messages"] + [" Using OpenAI"], "model": "openai"}

def model_gemini(state: RishiGPTPLUS) -> RishiGPTPLUS:
    return {"messages": state["messages"] + [" Using Gemini"], "model": "gemini"}

def choose_task(state: RishiGPTPLUS) -> str:
    return "chatbot"

def run_chatbot(state: RishiGPTPLUS) -> RishiGPTPLUS:
    return {"messages": state["messages"] + [" Simple chatbot running"], "model": state["model"]}

def choose_rag_flavor(state: RishiGPTPLUS) -> str:
    return "rag_session"

def rag_in_session(state: RishiGPTPLUS) -> RishiGPTPLUS:
    return {"messages": state["messages"] + [" RAG: In-session"], "model": state["model"]}

def rag_in_pinecone(state: RishiGPTPLUS) -> RishiGPTPLUS:
    return {"messages": state["messages"] + [" RAG: Pinecone"], "model": state["model"]}

def rag_hybrid(state: RishiGPTPLUS) -> RishiGPTPLUS:
    return {"messages": state["messages"] + [" RAG: Hybrid"], "model": state["model"]}

def web_search(state: RishiGPTPLUS) -> RishiGPTPLUS:
    return {"messages": state["messages"] + [" Web search"], "model": state["model"]}

def decide_next(state: RishiGPTPLUS) -> str:
    return "END"


build3 = StateGraph(RishiGPTPLUS)

build3.add_node("choose_model", choose_model)
build3.add_node("groq", model_groq)
build3.add_node("cohere", model_cohere)
build3.add_node("openai", model_openai)
build3.add_node("gemini", model_gemini)

build3.add_node("choose_task", choose_task)
build3.add_node("chatbot", run_chatbot)
build3.add_node("choose_rag_flavor", choose_rag_flavor)
build3.add_node("rag_session", rag_in_session)
build3.add_node("rag_pinecone", rag_in_pinecone)
build3.add_node("rag_hybrid", rag_hybrid)
build3.add_node("web_search", web_search)
build3.add_node("decide_next", decide_next)


build3.add_edge(START, "choose_model")

build3.add_conditional_edges(
    "choose_model",
    choose_model,
    {
        "groq": "groq",
        "cohere": "cohere",
        "openai": "openai",
        "gemini": "gemini"
    }
)


build3.add_edge("groq", "choose_task")
build3.add_edge("cohere", "choose_task")
build3.add_edge("openai", "choose_task")
build3.add_edge("gemini", "choose_task")

build3.add_conditional_edges(
    "choose_task",
    choose_task,
    {
        "chatbot": "chatbot",
        "rag": "choose_rag_flavor",
        "web": "web_search"
    }
)

build3.add_conditional_edges(
    "choose_rag_flavor",
    choose_rag_flavor,
    {
        "rag_session": "rag_session",
        "rag_pinecone": "rag_pinecone",
        "rag_hybrid": "rag_hybrid"
    }
)

build3.add_edge("chatbot", "decide_next")
build3.add_edge("web_search", "decide_next")
build3.add_edge("rag_session", "decide_next")
build3.add_edge("rag_pinecone", "decide_next")
build3.add_edge("rag_hybrid", "decide_next")

build3.add_conditional_edges(
    "decide_next",
    decide_next,
    {
        "chatbot": "chatbot",
        "rag": "choose_rag_flavor",
        "web": "web_search",
        "END": END
    }
)

graph3 = build3.compile()


from IPython.display import Image, display

try:
    display(Image(graph3.get_graph().draw_mermaid_png()))
except Exception as e:
    print("Could not display the graph:", e)
