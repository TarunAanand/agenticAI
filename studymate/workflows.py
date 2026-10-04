from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from studymate.groq_client import (
    AnswerCritique,
    ChatMessage,
    critique_grounded_answer,
    generate_quiz,
    get_chat_reply,
    plan_retrieval_queries,
)
from studymate.materials import RetrievedChunk, TextChunk, retrieve_chunks
from studymate.quiz import QuizQuestion
from studymate.storage import FocusTopic, SubjectProgress


class AdvisorWorkflowState(TypedDict, total=False):
    question: str
    history: list[ChatMessage]
    chunks: list[TextChunk]
    student_context: str
    agentic: bool
    model: str
    search_queries: list[str]
    retrieved: list[RetrievedChunk]
    reply: str
    critique_verdict: str
    error_stage: str


class QuizWorkflowState(TypedDict, total=False):
    topic: str
    question_count: int
    chunks: list[TextChunk]
    model: str
    questions: list[QuizQuestion]
    retrieved: list[RetrievedChunk]
    error_stage: str


def _plan_advisor_searches(state: AdvisorWorkflowState) -> dict:
    queries = [state["question"]]
    if state.get("agentic") and state.get("chunks"):
        queries.extend(plan_retrieval_queries(state["question"], state.get("model")))
    return {"search_queries": queries}


def _retrieve_advisor_materials(state: AdvisorWorkflowState) -> dict:
    ranked_by_key: dict[tuple[str, str], RetrievedChunk] = {}
    for query in state.get("search_queries", [state["question"]]):
        for result in retrieve_chunks(query, state.get("chunks", [])):
            key = (result["source"], result["content"])
            previous = ranked_by_key.get(key)
            if previous is None or result["score"] > previous["score"]:
                ranked_by_key[key] = result
    retrieved = sorted(
        ranked_by_key.values(),
        key=lambda result: result["score"],
        reverse=True,
    )[:4]
    return {"retrieved": retrieved}


def _draft_advisor_answer(state: AdvisorWorkflowState) -> dict:
    return {
        "reply": get_chat_reply(
            state["history"],
            state.get("retrieved", []),
            student_context=state.get("student_context", ""),
            model=state.get("model"),
        )
    }


def _critic_advisor_answer(state: AdvisorWorkflowState) -> dict:
    critique: AnswerCritique = critique_grounded_answer(
        state["question"],
        state["reply"],
        state["retrieved"],
        state.get("model"),
    )
    return {
        "reply": critique["answer"],
        "critique_verdict": critique["verdict"],
    }


def _route_to_critic(state: AdvisorWorkflowState) -> str:
    if state.get("agentic") and state.get("retrieved"):
        return "critic"
    return END


def _retrieve_quiz_materials(state: QuizWorkflowState) -> dict:
    chunks = state.get("chunks", [])
    retrieved = retrieve_chunks(state["topic"], chunks)
    return {"retrieved": retrieved}


def _generate_quiz(state: QuizWorkflowState) -> dict:
    retrieved = state.get("retrieved", [])
    questions = generate_quiz(
        state["topic"],
        state["question_count"],
        retrieved,
        model=state.get("model"),
    )
    return {"questions": questions}


def _build_advisor_graph():
    graph = StateGraph(AdvisorWorkflowState)
    graph.add_node("plan_searches", _plan_advisor_searches)
    graph.add_node("retrieve_materials", _retrieve_advisor_materials)
    graph.add_node("draft_answer", _draft_advisor_answer)
    graph.add_node("critique_answer", _critic_advisor_answer)
    graph.add_edge(START, "plan_searches")
    graph.add_edge("plan_searches", "retrieve_materials")
    graph.add_edge("retrieve_materials", "draft_answer")
    graph.add_conditional_edges(
        "draft_answer",
        _route_to_critic,
        {"critic": "critique_answer", END: END},
    )
    graph.add_edge("critique_answer", END)
    return graph.compile()


def _build_quiz_graph():
    graph = StateGraph(QuizWorkflowState)
    graph.add_node("retrieve_materials", _retrieve_quiz_materials)
    graph.add_node("generate_quiz", _generate_quiz)
    graph.add_edge(START, "retrieve_materials")
    graph.add_edge("retrieve_materials", "generate_quiz")
    graph.add_edge("generate_quiz", END)
    return graph.compile()


advisor_workflow = _build_advisor_graph()
quiz_workflow = _build_quiz_graph()


def run_advisor_workflow(
    question: str,
    history: list[ChatMessage],
    chunks: list[TextChunk],
    student_context: str,
    agentic: bool,
    model: str | None = None,
) -> AdvisorWorkflowState:
    """Run retrieval, chat generation, and an optional answer critique."""
    return advisor_workflow.invoke(
        {
            "question": question,
            "history": history,
            "chunks": chunks,
            "student_context": student_context,
            "agentic": agentic,
            **({"model": model} if model else {}),
        }
    )


def run_quiz_workflow(
    topic: str,
    question_count: int,
    chunks: list[TextChunk],
    model: str | None = None,
) -> QuizWorkflowState:
    """Run local source retrieval then quiz generation as a graph."""
    return quiz_workflow.invoke(
        {
            "topic": topic,
            "question_count": question_count,
            "chunks": chunks,
            **({"model": model} if model else {}),
        }
    )
