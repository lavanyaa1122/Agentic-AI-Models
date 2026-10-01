import os
from typing import TypedDict
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# PDF imports
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak
)
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch


# ============================================================
# 1. GROQ API KEY
# ============================================================

api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY is not set.\n"
        "Set it in Git Bash using:\n"
        'export GROQ_API_KEY="YOUR_API_KEY"'
    )


# ============================================================
# 2. INITIALIZE GROQ LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.3,
    api_key=api_key
)


# ============================================================
# 3. SHARED STATE
# ============================================================

class AgentState(TypedDict):
    question: str
    research: str
    technical: str
    report: str


# ============================================================
# 4. COORDINATOR AGENT
# ============================================================

def coordinator(state: AgentState):

    print("\n[Coordinator] Starting workflow...")

    return {
        "research": "",
        "technical": "",
        "report": ""
    }


# ============================================================
# 5. RESEARCH AGENT
# ============================================================

def research_agent(state: AgentState):

    print("\n[Research Agent] Working...")

    prompt = f"""
Analyze the following question as a Research Agent.

Identify:

- Important concepts
- Requirements
- Benefits
- Challenges
- Use cases

Question:

{state['question']}

Provide clear and factual research findings.
Do not invent unsupported facts.
"""

    response = llm.invoke([
        SystemMessage(
            content="You are an expert IT Research Agent."
        ),
        HumanMessage(
            content=prompt
        )
    ])

    return {
        "research": response.content
    }


# ============================================================
# 6. TECHNICAL AGENT
# ============================================================

def technical_agent(state: AgentState):

    print("\n[Technical Agent] Working...")

    prompt = f"""
You are a Kubernetes Technical Architect.

Based on the question and research below,
propose a technical solution.

Include:

1. Architecture
2. Kubernetes components
3. Deployment approach
4. Security
5. Monitoring
6. Implementation approach

User Question:

{state['question']}

Research Findings:

{state['research']}

Provide a practical and technically clear solution.
Do not invent unsupported facts.
"""

    response = llm.invoke([
        SystemMessage(
            content="You are an expert Kubernetes Technical Architect."
        ),
        HumanMessage(
            content=prompt
        )
    ])

    return {
        "technical": response.content
    }


# ============================================================
# 7. REPORT AGENT
# ============================================================

def report_agent(state: AgentState):

    print("\n[Report Agent] Working...")

    prompt = f"""
Prepare a professional technical report.

Use the information provided below.

The report must contain:

1. Executive Summary
2. Research Findings
3. Technical Architecture
4. Kubernetes Components
5. Deployment Approach
6. Security
7. Monitoring
8. Implementation Steps
9. Benefits
10. Challenges
11. Conclusion

User Question:

{state['question']}

Research Findings:

{state['research']}

Technical Solution:

{state['technical']}

Important:
Do not invent unsupported facts.
Use clear and professional language.
"""

    response = llm.invoke([
        SystemMessage(
            content="You are a professional Technical Report Writer."
        ),
        HumanMessage(
            content=prompt
        )
    ])

    return {
        "report": response.content
    }


# ============================================================
# 8. BUILD LANGGRAPH
# ============================================================

graph = StateGraph(AgentState)


# Register nodes
graph.add_node(
    "coordinator",
    coordinator
)

graph.add_node(
    "research",
    research_agent
)

graph.add_node(
    "technical",
    technical_agent
)

graph.add_node(
    "report",
    report_agent
)


# ============================================================
# 9. DEFINE WORKFLOW
# ============================================================

graph.add_edge(
    START,
    "coordinator"
)

graph.add_edge(
    "coordinator",
    "research"
)

graph.add_edge(
    "research",
    "technical"
)

graph.add_edge(
    "technical",
    "report"
)

graph.add_edge(
    "report",
    END
)


# ============================================================
# 10. COMPILE GRAPH
# ============================================================

app = graph.compile()


# ============================================================
# 11. PDF PAGE NUMBER
# ============================================================

def add_page_number(canvas, doc):

    canvas.saveState()

    page_number = canvas.getPageNumber()

    canvas.setFont(
        "Helvetica",
        9
    )

    canvas.drawCentredString(
        A4[0] / 2,
        20,
        f"Page {page_number}"
    )

    canvas.restoreState()


# ============================================================
# 12. CREATE PDF
# ============================================================

def create_pdf(question, report):

    # Create filename using date and time
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = f"kubernetes_report_{timestamp}.pdf"

    # Create PDF document
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=40
    )

    # Styles
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=28,
        alignment=TA_CENTER,
        spaceAfter=20
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=11,
        leading=16,
        alignment=TA_CENTER,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=15,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["BodyText"],
        fontSize=10,
        leading=15,
        spaceAfter=8
    )

    question_style = ParagraphStyle(
        "Question",
        parent=styles["BodyText"],
        fontSize=11,
        leading=16,
        spaceAfter=10
    )

    story = []


    # ========================================================
    # TITLE
    # ========================================================

    story.append(
        Paragraph(
            "KUBERNETES TECHNICAL REPORT",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Generated using LangGraph Multi-Agent AI",
            subtitle_style
        )
    )

    story.append(
        Spacer(1, 10)
    )


    # ========================================================
    # DATE
    # ========================================================

    current_date = datetime.now().strftime(
        "%d %B %Y, %I:%M %p"
    )

    story.append(
        Paragraph(
            f"<b>Generated:</b> {current_date}",
            body_style
        )
    )

    story.append(
        Spacer(1, 10)
    )


    # ========================================================
    # USER QUESTION
    # ========================================================

    story.append(
        Paragraph(
            "User Question",
            heading_style
        )
    )

    # Escape special HTML characters
    safe_question = (
        question
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    story.append(
        Paragraph(
            safe_question,
            question_style
        )
    )


    story.append(
        Spacer(1, 10)
    )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    story.append(
        Paragraph(
            "Final Report",
            heading_style
        )
    )


    # Split report into lines
    lines = report.split("\n")


    for line in lines:

        line = line.strip()

        # Empty line
        if not line:

            story.append(
                Spacer(1, 6)
            )

            continue


        # Escape HTML characters
        safe_line = (
            line
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )


        # Detect headings
        if (
            safe_line.startswith("1.")
            or safe_line.startswith("2.")
            or safe_line.startswith("3.")
            or safe_line.startswith("4.")
            or safe_line.startswith("5.")
            or safe_line.startswith("6.")
            or safe_line.startswith("7.")
            or safe_line.startswith("8.")
            or safe_line.startswith("9.")
            or safe_line.startswith("10.")
            or safe_line.startswith("11.")
            or safe_line.startswith("Executive Summary")
            or safe_line.startswith("Research Findings")
            or safe_line.startswith("Technical Architecture")
            or safe_line.startswith("Kubernetes Components")
            or safe_line.startswith("Deployment Approach")
            or safe_line.startswith("Security")
            or safe_line.startswith("Monitoring")
            or safe_line.startswith("Implementation Steps")
            or safe_line.startswith("Benefits")
            or safe_line.startswith("Challenges")
            or safe_line.startswith("Conclusion")
        ):

            story.append(
                Paragraph(
                    f"<b>{safe_line}</b>",
                    heading_style
                )
            )

        else:

            # Convert simple bullet points
            if safe_line.startswith("- "):

                safe_line = "• " + safe_line[2:]

            story.append(
                Paragraph(
                    safe_line,
                    body_style
                )
            )


    # ========================================================
    # BUILD PDF
    # ========================================================

    doc.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number
    )

    print(
        f"\nPDF created successfully!"
    )

    print(
        f"File: {filename}"
    )


# ============================================================
# 13. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("     LANGGRAPH MULTI-AGENT SYSTEM")
    print("========================================")

    question = input(
        "\nEnter your Kubernetes-related question: "
    ).strip()


    # Check empty question
    if not question:

        print(
            "\nPlease enter a question."
        )

    else:

        try:

            # Run LangGraph
            result = app.invoke({

                "question": question,

                "research": "",

                "technical": "",

                "report": ""
            })


            # Display final report
            print(
                "\n========== FINAL REPORT =========="
            )

            print(
                result["report"]
            )


            # Create PDF
            create_pdf(
                question,
                result["report"]
            )


        except Exception as e:

            print(
                "\nAn error occurred:"
            )

            print(
                str(e)
            )