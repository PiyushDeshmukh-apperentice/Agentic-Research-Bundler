from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
import os
import json

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# Pydantic Schemas (FORCED THINKING)
# --------------------------------------------------

class Analysis(BaseModel):
    research_domain: str = Field(
        ..., description="High-level research domain (e.g., Remote Sensing, NLP)"
    )
    sub_domain: str = Field(
        ..., description="Specific research sub-domain"
    )
    problem_type: str = Field(
        ..., description="Core ML task type (classification, detection, segmentation, etc.)"
    )
    data_modality: List[str] = Field(
        ..., description="Types of data involved"
    )
    key_techniques: List[str] = Field(
        ..., description="Relevant ML/DL techniques or architectures"
    )
    expected_outputs: List[str] = Field(
        ..., description="Expected outputs of the system or model"
    )


class SubTask(BaseModel):
    agent: str = Field(..., description="Agent to be invoked")
    goal: str = Field(..., description="Goal of the agent")
    rationale: str = Field(..., description="Why this agent is needed")
    inputs: List[str] = Field(
        ..., description="Inputs or constraints for the agent"
    )


class PlannerOutput(BaseModel):
    analysis: Analysis
    subtasks: List[SubTask]


# --------------------------------------------------
# Initialize Output Parser
# --------------------------------------------------

parser = PydanticOutputParser(pydantic_object=PlannerOutput)

# --------------------------------------------------
# Initialize Groq LLM (Planner = Strong Model)
# --------------------------------------------------

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
    api_key=os.getenv("GROQ_API")
)

# --------------------------------------------------
# Planner Prompt (ANALYTICAL)
# --------------------------------------------------

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a Research Planning Agent in a multi-agent AI system.

Your task is to ANALYZE the research query and produce a structured execution plan.

You must:
- Infer research domain and sub-domain
- Identify the core ML problem type
- Identify data modality
- Extract key technical concepts
- Justify why each agent must be invoked

Rules:
- Do NOT answer the research question
- Do NOT generate papers or datasets
- Output ONLY valid JSON
- Follow the schema exactly

Available agents:
1. paper_agent
2. dataset_agent
4. action_plan_agent

{format_instructions}
"""),
    ("human", "{query}")
]).partial(
    format_instructions=parser.get_format_instructions()
)

# --------------------------------------------------
# Build Planner Chain
# --------------------------------------------------

planner_chain = planner_prompt | llm | parser


# --------------------------------------------------
# Utility: Save output to JSON
# --------------------------------------------------

def save_output_to_json(data: dict, filename: str = "outputs/planner_agent_output.json") -> None:
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


# --------------------------------------------------
# Run Planner Agent
# --------------------------------------------------

def run_planner(query: str) -> dict:
    """
    Runs the planner agent on a research query and saves output
    """
    output = planner_chain.invoke({"query": query})
    save_output_to_json(output.dict())
    return output.dict()


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    query = "Deep Learning methods to classify AI generated music and human generated music"
    plan = run_planner(query)
    print(json.dumps(plan, indent=4))
