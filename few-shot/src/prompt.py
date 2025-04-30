import os
import json
from typing import Literal
from pydantic import BaseModel, Field

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

from utils.logging import get_logger
from utils.langchain_model import ChatModel

logger = get_logger()

# -------------------------------------------------------------------
# Example Selector
# -------------------------------------------------------------------

# Get the relative path to the examples file
EXAMPLES_FILE = os.path.join(os.path.dirname(__file__), "examples.json")

# Load the examples from the JSONL file
with open(EXAMPLES_FILE, "r") as f:
    examples = json.loads(f.read())

# Define the ExampleSelector
example_selector = SemanticSimilarityExampleSelector.from_examples(
    # This is the list of examples available to select from.
    examples,
    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
    OpenAIEmbeddings(),
    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
    Chroma(persist_directory="chroma_db"),
    # This is the number of examples to produce.
    k=3,
)

# -------------------------------------------------------------------
# Chat Model
# -------------------------------------------------------------------
model = ChatModel()


# -------------------------------------------------------------------
# Parser Models
# -------------------------------------------------------------------
class MistakeIdentificationModel(BaseModel):
    Mistake_Identification: Literal[
        "Yes",
        "No",
        "To some extent",
        "",
    ] = Field(default="")


# -------------------------------------------------------------------
# Prompt
# -------------------------------------------------------------------
prefix = """
You will be shown a short educational "Conversation" between a tutor and a student, including the student's solution and the tutor's follow-up "Response". Your task is to judge whether the tutor's response successfully **identifies a mistake** in the student's reasoning.

### Instructions
1. Read the entire dialogue to understand the context of the student's solution.  
2. Focus on whether the tutor's response explicitly or implicitly calls out an error.  
3. Reply **only** with one of the labels: `Yes`, `To some extent`, or `No`.  

### Labels
- `Yes`: The mistake is clearly identified/recognized in the tutor's response. The tutor implicitly or explicitly points out the error in the student's reasoning.
- `No`: The tutor's response does not identify any mistake in the student's reasoning. The tutor's response is either irrelevant or does not address the student's solution.
- `To some extent`: The tutor's response suggests that there may be a mistake, but it sounds as if the tutor is not certain.

### Format Instructions:
{format_instructions}
Return only the classification label without any additional commentary or extraneous details.

### Examples
"""

suffix = """
### Mistake Identification

### Conversation
{conversation}

### Response
{response}
"""

example_prompt = PromptTemplate.from_template(
    "Conversion: {conversation} \nResponse: {response}",
)

prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
)

parser = PydanticOutputParser(pydantic_object=MistakeIdentificationModel)
prompt = prompt.partial(
    format_instructions=parser.get_format_instructions(),
)

# -------------------------------------------------------------------
# Chain
# -------------------------------------------------------------------

chain = prompt | model | parser
