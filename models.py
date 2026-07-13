from pydantic import BaseModel, Field

class QuestionInput(BaseModel):
    question: str = Field(min_length=2)
    limit: int = Field(default=3, ge=1, le=10)

class LoadInput(BaseModel):
    file_path: str = Field(default="Movment at AIRPORTS.xlsx")