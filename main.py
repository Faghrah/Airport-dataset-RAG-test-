from fastapi import FastAPI, HTTPException
from models import QuestionInput,LoadInput
from rag import(create_vector_index,load_excel_to_database,search_rag,ask_rag,list_chunks,clear_database)

app=FastAPI()

@app.get("/")
async def home():
    return{"message":"Airport RAG API running"}

@app.post("/create-index")
async def create_index():
    try:
        result=create_vector_index()
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/load-excel")
async def load_excel(data:LoadInput):
    try:
        result=load_excel_to_database(data.file_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/search")
async def search_question(data:QuestionInput):
    try:
        result=search_rag(data.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/ask")
async def ask_question(data:QuestionInput):
    try:
        result=ask_rag(data.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/chunks")
async def chunks():
    return {"chunks":list_chunks()}

@app.delete("/clear")
async def clear():
    result=clear_database()
    return result