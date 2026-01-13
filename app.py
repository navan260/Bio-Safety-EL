import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fpdf import FPDF
from main import RagFetcher
import uvicorn
import uuid

# Initialize FastAPI app
app = FastAPI(title="Bio-Safety RAG Interface")

# Initialize RAG Fetcher
init_error = None
try:
    rag_fetcher = RagFetcher()
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error initializing RagFetcher: {e}")
    rag_fetcher = None
    init_error = str(e)

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class QueryRequest(BaseModel):
    food_name: str
    weather: str
    storage_duration: str
    additional_context: str = ""
    offline_mode: bool = False

class ReportRequest(BaseModel):
    food_name: str
    weather: str
    storage_duration: str
    additional_context: str = ""
    answer: str

@app.get("/")
async def read_root():
    from fastapi.responses import FileResponse
    return FileResponse('static/index.html')

@app.post("/api/generate")
async def generate_answer(request: QueryRequest):
    if not rag_fetcher:
        raise HTTPException(status_code=500, detail=f"RAG system not initialized. Error: {init_error}")
    
    # Construct the query from inputs
    query = (
        f"I have {request.food_name}. The weather is {request.weather}. "
        f"I plan to store it for {request.storage_duration}. "
        f"{request.additional_context} "
        "Based on this, how should I store it and what are the safety considerations?"
    )
    
    try:
        # Determine mode based on request
        mode = 'offline' if request.offline_mode else 'online'
        print(f"Generating answer for query: {query} (Mode: {mode})")
        
        answer = rag_fetcher.query(query, mode=mode)
        
        print(f"Answer generated successfully: {answer[:50]}...")
        return {"answer": answer}
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error generating answer: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.post("/api/report")
async def generate_report(request: ReportRequest):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Bio-Safety Food Storage Report", ln=True, align="C")
    pdf.ln(10)
    
    # Input Details Table
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Input Details:", ln=True)
    
    pdf.set_font("Arial", "", 12)
    col_width = 90
    row_height = 10
    
    data = [
        ("Food Name", request.food_name),
        ("Weather Conditions", request.weather),
        ("Storage Duration", request.storage_duration),
        ("Additional Context", request.additional_context)
    ]
    
    for row in data:
        pdf.cell(col_width, row_height, row[0], border=1)
        pdf.cell(col_width, row_height, str(row[1]), border=1)
        pdf.ln(row_height)
        
    pdf.ln(10)
    
    # Answer Section
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Recommendations & Safety Analysis:", ln=True)
    pdf.set_font("Arial", "", 11)
    
    # Multi-cell for text wrapping
    pdf.multi_cell(0, 7, request.answer)
    
    # Output file
    filename = f"reports/report_{uuid.uuid4()}.pdf"
    pdf.output(filename)
    
    from fastapi.responses import FileResponse
    return FileResponse(filename, filename="BioSafety_Report.pdf")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
