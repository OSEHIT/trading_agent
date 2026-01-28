from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import io
import os
from src.monitoring.guardian import DriftGuardian

app = FastAPI(title="Monitoring Service (Evidently)", version="1.0")
guardian = DriftGuardian()
REPORT_DIR = "/app/monitoring/reports"
os.makedirs(REPORT_DIR, exist_ok=True)

@app.get("/")
def health_check():
    return {"status": "Monitoring Service Operational"}

@app.post("/check_drift")
async def check_drift(
    reference_file: UploadFile = File(...),
    current_file: UploadFile = File(...)
):
    """
    Receives two CSV files (reference and current), checks for drift.
    Returns JSON status.
    """
    try:
        ref_df = pd.read_csv(io.BytesIO(await reference_file.read()))
        curr_df = pd.read_csv(io.BytesIO(await current_file.read()))
        
        result = guardian.check_drift(ref_df, curr_df)
        
        # Also generate HTML for viewing
        report_path = os.path.join(REPORT_DIR, "latest.html")
        guardian.generate_html_report(ref_df, curr_df, report_path)
        
        # SAVE JSON FOR AGENT/UI
        SHARED_REPORT_DIR = "/app/reporting"
        os.makedirs(SHARED_REPORT_DIR, exist_ok=True)
        json_path = os.path.join(SHARED_REPORT_DIR, "drift_report.json")
        with open(json_path, 'w') as f:
            json.dump(result, f)
        
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/report", response_class=HTMLResponse)
def get_report():
    """Serves the latest HTML report."""
    report_path = os.path.join(REPORT_DIR, "latest.html")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding='utf-8') as f:
            return f.read()
    return "<h1>No report generated yet.</h1>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)
