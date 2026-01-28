import json
import os
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field

class DriftAuditorInput(BaseModel):
    ignore: str = Field(default="", description="Ignore this input (tool takes no arguments)")

class DriftAuditor(BaseTool):
    name = "drift_auditor"
    description = "Checks the latest data stability report for any drift warnings. Input should be an empty string."
    args_schema: Type[BaseModel] = DriftAuditorInput

    def _run(self, ignore: str = "") -> str:
        # Ideally this reads from a shared volume where Evidently writes reports
        # For this setup, we'll look for a simulated report file
        report_path = "/app/reporting/drift_report.json" # Path in container
        
        if not os.path.exists(report_path):
             return "Drift Status: Unknown (No report found)"

        try:
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            # Simplified parsing logic
            drift_detected = report.get("drift_detected", False)
            score = report.get("drift_score", 0.0)
            
            if drift_detected:
                return f"⚠️ DRIFT DETECTED! Drift Score: {score}. The model may be unreliable."
            else:
                return f"✅ System Stable. Drift Score: {score}. Model is reliable."
                
        except Exception as e:
            return f"Error reading drift report: {str(e)}"

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("DriftAuditor does not support async")
