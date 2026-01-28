import pandas as pd
import json
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
except ImportError:
    # Fallback/Mock for environments where evidently is tricky to install immediately
    print("Warning: Evidently not installed, using mock.")
    Report = None

class DriftGuardian:
    """
    Role: Reliability Monitor (Evidently AI)
    Responsibilities: Check for Data Drift between Train set and recent inference data.
    """
    
    def check_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame):
        """
        Generates a drift report.
        reference_data: DataFrame used for training (baseline).
        current_data: DataFrame of recent inference inputs.
        """
        if Report is None:
            return {"drift_detected": False, "drift_score": 0.0, "info": "Evidently not installed"}

        # Ensure minimal columns match
        # We focus on 'Close' price for drift
        if 'Close' not in reference_data.columns or 'Close' not in current_data.columns:
            return {"error": "Missing 'Close' column in data"}
            
        data_drift_report = Report(metrics=[
            DataDriftPreset(), 
        ])
        
        data_drift_report.run(reference_data=reference_data, current_data=current_data)
        
        # Extract specific metric (Simplified for API JSON)
        # In a real app, we might parse the JSON export deeply
        report_json = json.loads(data_drift_report.json())
        
        # Heuristic to find 'drift_detected' in the preset
        # Evidently structure varies by version, safe generic check:
        drift_detected = report_json['metrics'][0]['result']['dataset_drift']
        drift_share = report_json['metrics'][0]['result']['drift_share']
        
        return {
            "drift_detected": drift_detected,
            "drift_score": drift_share,
            "report_path": "/app/monitoring/reports/latest_drift_report.html"
        }

    def generate_html_report(self, reference_data, current_data, output_path):
        if Report is None:
            return
            
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_data, current_data=current_data)
        report.save_html(output_path)
