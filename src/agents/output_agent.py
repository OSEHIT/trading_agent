
class OutputAgent:
    """
    Role: Financial Expert
    Responsibilities: Synthesize inputs and check reliability.
    """
    
    def synthesize(self, context, prediction, drift_info, current_price):
        """
        Combines all analysis into a final recommendation.
        """
        # Unwrap values if they are single-element arrays/tensors
        try:
            pred_val = float(prediction)
        except:
            pred_val = prediction

        try:
            curr_val = float(current_price)
        except:
            curr_val = current_price

        # Basic Heuristic logic
        signal = "HOLD"
        confidence = 0.6
        
        # Drift Check
        drift_detected = drift_info.get("drift_detected", False)
        drift_score = drift_info.get("drift_score", 0.0)

        if drift_detected:
            confidence = 0.2
            explanation = (
                f"⚠️ ALERT: Our reliability monitor detected data drift (score: {drift_score:.4f}). "
                "The market behavior has shifted significantly from our training data. "
                "The model prediction should be treated with extreme caution until retraining occurs."
            )
        else:
            # Price logic
            if pred_val > curr_val * 1.01:
                signal = "BUY"
                explanation = f"Model predicts a rise to ${pred_val:.2f} (Current: ${curr_val:.2f}). Market sentiment is {context}."
            elif pred_val < curr_val * 0.99:
                signal = "SELL"
                explanation = f"Model predicts a drop to ${pred_val:.2f} (Current: ${curr_val:.2f}). Market sentiment is {context}."
            else:
                explanation = f"Model predicts consolidation around ${pred_val:.2f}. Sentiment: {context}."

        return {
            "signal": signal,
            "target_price": pred_val,
            "current_price": curr_val,
            "confidence": confidence,
            "rationale": explanation,
            "drift_status": drift_info
        }
