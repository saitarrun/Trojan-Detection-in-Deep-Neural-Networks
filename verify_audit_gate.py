#!/usr/bin/env python3
import json
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Gemini IARPA Audit Gate: Automated CI/CD Model Security Gating")
    parser.add_argument("report_path", help="Path to the generated Gemini IARPA Audit JSON report")
    parser.add_argument("--threshold", type=float, default=0.5, help="Risk score threshold (default: 0.5)")
    args = parser.parse_args()

    try:
        with open(args.report_path, 'r') as f:
            report = json.load(f)
    except Exception as e:
        print(f"Error: Could not read report at {args.report_path}: {e}")
        sys.exit(1)

    risk_score = report.get("model_summary", {}).get("risk_fusion_score", 1.0)
    verdict = report.get("model_summary", {}).get("verdict", "UNKNOWN")
    model_name = report.get("report_metadata", {}).get("task_id", "Unknown-Model")

    print("="*60)
    print(f"GEMINI IARPA AUDIT GATE: {model_name}")
    print("="*60)
    print(f"Unified Risk Score: {risk_score:.4f}")
    print(f"Program Verdict:    {verdict}")
    print(f"Gate Threshold:     {args.threshold}")
    print("-"*60)

    if risk_score > args.threshold:
        print(f"CRITICAL: Model risk score ({risk_score:.4f}) EXCEEDS security threshold ({args.threshold}).")
        print("ACTION: CI/CD PIPELINE BLOCKED. DEPLOYMENT REJECTED.")
        print("="*60)
        sys.exit(1)
    else:
        print("SUCCESS: Model passed security audit. Risk is within operational bounds.")
        print("ACTION: CI/CD PIPELINE PROCEEDED. DEPLOYMENT APPROVED.")
        print("="*60)
        sys.exit(0)

if __name__ == "__main__":
    main()
