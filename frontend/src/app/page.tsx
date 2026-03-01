"use client";

import React, { useState, useEffect } from 'react';
import {
  Shield,
  Zap,
  Upload,
  Activity,
  FileText,
  Layout,
  ChevronRight,
  CheckCircle2,
  AlertTriangle,
  Loader2,
  RefreshCcw,
  Search,
  Download,
  AlertOctagon
} from 'lucide-react';

// API Configuration
const API_BASE = ""; // Next.js proxy for status/reports
export default function Dashboard() {


  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [targetClass, setTargetClass] = useState("-1");
  const [triggerType, setTriggerType] = useState("Auto-Detect (Black-Box)");
  const [isScanning, setIsScanning] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [scanStatus, setScanStatus] = useState<string>("IDLE");
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  // Trigger Types listing
  const triggerOptions = [
    "Auto-Detect (Black-Box)",
    "checkerboard",
    "square",
    "blending",
    "clean_label",
    "dynamic",
    "instagram_filter",
    "spatial_conditional",
    "natural_trojan_bias"
  ];

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const startScan = async () => {
    if (!selectedFile) return;

    setIsScanning(true);
    setError(null);
    setResult(null);
    setProgress(10);
    setScanStatus("INITIALIZING");

    const formData = new FormData();
    formData.append('model_file', selectedFile);
    formData.append('target_class', targetClass);
    formData.append('trigger_type', triggerType);

    const directApiBase = typeof window !== 'undefined'
      ? `${window.location.protocol}//${window.location.hostname}:8000`
      : "http://localhost:8000";

    const fetchUrl = `${directApiBase}/api/v1/scan-model`;

    try {
      console.log(`[STABLE-v2.1.2] Bypassing Proxy. Targeted Internal Port 8000: ${fetchUrl}`);
      const response = await fetch(fetchUrl, {
        method: 'POST',
        body: formData,
        cache: 'no-store'
      });

      if (!response.ok) throw new Error("Failed to submit model for scanning.");

      const data = await response.json();
      setTaskId(data.task_id);
      setScanStatus("ACCEPTED");
    } catch (err: any) {
      setError(err.message);
      setIsScanning(false);
    }
  };

  // Polling logic
  useEffect(() => {
    if (!taskId || !isScanning) return;

    const interval = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE}/api/v1/scan-status/${taskId}`);
        if (!response.ok) return;

        const data = await response.json();
        setScanStatus(data.status);

        if (data.status === 'PROGRESS') {
          setProgress((prev) => Math.min(prev + 5, 90));
        } else if (data.status === 'SUCCESS') {
          setResult(data.result);
          setIsScanning(false);
          setTaskId(null);
          setProgress(100);
          clearInterval(interval);
        } else if (data.status === 'FAILURE') {
          setError(data.message || "Scan failed.");
          setIsScanning(false);
          setTaskId(null);
          clearInterval(interval);
        }
      } catch (err) {
        console.error("Polling error:", err);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [taskId, isScanning]);

  const downloadReport = async () => {
    if (!result || !result.task_id) {
      // Fallback if task_id isn't in result (it should be in scanStatus check though)
      if (!taskId && !result.task_id) return;
    }
    const id = result.task_id || taskId;
    try {
      const response = await fetch(`${API_BASE}/api/v1/audit-report/${id}`);
      if (!response.ok) throw new Error("Failed to generate report.");
      const data = await response.json();

      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `Gemini_IARPA_Audit_${id}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err: any) {
      alert("Error exporting report: " + err.message);
    }
  };

  const AuditStep = ({ label, status, subtext }: { label: string, status: 'pending' | 'active' | 'complete', subtext?: string }) => {
    const isActive = status === 'active';
    const isComplete = status === 'complete';

    return (
      <div className={`step-item ${isActive ? 'active' : ''} ${isComplete ? 'complete' : ''}`}>
        <div className="step-marker-container">
          <div className="step-marker">
            {isComplete ? <CheckCircle2 size={16} /> : (isActive ? <Loader2 className="animate-spin" size={16} /> : <div className="dot" />)}
          </div>
          <div className="step-line"></div>
        </div>
        <div className="step-text-container">
          <p className="step-label">{label}</p>
          {subtext && isActive && <p className="step-subtext">{subtext}</p>}
        </div>
      </div>
    );
  };

  const getStepStatus = (stepName: string) => {
    const msg = scanStatus.toUpperCase();

    const steps = [
      { id: 'INIT', keys: ['INITIALIZING', 'LOADING'] },
      { id: 'NC', keys: ['NEURAL CLEANSE', 'TRIGGER'] },
      { id: 'STRIP', keys: ['STRIP', 'ENTROPY'] },
      { id: 'AC', keys: ['CLUSTERING', 'ACTIVATION'] },
      { id: 'LWA', keys: ['WEIGHT ANALYSIS', 'LINEAR'] },
      { id: 'FUSION', keys: ['FUSION', 'RISK'] }
    ];

    const currentIdx = steps.findIndex(s => s.keys.some(k => msg.includes(k)));
    const stepIdx = steps.findIndex(s => s.id === stepName);

    if (currentIdx === -1 && msg === 'ACCEPTED') return stepIdx === 0 ? 'active' : 'pending';
    if (currentIdx === -1) return 'pending';
    if (stepIdx < currentIdx) return 'complete';
    if (stepIdx === currentIdx) return 'active';
    return 'pending';
  };

  return (
    <div className="dashboard-container">
      {/* Sidebar Content */}
      <aside className="sidebar glass">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
          <div className="status-ring pulsate" style={{ background: 'var(--accent)' }}></div>
          <h2 className="title-gradient" style={{ fontSize: '1.25rem', fontWeight: 800 }}>GEMINI CORE</h2>
        </div>

        <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div>
            <label className="label">Neural Model Source</label>
            <div
              className="input-field"
              style={{ border: '2px dashed var(--card-border)', padding: '1.5rem', textAlign: 'center', cursor: 'pointer', position: 'relative' }}
              onClick={() => document.getElementById('file-upload')?.click()}
            >
              <Upload size={24} style={{ color: 'var(--accent)', marginBottom: '0.5rem' }} />
              <p style={{ fontSize: '0.8rem', color: '#a0a0b0' }}>
                {selectedFile ? selectedFile.name : "Click to upload .pth or .onnx"}
              </p>
              <input
                id="file-upload"
                type="file"
                style={{ display: 'none' }}
                onChange={handleFileChange}
                accept=".pth,.onnx"
              />
            </div>
          </div>

          <div>
            <label className="label">Target Class Strategy</label>
            <select
              className="input-field"
              value={targetClass}
              onChange={(e) => setTargetClass(e.target.value)}
            >
              <option value="-1">Auto-Detect (Black-Box)</option>
              {[...Array(10)].map((_, i) => (
                <option key={i} value={i}>Class {i}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="label">Trigger Heuristic</label>
            <select
              className="input-field"
              value={triggerType}
              onChange={(e) => setTriggerType(e.target.value)}
            >
              {triggerOptions.map(opt => (
                <option key={opt} value={opt}>{opt}</option>
              ))}
            </select>
          </div>

          <button
            className="button-primary"
            style={{ marginTop: '1rem', width: '100%' }}
            onClick={startScan}
            disabled={!selectedFile || isScanning}
          >
            {isScanning ? <Loader2 className="animate-spin" size={20} /> : <Zap size={20} />}
            {isScanning ? "Auditing Network..." : "Execute Enterprise Audit"}
          </button>
        </div>

        <div style={{ marginTop: 'auto', paddingTop: '1rem', borderTop: '1px solid var(--card-border)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.7rem', color: '#666' }}>VERSION 2.1.2-STABLE</span>
            <Shield size={14} style={{ color: '#444' }} />
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        <header style={{ marginBottom: '3rem', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div className="fade-in">
            <h1 style={{ fontSize: '2.5rem', fontWeight: 900, marginBottom: '0.5rem' }}>MLOps Command Center</h1>
            <p style={{ color: '#a0a0b0', maxWidth: '600px' }}>
              Advanced Neural Trojan Auditing & Forensic Sanitization Pipeline. Deployment-grade security for mission-critical vision models.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '1rem' }}>
            <div className="glass" style={{ padding: '0.75rem 1.25rem', borderRadius: '12px', textAlign: 'center' }}>
              <p className="label" style={{ fontSize: '0.6rem' }}>Pipeline Status</p>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.2rem' }}>
                <Activity size={14} style={{ color: 'var(--success)' }} />
                <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--success)' }}>OPERATIONAL</span>
              </div>
            </div>
            <div className="glass" style={{ padding: '0.75rem 1.25rem', borderRadius: '12px', textAlign: 'center', border: '1px solid rgba(138, 43, 226, 0.3)' }}>
              <p className="label" style={{ fontSize: '0.6rem' }}>Analysis Engine</p>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.2rem' }}>
                <Zap size={14} style={{ color: 'var(--accent)' }} />
                <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--accent)' }}>DIRECT CORE</span>
              </div>
            </div>
          </div>
        </header>

        {error && (
          <div className="card fade-in" style={{ background: 'rgba(255, 118, 117, 0.1)', border: '1px solid var(--danger)' }}>
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
              <AlertTriangle style={{ color: 'var(--danger)' }} />
              <div>
                <p style={{ fontWeight: 'bold', color: 'var(--danger)' }}>Audit Interrupted</p>
                <p style={{ fontSize: '0.9rem', opacity: 0.8 }}>{error}</p>
              </div>
            </div>
          </div>
        )}

        {!isScanning && !result && !error && (
          <div className="fade-in" style={{ textAlign: 'center', marginTop: '10rem', opacity: 0.5 }}>
            <Search size={64} style={{ margin: '0 auto 1.5rem', color: '#333' }} />
            <h3 style={{ fontSize: '1.2rem', color: '#fff' }}>No Active Audit</h3>
            <p>Upload a model and configure parameters to begin scanning.</p>
          </div>
        )}

        {isScanning && (
          <div className="fade-in">
            <div className="card glass" style={{ border: '1px solid rgba(138, 43, 226, 0.2)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <Activity size={24} style={{ color: 'var(--accent)' }} />
                  <h3 style={{ fontSize: '1.25rem', fontWeight: 700 }}>Orchestrating Audit Pipeline</h3>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <span className="badge badge-warning" style={{ marginBottom: '0.2rem', display: 'inline-block' }}>{progress}% COMPLETE</span>
                  <p style={{ fontSize: '0.7rem', color: '#666', fontWeight: 600, letterSpacing: '0.05em' }}>SESSION: {taskId?.substring(0, 8)}</p>
                </div>
              </div>

              <div className="audit-stepper">
                <AuditStep label="Model Initialization" status={getStepStatus('INIT')} subtext="Decoupling model tensors and loading validation sets..." />
                <AuditStep label="Neural Cleanse" status={getStepStatus('NC')} subtext={scanStatus} />
                <AuditStep label="STRIP Analysis" status={getStepStatus('STRIP')} subtext="Perturbing inputs for entropy entropy cross-verification..." />
                <AuditStep label="Activation Clustering" status={getStepStatus('AC')} subtext="Scanning latent space for artificial bifurcations..." />
                <AuditStep label="Linear Weight Analysis" status={getStepStatus('LWA')} subtext="Auditing final layer for anomalous L2 norms..." />
                <AuditStep label="Risk Fusion Engine" status={getStepStatus('FUSION')} subtext="Synchronizing telemetry into unified risk score..." />
              </div>

              <div style={{ marginTop: '2rem', background: 'rgba(255,255,255,0.03)', height: '6px', borderRadius: '3px', position: 'relative', overflow: 'hidden' }}>
                <div
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    height: '100%',
                    background: 'linear-gradient(90deg, var(--accent), #8a2be2)',
                    width: `${progress}%`,
                    transition: 'width 1s cubic-bezier(0.4, 0, 0.2, 1)'
                  }}
                />
              </div>
            </div>
          </div>
        )}

        {result && (
          <div className="fade-in">
            <div className="grid-cols-2">
              {/* Left Column: Summary Metrics */}
              <div>
                <div className="card glass" style={{ borderLeft: `4px solid ${result.fusion_risk_score > 0.5 ? 'var(--danger)' : 'var(--success)'}` }}>
                  <label className="label">Unified Trojan Risk Score</label>
                  <div style={{ display: 'flex', alignItems: 'baseline', gap: '1rem', marginTop: '0.5rem' }}>
                    <h2 style={{ fontSize: '4rem', fontWeight: 900 }}>{(result.fusion_risk_score * 100).toFixed(0)}%</h2>
                    <span className={result.fusion_risk_score > 0.5 ? 'badge badge-danger' : 'badge badge-success'}>
                      {result.fusion_risk_score > 0.5 ? 'CRITICAL RISK' : 'LOW RISK'}
                    </span>
                  </div>
                  <p style={{ marginTop: '1rem', color: '#a0a0b0', fontSize: '0.9rem' }}>
                    Calculated via RiskFusionEngine™ integrating Neural Cleanse, STRIP, and Weight Analysis telemetry.
                  </p>
                </div>

                <div className="card glass">
                  <h3 style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Activity size={20} style={{ color: 'var(--accent)' }} />
                    Defense Telemetry
                  </h3>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1.2rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: '0.9rem' }}>STRIP Entropy Ratio</span>
                      <span style={{ fontWeight: 600 }}>{result.details.strip_fr_ratio.toFixed(4)}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: '0.9rem' }}>AC Silhouette Score</span>
                      <span style={{ fontWeight: 600 }}>{result.details.clustering_silhouette_score.toFixed(4)}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: '0.9rem' }}>Neural Cleanse Flag</span>
                      <span className={result.details.nc_flagged_classes.length > 0 ? 'text-danger' : 'text-success'}>
                        {result.details.nc_flagged_classes.length > 0 ? `Target Class ${result.details.nc_flagged_classes[0]}` : 'None'}
                      </span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: '0.9rem' }}>Natural Trojan Profiling</span>
                      <span className={result.details.natural_trojan_risk > 0.4 ? 'text-warning' : 'text-success'} style={{ fontWeight: 600 }}>
                        {(result.details.natural_trojan_risk * 100).toFixed(1)}% Bias
                      </span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: '0.9rem' }}>Weight Anomaly L2</span>
                      <span style={{ fontWeight: 600 }}>
                        {result.details.weight_analysis_risk.toFixed(4)}
                      </span>
                    </div>
                  </div>

                  <button
                    className="button-secondary"
                    style={{ marginTop: '2rem', width: '100%', border: '1px solid var(--card-border)', gap: '0.5rem' }}
                    onClick={downloadReport}
                  >
                    <Download size={18} />
                    Export IARPA-Audit Report
                  </button>
                </div>
              </div>

              {/* Right Column: Visualizations */}
              <div className="card glass">
                <h3 style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Layout size={20} style={{ color: 'var(--accent)' }} />
                  Mechanistic Interpretability
                </h3>

                {result.gradcam_heatmap_b64 ? (
                  <div style={{ borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--card-border)' }}>
                    <img
                      src={`data:image/png;base64,${result.gradcam_heatmap_b64}`}
                      alt="Grad-CAM Heatmap"
                      style={{ width: '100%', display: 'block' }}
                    />
                    <div style={{ padding: '1rem', background: 'rgba(0,0,0,0.5)', fontSize: '0.8rem', color: '#a0a0b0' }}>
                      Grad-CAM Heatmap showing model activation regions for Target Class. High intensity regions may indicate trigger localization.
                    </div>
                  </div>
                ) : (
                  <div style={{ height: '300px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: 'rgba(255,255,255,0.02)', borderRadius: '12px', border: '1px dashed var(--card-border)' }}>
                    <Activity size={48} style={{ color: '#222', marginBottom: '1rem' }} />
                    <p style={{ color: '#444' }}>Visual telemetry not generated for this model.</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>

  // Global styles for Stepper & Animations
      <style jsx global>{`
    .animate-spin {
      animation: spin 1s linear infinite;
    }
    @keyframes spin {
      from { transform: rotate(0deg); }
      to { transform: rotate(360deg); }
    }
    .text-danger { color: var(--danger); }
    .text-success { color: var(--success); }

    .audit-stepper {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }

    .step-item {
      display: flex;
      gap: 1.5rem;
      opacity: 0.3;
      transition: all 0.4s ease;
    }

    .step-item.active {
      opacity: 1;
      transform: translateX(10px);
    }

    .step-item.complete {
      opacity: 0.7;
    }

    .step-marker-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      width: 20px;
    }

    .step-marker {
      width: 20px;
      height: 20px;
      border-radius: 50%;
      background: rgba(255,255,255,0.05);
      border: 1px solid var(--card-border);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 1;
      color: var(--accent);
      font-size: 0.7rem;
    }

    .step-item.active .step-marker {
      background: var(--accent);
      color: white;
      border-color: var(--accent);
      box-shadow: 0 0 15px var(--accent);
    }

    .step-item.complete .step-marker {
      background: var(--success);
      color: white;
      border-color: var(--success);
    }

    .step-line {
      flex: 1;
      width: 2px;
      background: var(--card-border);
      margin: 4px 0;
    }

    .step-item:last-child .step-line {
      display: none;
    }

    .step-item.complete .step-line {
      background: var(--success);
    }

    .step-text-container {
      padding-bottom: 1.5rem;
    }

    .step-label {
      font-weight: 700;
      font-size: 0.95rem;
      color: #fff;
    }

    .step-subtext {
      font-size: 0.75rem;
      color: #a0a0b0;
      margin-top: 0.2rem;
      animation: fadeIn 0.3s ease;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(-5px); }
      to { opacity: 1; transform: translateY(0); }
    }
  `}</style>
    </div>
  );
}
