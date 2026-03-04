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
  Search,
  Download,
  ShieldCheck,
  Server,
  Database,
  Cpu,
  BarChart3
} from 'lucide-react';

// API Configuration
const API_BASE = typeof window !== 'undefined'
  ? (window.location.port === '8000' || window.location.port === '3000'
    ? ""
    : window.location.pathname.includes('/proxy/3000/')
      ? window.location.pathname.split('/proxy/3000/')[0] + '/proxy/8000'
      : "")
  : "";

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

    const fetchUrl = `${API_BASE}/api/v1/scan-model`;

    try {
      const response = await fetch(fetchUrl, {
        method: 'POST',
        body: formData,
        cache: 'no-store'
      });

      if (!response.ok) throw new Error("Audit initiation failed.");

      const data = await response.json();
      setTaskId(data.task_id);
    } catch (err: any) {
      setError(err.message);
      setIsScanning(false);
    }
  };

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
          setError(data.message || "Audit failed.");
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
    const id = result?.task_id || taskId;
    if (!id) return;
    try {
      const response = await fetch(`${API_BASE}/api/v1/audit-report/${id}`);
      if (!response.ok) throw new Error("Failed to generate report.");
      const data = await response.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `Gemini_Audit_${id.substring(0, 8)}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err: any) {
      alert("Error: " + err.message);
    }
  };

  const AuditStep = ({ label, status, subtext }: { label: string, status: 'pending' | 'active' | 'complete', subtext?: string }) => {
    const isActive = status === 'active';
    const isComplete = status === 'complete';

    return (
      <div className={`step-item ${isActive ? 'active' : ''} ${isComplete ? 'complete' : ''}`} style={{ opacity: isActive ? 1 : (isComplete ? 0.7 : 0.2) }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: '24px' }}>
          <div className={`pulsate`} style={{
            width: '24px', height: '24px', borderRadius: '50%',
            background: isComplete ? 'var(--success)' : (isActive ? 'var(--accent)' : 'rgba(255,255,255,0.05)'),
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            border: `1px solid ${isComplete ? 'var(--success)' : (isActive ? 'var(--accent)' : 'var(--card-border)')}`,
            animation: isActive ? 'pulse-ring 2s infinite' : 'none'
          }}>
            {isComplete ? <CheckCircle2 size={14} color="white" /> : (isActive ? <Loader2 className="animate-spin" size={14} color="white" /> : <div style={{ width: '4px', height: '4px', background: 'rgba(255,255,255,0.3)', borderRadius: '50%' }} />)}
          </div>
          <div style={{ width: '1px', height: '24px', background: 'var(--card-border)', margin: '4px 0' }}></div>
        </div>
        <div style={{ paddingBottom: '1.5rem', marginLeft: '1rem' }}>
          <p style={{ fontWeight: 700, fontSize: '0.9rem', color: isActive ? '#fff' : '#64748b' }}>{label}</p>
          {subtext && isActive && <p style={{ fontSize: '0.7rem', color: 'var(--accent)', marginTop: '0.2rem', fontWeight: 500 }}>{subtext}</p>}
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
      <aside className="sidebar">
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
          <div className="pulsate" style={{ width: '12px', height: '12px', borderRadius: '50%', background: 'var(--accent)' }}></div>
          <h2 className="title-gradient" style={{ fontSize: '1.4rem', fontWeight: 900, letterSpacing: '-0.02em' }}>GEMINI CORE</h2>
        </div>

        <div className="stagger-1" style={{ display: 'flex', flexDirection: 'column', gap: '1.8rem' }}>
          <div>
            <label className="label">Neural Model Source</label>
            <div
              className="glass-hover"
              style={{
                marginTop: '0.6rem', border: '2px dashed var(--card-border)', padding: '2rem 1.5rem',
                textAlign: 'center', cursor: 'pointer', borderRadius: '16px', transition: 'all 0.3s ease'
              }}
              onClick={() => document.getElementById('file-upload')?.click()}
            >
              <Upload size={28} style={{ color: 'var(--accent)', marginBottom: '0.75rem', opacity: 0.8 }} />
              <p style={{ fontSize: '0.8rem', color: '#94a3b8', fontWeight: 500 }}>
                {selectedFile ? selectedFile.name : "Drop .pth or .onnx bundle"}
              </p>
              <input id="file-upload" type="file" style={{ display: 'none' }} onChange={handleFileChange} accept=".pth,.onnx" />
            </div>
          </div>

          <div>
            <label className="label">Target Class Strategy</label>
            <select className="input-field" value={targetClass} onChange={(e) => setTargetClass(e.target.value)}>
              <option value="-1">Auto-Detect (Black-Box)</option>
              {[...Array(10)].map((_, i) => (
                <option key={i} value={i}>Class {i}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="label">Trigger Heuristic</label>
            <select className="input-field" value={triggerType} onChange={(e) => setTriggerType(e.target.value)}>
              {triggerOptions.map(opt => (
                <option key={opt} value={opt}>{opt}</option>
              ))}
            </select>
          </div>

          <button className="button-primary" onClick={startScan} disabled={!selectedFile || isScanning} style={{ width: '100%', marginTop: '0.5rem' }}>
            {isScanning ? <Loader2 className="animate-spin" size={20} /> : <Zap size={20} className="fill-current" />}
            {isScanning ? "Scrutinizing..." : "Execute Enterprise Audit"}
          </button>
        </div>

        <div style={{ marginTop: 'auto', paddingTop: '1.5rem', borderTop: '1px solid var(--card-border)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', opacity: 0.5 }}>
            <span style={{ fontSize: '0.65rem', fontWeight: 700, letterSpacing: '1px' }}>V2.2.0 ENTERPRISE</span>
            <ShieldCheck size={16} />
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header style={{ marginBottom: '4rem', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div className="stagger-1">
            <h1 style={{ fontSize: '3rem', fontWeight: 900, letterSpacing: '-0.04em', lineHeight: 1.1, marginBottom: '0.75rem' }}>
              MLOps Command Center
            </h1>
            <p style={{ color: '#94a3b8', maxWidth: '650px', fontSize: '1.05rem', lineHeight: 1.6 }}>
              Automated Forensic Audit Pipeline for Mission-Critical Vision Models.
              Powered by <span style={{ color: 'var(--accent)', fontWeight: 600 }}>RiskFusionEngine™</span>.
            </p>
          </div>

          <div className="stagger-2" style={{ display: 'flex', gap: '1.2rem' }}>
            <div className="card glass" style={{ padding: '0.8rem 1.4rem', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
              <p className="label" style={{ fontSize: '0.6rem', marginBottom: '0.4rem' }}>Infrastructure</p>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                <Server size={14} color="var(--success)" />
                <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--success)' }}>STABLE</span>
              </div>
            </div>
            <div className="card glass" style={{ padding: '0.8rem 1.4rem', border: '1px solid var(--accent-glow)' }}>
              <p className="label" style={{ fontSize: '0.6rem', marginBottom: '0.4rem' }}>Audit Mode</p>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                <Database size={14} color="var(--accent)" />
                <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--accent)' }}>DEEP SCAN</span>
              </div>
            </div>
          </div>
        </header>

        {error && (
          <div className="card stagger-1" style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--danger)', marginBottom: '3rem' }}>
            <div style={{ display: 'flex', gap: '1.25rem', alignItems: 'center' }}>
              <div style={{ padding: '0.75rem', borderRadius: '12px', background: 'var(--danger)' }}>
                <AlertTriangle color="white" size={24} />
              </div>
              <div>
                <p style={{ fontWeight: 800, color: '#fff', fontSize: '1rem' }}>Audit Execution Interrupted</p>
                <p style={{ fontSize: '0.9rem', color: 'rgba(255,255,255,0.7)', marginTop: '0.2rem' }}>{error}</p>
              </div>
            </div>
          </div>
        )}

        {!isScanning && !result && !error && (
          <div className="stagger-3" style={{ textAlign: 'center', marginTop: '8rem', padding: '4rem', opacity: 0.4 }}>
            <div style={{ position: 'relative', width: '80px', height: '80px', margin: '0 auto 2rem' }}>
              <div style={{ position: 'absolute', inset: 0, borderRadius: '50%', background: 'var(--accent)', opacity: 0.1, animation: 'pulse-ring 2s infinite' }}></div>
              <Search size={80} style={{ position: 'relative', color: '#334155' }} />
            </div>
            <h3 style={{ fontSize: '1.5rem', fontWeight: 800, color: '#fff', marginBottom: '0.75rem' }}>No Active Scrutiny</h3>
            <p style={{ fontSize: '1rem' }}>Deploy a neural bundle to initialize forensic diagnostics.</p>
          </div>
        )}

        {isScanning && (
          <div className="stagger-1">
            <div className="card glass" style={{ border: '1px solid rgba(99, 102, 241, 0.2)', padding: '2.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '3rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <div style={{ background: 'var(--accent)', padding: '0.75rem', borderRadius: '14px' }}>
                    <Cpu size={28} color="white" />
                  </div>
                  <div>
                    <h3 style={{ fontSize: '1.4rem', fontWeight: 800 }}>Orchestrating Forensic Pipeline</h3>
                    <p style={{ color: '#94a3b8', fontSize: '0.85rem' }}>System-level audit of model tensors and latent activations.</p>
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <span className="badge badge-warning" style={{ fontSize: '0.8rem', padding: '0.5rem 1.2rem' }}>{progress}% ANALYZED</span>
                  <p style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 800, marginTop: '0.6rem', letterSpacing: '1px' }}>ID: {taskId?.substring(0, 12)}</p>
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '2rem' }}>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <AuditStep label="Network Initialization" status={getStepStatus('INIT')} subtext="Decoupling model weights..." />
                  <AuditStep label="Neural Inversion" status={getStepStatus('NC')} subtext={scanStatus} />
                </div>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <AuditStep label="STRIP Robustness" status={getStepStatus('STRIP')} />
                  <AuditStep label="Activation Clustering" status={getStepStatus('AC')} />
                </div>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <AuditStep label="Linear Weight Audit" status={getStepStatus('LWA')} />
                  <AuditStep label="Risk Fusion Execution" status={getStepStatus('FUSION')} />
                </div>
              </div>

              <div style={{ marginTop: '2.5rem', background: 'rgba(255,255,255,0.03)', height: '10px', borderRadius: '5px', position: 'relative', overflow: 'hidden' }}>
                <div
                  style={{
                    position: 'absolute', top: 0, left: 0, height: '100%',
                    background: 'linear-gradient(90deg, var(--accent), var(--cyan))',
                    width: `${progress}%`, borderRadius: '5px',
                    transition: 'width 1s cubic-bezier(0.4, 0, 0.2, 1)'
                  }}
                />
              </div>
            </div>
          </div>
        )}

        {result && (
          <div className="stagger-1">
            <div className="grid-cols-2">
              <div style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>
                <div className="card glass stagger-2" style={{ borderLeft: `6px solid ${result.fusion_risk_score > 0.5 ? 'var(--danger)' : 'var(--success)'}`, padding: '2.5rem' }}>
                  <label className="label" style={{ marginBottom: '1rem', display: 'block' }}>Unified Trojan Integrity Score</label>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '2rem' }}>
                    <h2 style={{ fontSize: '5.5rem', fontWeight: 900, lineHeight: 1, letterSpacing: '-0.05em' }}>
                      {(result.fusion_risk_score * 100).toFixed(0)}<span style={{ fontSize: '2.5rem', opacity: 0.3 }}>%</span>
                    </h2>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                      <span className={result.fusion_risk_score > 0.5 ? 'badge badge-danger' : 'badge badge-success'} style={{ fontSize: '0.85rem', padding: '0.6rem 1.5rem' }}>
                        {result.fusion_risk_score > 0.5 ? 'CRITICAL RISK DETECTED' : 'DEEMED INTEGRITY SECURE'}
                      </span>
                      <p style={{ fontSize: '0.85rem', color: '#94a3b8', maxWidth: '180px', lineHeight: 1.4 }}>
                        Weighted analysis of 5 cross-verified forensic signals.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="card glass stagger-3">
                  <h3 style={{ marginBottom: '2rem', display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '1.2rem', fontWeight: 800 }}>
                    <BarChart3 size={22} color="var(--accent)" />
                    Deep Telemetry Signatures
                  </h3>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                      <div className="telemetry-row">
                        <p className="label" style={{ fontSize: '0.6rem' }}>STRIP Entropy</p>
                        <p style={{ fontSize: '1.2rem', fontWeight: 700, marginTop: '0.2rem' }}>{result.details.strip_fr_ratio.toFixed(4)}</p>
                      </div>
                      <div className="telemetry-row">
                        <p className="label" style={{ fontSize: '0.6rem' }}>AC Silhouette</p>
                        <p style={{ fontSize: '1.2rem', fontWeight: 700, marginTop: '0.2rem' }}>{result.details.clustering_silhouette_score.toFixed(4)}</p>
                      </div>
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                      <div className="telemetry-row">
                        <p className="label" style={{ fontSize: '0.6rem' }}>Weight Anomaly</p>
                        <p style={{ fontSize: '1.2rem', fontWeight: 700, marginTop: '0.2rem' }}>{result.details.weight_analysis_risk.toFixed(4)}</p>
                      </div>
                      <div className="telemetry-row">
                        <p className="label" style={{ fontSize: '0.6rem' }}>Natural Bias</p>
                        <p style={{ fontSize: '1.2rem', fontWeight: 700, marginTop: '0.2rem', color: result.details.natural_trojan_risk > 0.4 ? 'var(--warning)' : 'inherit' }}>
                          {(result.details.natural_trojan_risk * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                  </div>

                  <div style={{ marginTop: '2.5rem', padding: '1.5rem', background: 'rgba(255,255,255,0.02)', borderRadius: '14px', border: '1px solid var(--card-border)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div>
                        <p className="label" style={{ fontSize: '0.6rem' }}>Neural Cleanse Result</p>
                        <p style={{ fontSize: '0.95rem', fontWeight: 700, marginTop: '0.3rem', color: result.details.nc_flagged_classes.length > 0 ? 'var(--danger)' : 'var(--success)' }}>
                          {result.details.nc_flagged_classes.length > 0 ? `Flagged Target: Class ${result.details.nc_flagged_classes[0]}` : 'No Trigger Signatures Recorded'}
                        </p>
                      </div>
                      <ShieldCheck size={24} color={result.details.nc_flagged_classes.length > 0 ? 'var(--danger)' : 'var(--success)'} style={{ opacity: 0.5 }} />
                    </div>
                  </div>

                  <button className="button-primary stagger-4" style={{ marginTop: '2.5rem', width: '100%', background: 'rgba(255,255,255,0.05)', border: '1px solid var(--card-border)', boxShadow: 'none' }} onClick={downloadReport}>
                    <Download size={18} />
                    Export Forensic IARPA-Audit Report
                  </button>
                </div>
              </div>

              <div className="card glass stagger-2" style={{ padding: '0', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
                <div style={{ padding: '2rem', borderBottom: '1px solid var(--card-border)' }}>
                  <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '1.2rem', fontWeight: 800 }}>
                    <Layout size={22} color="var(--accent)" />
                    Mechanistic Interpretability
                  </h3>
                </div>

                <div style={{ flex: 1, minHeight: '400px', padding: '2.5rem', position: 'relative' }}>
                  {result.gradcam_heatmap_b64 ? (
                    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
                      <div style={{ border: '1px solid var(--card-border)', borderRadius: '16px', overflow: 'hidden', boxShadow: '0 20px 50px rgba(0,0,0,0.5)' }}>
                        <img src={`data:image/png;base64,${result.gradcam_heatmap_b64}`} alt="Grad-CAM" style={{ width: '100%', display: 'block' }} />
                      </div>
                      <div className="card" style={{ background: 'rgba(0,0,0,0.3)', border: '1px dashed var(--card-border)', padding: '1.5rem' }}>
                        <p style={{ fontSize: '0.85rem', color: '#94a3b8', lineHeight: 1.6 }}>
                          <span style={{ color: '#fff', fontWeight: 600, display: 'block', marginBottom: '0.4rem' }}>Spatial Saliency Audit</span>
                          Visual telemetry highlighting localized tensor activations. High-intensity convergence areas often correlate to pixel-space Trojan backdoors.
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', opacity: 0.2 }}>
                      <Activity size={64} style={{ marginBottom: '1.5rem' }} />
                      <p style={{ fontWeight: 700 }}>Visual telemetry offline for this architecture.</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      <style jsx global>{`
        .animate-spin { animation: spin 1s linear infinite; }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        
        @keyframes pulse-ring {
          0% { transform: scale(1); box-shadow: 0 0 0 0 var(--accent-glow); }
          70% { transform: scale(1.05); box-shadow: 0 0 0 10px rgba(99, 102, 241, 0); }
          100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(99, 102, 241, 0); }
        }

        .telemetry-row {
          padding-left: 1rem;
          border-left: 2px solid var(--card-border);
          transition: all 0.3s ease;
        }
        .telemetry-row:hover { border-left-color: var(--accent); }
      `}</style>
    </div>
  );
}
