"use client";

import React, { useState, useEffect } from 'react';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
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
  BarChart3,
  GitBranch,
  TrendingUp
} from 'lucide-react';
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
  Legend
} from 'recharts';

// ─── Signal Gauge Component ─────────────────────────────────────────────────
function SignalGauge({ label, value, max = 1.0 }: { label: string; value: number; max?: number }) {
  const pct = Math.min((value / max) * 100, 100);
  const color = pct > 70 ? '#ef4444' : pct > 40 ? '#f59e0b' : '#10b981';
  return (
    <div style={{ marginBottom: '1.1rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
        <span style={{ fontSize: '0.72rem', fontWeight: 700, letterSpacing: '0.5px', color: '#94a3b8' }}>{label}</span>
        <span style={{ fontSize: '0.72rem', fontWeight: 800, color }}>{(pct).toFixed(1)}%</span>
      </div>
      <div style={{ height: '7px', background: 'rgba(255,255,255,0.07)', borderRadius: '4px', overflow: 'hidden' }}>
        <div
          style={{
            height: '100%',
            width: `${pct}%`,
            background: `linear-gradient(90deg, ${color}99, ${color})`,
            borderRadius: '4px',
            transition: 'width 1.2s cubic-bezier(0.4,0,0.2,1)',
            boxShadow: `0 0 8px ${color}66`
          }}
        />
      </div>
    </div>
  );
}

// ─── Radar Chart Component ──────────────────────────────────────────────────
function RiskRadarChart({ details }: { details: any }) {
  const data = [
    { signal: 'Neural\nCleanse', risk: +((details.neural_cleanse_risk ?? 0) * 100).toFixed(1) },
    { signal: 'STRIP', risk: +((details.strip_risk ?? 0) * 100).toFixed(1) },
    { signal: 'Clustering', risk: +((details.clustering_risk ?? 0) * 100).toFixed(1) },
    { signal: 'Weight\nAudit', risk: +((details.weight_analysis_risk ?? 0) * 100).toFixed(1) },
    { signal: 'Natural\nTrojan', risk: +((details.natural_trojan_risk ?? 0) * 100).toFixed(1) },
    { signal: 'Gradient\nSimilarity', risk: +((details.gradient_similarity_risk ?? 0) * 100).toFixed(1) },
  ];
  return (
    <ResponsiveContainer width="100%" height={280}>
      <RadarChart data={data} margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
        <PolarGrid stroke="rgba(255,255,255,0.08)" />
        <PolarAngleAxis
          dataKey="signal"
          tick={{ fill: '#94a3b8', fontSize: 10, fontWeight: 700 }}
        />
        <PolarRadiusAxis
          angle={30}
          domain={[0, 100]}
          tick={{ fill: '#475569', fontSize: 8 }}
          tickCount={4}
        />
        <Radar
          name="Risk %"
          dataKey="risk"
          stroke="#6366f1"
          fill="#6366f1"
          fillOpacity={0.25}
          strokeWidth={2}
          dot={{ fill: '#6366f1', r: 4 }}
        />
        <Tooltip
          contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '10px', fontSize: '0.8rem' }}
          formatter={(v: any) => [`${v}%`, 'Risk Signal']}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
}

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
  const [useLocalPath, setUseLocalPath] = useState(false);
  const [localPath, setLocalPath] = useState("");

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
    if (!useLocalPath && !selectedFile) return;
    if (useLocalPath && !localPath) return;

    setIsScanning(true);
    setError(null);
    setResult(null);
    setProgress(10);
    setScanStatus("INITIALIZING");

    try {
      let response;
      if (useLocalPath) {
        response = await fetch(`${API_BASE}/api/v1/scan-local-path`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model_path: localPath,
            target_class: parseInt(targetClass),
            trigger_type: triggerType
          }),
          cache: 'no-store'
        });
      } else {
        const formData = new FormData();
        formData.append('model_file', selectedFile!);
        formData.append('target_class', targetClass);
        formData.append('trigger_type', triggerType);

        response = await fetch(`${API_BASE}/api/v1/scan-model`, {
          method: 'POST',
          body: formData,
          cache: 'no-store'
        });
      }

      if (!response.ok) {
        let errorMsg = "Audit initiation failed.";
        try {
          const contentType = response.headers.get("content-type");
          if (contentType && contentType.includes("application/json")) {
            const errData = await response.json();
            errorMsg = errData.detail || errorMsg;
          } else {
            const textError = await response.text();
            errorMsg = textError || `HTTP Error ${response.status}`;
          }
        } catch (e) {
          errorMsg = `Server connectivity issue (${response.status})`;
        }
        throw new Error(errorMsg);
      }

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
          setResult({ ...data.result, task_id: taskId });
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

      const doc = new jsPDF();

      // Title
      doc.setFontSize(22);
      doc.setTextColor(40, 40, 40);
      doc.text("IARPA TrojAI Security Audit Report", 14, 22);

      // Metadata
      doc.setFontSize(10);
      doc.setTextColor(100, 100, 100);
      doc.text(`Task ID: ${data.report_metadata?.task_id || id}`, 14, 30);
      doc.text(`Timestamp: ${data.report_metadata?.audit_timestamp || new Date().toISOString()}`, 14, 35);
      doc.text(`Version: ${data.report_metadata?.version || '1.0'}`, 14, 40);

      // Summary
      doc.setFontSize(14);
      doc.setTextColor(40, 40, 40);
      doc.text("Model Summary", 14, 55);

      autoTable(doc, {
        startY: 60,
        head: [['Metric', 'Value']],
        body: [
          ['Architecture', data.model_summary?.architecture || 'N/A'],
          ['Framework', data.model_summary?.framework || 'N/A'],
          ['Risk Fusion Score', `${((data.model_summary?.risk_fusion_score || 0) * 100).toFixed(1)}%`],
          ['Verdict', data.model_summary?.verdict || 'N/A'],
        ],
        theme: 'striped',
        headStyles: { fillColor: [41, 128, 185] },
      });

      // Forensics
      doc.text("Trojan Forensics", 14, (doc as any).lastAutoTable.finalY + 15);

      const forensicsBody = [];
      if (data.trojan_forensics) {
        const tf = data.trojan_forensics;
        if (tf.trigger_inversion) forensicsBody.push(['Trigger Inversion (NC)', `Anomaly Index: ${tf.trigger_inversion.neural_cleanse_index?.toFixed(2) || 'N/A'}`]);
        if (tf.test_time_checks) {
          forensicsBody.push(['STRIP False Acc.', `${((tf.test_time_checks.strip_false_acceptance || 0) * 100).toFixed(1)}%`]);
          forensicsBody.push(['STRIP False Rej.', `${((tf.test_time_checks.strip_false_rejection || 0) * 100).toFixed(1)}%`]);
        }
        if (tf.weight_analysis) forensicsBody.push(['Weight Anomaly L2', `${tf.weight_analysis.max_anomaly_l2_norm?.toFixed(2) || 'N/A'}`]);
        if (tf.natural_vulnerability_profiling) forensicsBody.push(['Shortcut Sensitivity', `${((tf.natural_vulnerability_profiling.shortcut_sensitivity || 0) * 100).toFixed(1)}%`]);
      }

      autoTable(doc, {
        startY: (doc as any).lastAutoTable.finalY + 20,
        head: [['Analysis Vector', 'Result']],
        body: forensicsBody.length > 0 ? forensicsBody : [['No Forensics Data', '--']],
        theme: 'striped',
        headStyles: { fillColor: [41, 128, 185] },
      });

      // Recommendations
      const lastY = (doc as any).lastAutoTable.finalY;
      doc.text("Strategic Recommendations", 14, lastY + 15);
      doc.setFontSize(10);
      doc.setTextColor(60, 60, 60);

      let recY = lastY + 22;
      const recs = data.strategic_recommendations || [];
      recs.forEach((rec: string) => {
        const textLines = doc.splitTextToSize(`• ${rec}`, 180);
        doc.text(textLines, 14, recY);
        recY += textLines.length * 6;
      });

      // Use explicit Blob download (bypasses some iframe/proxy blockers)
      const pdfBlob = doc.output('blob');
      const url = window.URL.createObjectURL(pdfBlob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `Gemini_Audit_${id.substring(0, 8)}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err: any) {
      alert("Error generating PDF: " + err.message);
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
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.6rem' }}>
              <label className="label">Neural Model Source</label>
              <button
                onClick={() => setUseLocalPath(!useLocalPath)}
                style={{ background: 'none', border: 'none', color: 'var(--accent)', fontSize: '0.7rem', fontWeight: 700, cursor: 'pointer', textDecoration: 'underline' }}
              >
                {useLocalPath ? "Switch to Upload" : "Use Server Path"}
              </button>
            </div>

            {useLocalPath ? (
              <input
                type="text"
                className="input-field"
                placeholder="/path/to/model.pth"
                value={localPath}
                onChange={(e) => setLocalPath(e.target.value)}
              />
            ) : (
              <div
                className="glass-hover"
                style={{
                  border: '2px dashed var(--card-border)', padding: '2rem 1.5rem',
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
            )}
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

          <button className="button-primary" onClick={startScan} disabled={(!useLocalPath && !selectedFile) || (useLocalPath && !localPath) || isScanning} style={{ width: '100%', marginTop: '0.5rem' }}>
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
                  <AuditStep label="Gradient Similarity" status={getStepStatus('GS')} />
                  <AuditStep label="6-Signal Fusion" status={getStepStatus('FUSION')} />
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

            {/* ── VERDICT BANNER ── */}
            {(() => {
              const infected = result.fusion_risk_score > 0.5;
              const borderColor = infected ? 'var(--danger)' : 'var(--success)';
              const bgColor = infected ? 'rgba(239,68,68,0.08)' : 'rgba(16,185,129,0.07)';
              const glowColor = infected ? 'rgba(239,68,68,0.25)' : 'rgba(16,185,129,0.2)';
              return (
                <div style={{
                  background: bgColor,
                  border: `2px solid ${borderColor}`,
                  borderRadius: '20px',
                  padding: '2rem 2.5rem',
                  marginBottom: '2rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '1.5rem',
                  boxShadow: `0 0 40px ${glowColor}`,
                  animation: infected ? 'verdict-pulse 2.5s ease-in-out infinite' : 'none',
                }}>
                  {/* Icon */}
                  <div style={{
                    width: '64px', height: '64px', borderRadius: '50%',
                    background: infected ? 'rgba(239,68,68,0.15)' : 'rgba(16,185,129,0.15)',
                    border: `2px solid ${borderColor}`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    flexShrink: 0,
                  }}>
                    {infected
                      ? <AlertTriangle size={30} color="var(--danger)" />
                      : <ShieldCheck size={30} color="var(--success)" />
                    }
                  </div>

                  {/* Text */}
                  <div style={{ flex: 1 }}>
                    <p style={{ fontSize: '0.65rem', fontWeight: 900, letterSpacing: '2px', color: borderColor, marginBottom: '0.3rem' }}>
                      AUDIT VERDICT
                    </p>
                    <h2 style={{ fontSize: '2rem', fontWeight: 900, letterSpacing: '-0.03em', color: '#fff', marginBottom: '0.4rem' }}>
                      {infected ? '⚠️ TROJAN DETECTED' : '✅ MODEL IS CLEAN'}
                    </h2>
                    <p style={{ fontSize: '0.88rem', color: '#94a3b8', lineHeight: 1.5, maxWidth: '600px' }}>
                      {infected
                        ? `This model exhibits strong Trojan indicators across multiple forensic channels. Confidence: ${(result.fusion_risk_score * 100).toFixed(0)}%. Do NOT deploy in a production environment.`
                        : `No Trojan implants were detected. The model passed all 6 forensic defense channels. Risk score: ${(result.fusion_risk_score * 100).toFixed(0)}%. Safe for deployment.`
                      }
                    </p>
                  </div>

                  {/* Score pill */}
                  <div style={{
                    padding: '1rem 2rem',
                    borderRadius: '14px',
                    background: infected ? 'rgba(239,68,68,0.15)' : 'rgba(16,185,129,0.12)',
                    border: `1px solid ${borderColor}`,
                    textAlign: 'center',
                    flexShrink: 0,
                  }}>
                    <p style={{ fontSize: '0.6rem', fontWeight: 800, letterSpacing: '1.5px', color: borderColor, marginBottom: '0.2rem' }}>RISK SCORE</p>
                    <p style={{ fontSize: '2.2rem', fontWeight: 900, lineHeight: 1, color: '#fff' }}>
                      {(result.fusion_risk_score * 100).toFixed(0)}<span style={{ fontSize: '1rem', opacity: 0.5 }}>%</span>
                    </p>
                  </div>
                </div>
              );
            })()}

            {/* ── Top Score Banner ── */}
            <div className="card glass stagger-1" style={{ borderLeft: `6px solid ${result.fusion_risk_score > 0.5 ? 'var(--danger)' : 'var(--success)'}`, padding: '2.5rem', marginBottom: '2.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1.5rem' }}>
              <div>
                <label className="label" style={{ display: 'block', marginBottom: '0.5rem' }}>Unified Trojan Integrity Score</label>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: '1rem' }}>
                  <h2 style={{ fontSize: '5rem', fontWeight: 900, lineHeight: 1, letterSpacing: '-0.05em' }}>
                    {(result.fusion_risk_score * 100).toFixed(0)}<span style={{ fontSize: '2rem', opacity: 0.3 }}>%</span>
                  </h2>
                  <span className={result.fusion_risk_score > 0.5 ? 'badge badge-danger' : 'badge badge-success'} style={{ fontSize: '0.8rem', padding: '0.5rem 1.2rem' }}>
                    {result.fusion_risk_score > 0.5 ? 'CRITICAL RISK DETECTED' : 'INTEGRITY SECURE'}
                  </span>
                </div>
                <p style={{ fontSize: '0.82rem', color: '#94a3b8', marginTop: '0.5rem' }}>6-signal confidence-weighted fusion · RiskFusionEngine™ v2</p>
              </div>
              <div style={{ display: 'flex', gap: '1rem' }}>
                <button className="button-primary" style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid var(--card-border)', boxShadow: 'none', padding: '0.6rem 1.4rem', fontSize: '0.82rem' }} onClick={downloadReport}>
                  <Download size={16} /> Export IARPA Report
                </button>
              </div>
            </div>

            {/* ── Radar + Gauges row ── */}
            <div className="grid-cols-2" style={{ marginBottom: '2.5rem' }}>
              {/* Radar Chart */}
              <div className="card glass stagger-2">
                <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '1.1rem', fontWeight: 800, marginBottom: '1.5rem' }}>
                  <TrendingUp size={20} color="var(--accent)" />
                  6-Signal Risk Radar
                </h3>
                <RiskRadarChart details={result.details} />
              </div>

              {/* Signal Gauges */}
              <div className="card glass stagger-2">
                <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '1.1rem', fontWeight: 800, marginBottom: '1.8rem' }}>
                  <BarChart3 size={20} color="var(--accent)" />
                  Defense Signal Breakdown
                </h3>
                <SignalGauge label="NEURAL CLEANSE" value={result.details.neural_cleanse_risk ?? 0} />
                <SignalGauge label="STRIP ROBUSTNESS" value={result.details.strip_risk ?? 0} />
                <SignalGauge label="ACTIVATION CLUSTERING" value={result.details.clustering_risk ?? 0} />
                <SignalGauge label="LINEAR WEIGHT AUDIT" value={result.details.weight_analysis_risk ?? 0} />
                <SignalGauge label="NATURAL TROJAN PROFILER" value={result.details.natural_trojan_risk ?? 0} />
                <SignalGauge label="GRADIENT SIMILARITY" value={result.details.gradient_similarity_risk ?? 0} />

                {/* NC Verdict */}
                <div style={{ marginTop: '1.5rem', padding: '1rem 1.25rem', background: 'rgba(255,255,255,0.03)', borderRadius: '10px', border: '1px solid var(--card-border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <p className="label" style={{ fontSize: '0.6rem' }}>Neural Cleanse Verdict</p>
                    <p style={{ fontSize: '0.9rem', fontWeight: 700, marginTop: '0.2rem', color: result.details.nc_flagged_classes.length > 0 ? 'var(--danger)' : 'var(--success)' }}>
                      {result.details.nc_flagged_classes.length > 0 ? `Trigger Detected → Class ${result.details.nc_flagged_classes[0]}` : 'No Trigger Signatures Found'}
                    </p>
                  </div>
                  <ShieldCheck size={22} color={result.details.nc_flagged_classes.length > 0 ? 'var(--danger)' : 'var(--success)'} style={{ opacity: 0.6 }} />
                </div>
              </div>
            </div>

            {/* ── GradCAM + Raw Telemetry row ── */}
            <div className="grid-cols-2" style={{ marginBottom: '2.5rem' }}>
              {/* GradCAM */}
              <div className="card glass stagger-3" style={{ padding: '0', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
                <div style={{ padding: '1.5rem 2rem', borderBottom: '1px solid var(--card-border)' }}>
                  <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '1.1rem', fontWeight: 800 }}>
                    <Layout size={20} color="var(--accent)" />
                    Mechanistic Interpretability (Grad-CAM)
                  </h3>
                </div>
                <div style={{ flex: 1, minHeight: '360px', padding: '2rem', position: 'relative' }}>
                  {result.gradcam_heatmap_b64 ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                      <div style={{ border: '1px solid var(--card-border)', borderRadius: '14px', overflow: 'hidden', boxShadow: '0 20px 50px rgba(0,0,0,0.5)' }}>
                        <img src={`data:image/jpeg;base64,${result.gradcam_heatmap_b64}`} alt="Grad-CAM Heatmap" style={{ width: '100%', display: 'block' }} />
                      </div>
                      <p style={{ fontSize: '0.82rem', color: '#94a3b8', lineHeight: 1.6 }}>
                        <span style={{ color: '#fff', fontWeight: 600, display: 'block', marginBottom: '0.3rem' }}>Spatial Saliency Audit</span>
                        High-intensity convergence indicates pixel-space Trojan anchor regions.
                      </p>
                    </div>
                  ) : (
                    <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: 0.08 }}>
                      <Activity size={90} />
                    </div>
                  )}
                </div>
              </div>

              {/* Raw Telemetry */}
              <div className="card glass stagger-3">
                <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '1.1rem', fontWeight: 800, marginBottom: '1.8rem' }}>
                  <GitBranch size={20} color="var(--accent)" />
                  Raw Telemetry Values
                </h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                  {[
                    { label: 'STRIP False Rejection', value: result.details.strip_fr_ratio?.toFixed(4) },
                    { label: 'STRIP False Acceptance', value: result.details.strip_fa_ratio?.toFixed(4) },
                    { label: 'AC Silhouette Score', value: result.details.clustering_silhouette_score?.toFixed(4) },
                    { label: 'Weight Anomaly Index', value: result.details.weight_analysis_risk?.toFixed(4) },
                    { label: 'Gradient Similarity', value: result.details.gradient_similarity?.toFixed(4) },
                    { label: 'Natural Shortcut Sensitivity', value: result.details.natural_sensitivity?.toFixed(4) },
                  ].map(({ label, value }) => (
                    <div key={label} className="telemetry-row" style={{ marginBottom: '0.5rem' }}>
                      <p className="label" style={{ fontSize: '0.6rem', marginBottom: '0.2rem' }}>{label}</p>
                      <p style={{ fontSize: '1.1rem', fontWeight: 700 }}>{value ?? 'N/A'}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>


            {/* NEW: Forensic Analysis Breakdown */}
            {result.details.forensic_analysis && result.details.forensic_analysis.length > 0 && (
              <div className="card glass stagger-3" style={{ marginTop: '2.5rem', padding: '2.5rem' }}>
                <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '1.2rem', fontWeight: 800, marginBottom: '2rem' }}>
                  <Shield size={22} color="var(--accent)" />
                  Forensic Discovery Narrative
                </h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(400px, 1fr))', gap: '2rem' }}>
                  {result.details.forensic_analysis.map((item: any, idx: number) => (
                    <div key={idx} className="glass-hover" style={{ padding: '1.5rem', borderLeft: `4px solid ${item.severity === 'CRITICAL' ? 'var(--danger)' : item.severity === 'HIGH' ? 'var(--warning)' : 'var(--accent)'}`, background: 'rgba(255,255,255,0.02)', borderRadius: '0 12px 12px 0' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
                        <div>
                          <p style={{ fontSize: '0.65rem', fontWeight: 900, letterSpacing: '1px', color: 'var(--accent)' }}>{item.method.toUpperCase()} AUDIT</p>
                          <p style={{ fontSize: '0.8rem', fontWeight: 600, color: '#94a3b8' }}>{item.layer}</p>
                        </div>
                        <span className={`badge ${item.severity === 'CRITICAL' ? 'badge-danger' : 'badge-warning'}`} style={{ fontSize: '0.6rem' }}>{item.severity}</span>
                      </div>
                      <p style={{ fontSize: '0.9rem', lineHeight: 1.6, color: 'rgba(255,255,255,0.8)' }}>
                        {item.reasoning}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
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

        @keyframes verdict-pulse {
          0%   { box-shadow: 0 0 30px rgba(239,68,68,0.2); }
          50%  { box-shadow: 0 0 60px rgba(239,68,68,0.45); }
          100% { box-shadow: 0 0 30px rgba(239,68,68,0.2); }
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
