import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Cell, ReferenceLine,
} from 'recharts';

const API = 'http://localhost:5001/api';

const COLORS = {
  AAPL:  '#3d8bff',
  MSFT:  '#00e5a0',
  GOOGL: '#f5c842',
  AMZN:  '#ff9f43',
  NVDA:  '#a855f7',
};

export default function Performance() {
  const [data,    setData]    = useState(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState('');

  useEffect(() => {
    axios.get(`${API}/performance`)
      .then(r => { setData(r.data); setLoading(false); })
      .catch(() => { setError('Could not load metrics.'); setLoading(false); });
  }, []);

  const metrics = data?.metrics || {};
  const tickers = Object.keys(metrics);

  const aucData = tickers.map(t => ({
    name: t, auc: metrics[t].auc, fill: COLORS[t],
  }));

  const radarData = tickers.map(t => ({
    stock: t,
    AUC:       Math.round(metrics[t].auc      * 100),
    Accuracy:  Math.round(metrics[t].accuracy * 100),
  }));

  return (
    <div>
      <div style={styles.header}>
        <h1 style={styles.title}>Model Performance</h1>
        <p style={styles.subtitle}>Phase 3 evaluation results from test set — all 5 US stocks</p>
      </div>

      {error   && <div style={styles.error}>{error}</div>}
      {loading && <div style={styles.loading}>Loading metrics...</div>}

      {data && (
        <>
          {/* Metric cards */}
          <div style={styles.metricGrid}>
            {tickers.map(t => (
              <MetricCard key={t} ticker={t} m={metrics[t]} color={COLORS[t]} />
            ))}
          </div>

          {/* AUC bar chart */}
          <div style={{ marginTop: 36 }}>
            <SectionLabel>AUC Score Comparison</SectionLabel>
            <div className="card">
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={aucData} barSize={50}>
                  <CartesianGrid stroke="#1e2d42" strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fill: '#7a92b0', fontSize: 13, fontFamily: 'DM Mono' }} />
                  <YAxis domain={[0.4, 0.65]} tick={{ fill: '#4a6180', fontSize: 11, fontFamily: 'DM Mono' }} />
                  <Tooltip contentStyle={tooltipStyle}
                    formatter={v => [v.toFixed(3), 'AUC']} />
                  <ReferenceLine y={0.5} stroke="#ffffff30" strokeDasharray="4 2"
                    label={{ value: 'Random (0.5)', fill: '#4a6180', fontSize: 11, fontFamily: 'DM Mono' }} />
                  <Bar dataKey="auc" radius={[6,6,0,0]}
                    label={{ position: 'top', fill: '#7a92b0', fontSize: 12,
                      fontFamily: 'DM Mono', formatter: v => v.toFixed(3) }}>
                    {aucData.map((entry, i) => (
                      <Cell key={i} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Confusion matrix explainer */}
          <div style={{ marginTop: 36 }}>
            <SectionLabel>AAPL Test Confusion Matrix</SectionLabel>
            <div style={styles.confGrid}>
              <div className="card" style={{ flex: 1 }}>
                <ConfusionMatrix />
              </div>
              <div style={styles.confExplain}>
                <ExplainRow label="True Positives"  value="106" desc="Correctly predicted UP"   color="var(--up)"     />
                <ExplainRow label="True Negatives"  value="64"  desc="Correctly predicted DOWN" color="var(--up)"     />
                <ExplainRow label="False Positives" value="83"  desc="Predicted UP, was DOWN"   color="var(--down)"   />
                <ExplainRow label="False Negatives" value="87"  desc="Predicted DOWN, was UP"   color="var(--down)"   />
                <div style={styles.accuracyBox}>
                  <span style={{ color: 'var(--text2)', fontSize: 13 }}>Test Accuracy</span>
                  <span style={{ color: 'var(--accent)', fontWeight: 800, fontSize: 22, fontFamily: 'var(--font-mono)' }}>50.0%</span>
                </div>
              </div>
            </div>
          </div>

          {/* Key findings */}
          <div style={{ marginTop: 36 }}>
            <SectionLabel>Key Findings</SectionLabel>
            <div style={styles.findingsGrid}>
              <Finding icon="📈" title="NVDA Best AUC" text="NVDA achieved 0.566 AUC — highest of all stocks, likely due to high volatility creating stronger technical patterns during the training period." />
              <Finding icon="🎯" title="~50% Accuracy" text="All models converge near 50% test accuracy, consistent with the Efficient Market Hypothesis — prices reflect all available information." />
              <Finding icon="⚡" title="Overfitting" text="Training curves show clear overfitting: train accuracy reaches 61% while validation stays at ~50%, confirming financial time series are hard to generalise." />
              <Finding icon="🌏" title="OOD Transfer" text="Indian stock predictions via proxy models produce near-random outputs, validating that market-specific factors dominate over technical cross-market patterns." />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function MetricCard({ ticker, m, color }) {
  return (
    <div style={{ ...styles.mCard, borderTop: `3px solid ${color}` }}>
      <div style={{ fontSize: 20, fontWeight: 800, marginBottom: 16 }}>{ticker}</div>
      <Meter label="AUC"      value={m.auc}      max={1}   color={color} format={v => v.toFixed(3)} />
      <Meter label="Accuracy" value={m.accuracy} max={1}   color={color} format={v => `${(v*100).toFixed(0)}%`} />
    </div>
  );
}

function Meter({ label, value, max, color, format }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
        <span style={{ fontSize: 12, color: 'var(--text3)', fontFamily: 'var(--font-mono)' }}>{label}</span>
        <span style={{ fontSize: 12, fontWeight: 700, color, fontFamily: 'var(--font-mono)' }}>{format(value)}</span>
      </div>
      <div style={{ height: 5, background: 'var(--bg3)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${(value/max)*100}%`, background: color, borderRadius: 3 }} />
      </div>
    </div>
  );
}

function ConfusionMatrix() {
  const cells = [
    { label: 'TN', value: 64,  desc: 'Actual DOWN\nPred DOWN', bg: '#3d8bff20', border: '#3d8bff40' },
    { label: 'FP', value: 83,  desc: 'Actual DOWN\nPred UP',   bg: '#ff4d6d15', border: '#ff4d6d30' },
    { label: 'FN', value: 87,  desc: 'Actual UP\nPred DOWN',   bg: '#ff4d6d15', border: '#ff4d6d30' },
    { label: 'TP', value: 106, desc: 'Actual UP\nPred UP',     bg: '#00e5a020', border: '#00e5a040' },
  ];
  return (
    <div>
      <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 16, color: 'var(--text2)' }}>
        AAPL — Test Confusion Matrix
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        {cells.map(c => (
          <div key={c.label} style={{
            background: c.bg, border: `1px solid ${c.border}`,
            borderRadius: 10, padding: '18px',
            textAlign: 'center',
          }}>
            <div style={{ fontSize: 28, fontWeight: 800 }}>{c.value}</div>
            <div style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'var(--font-mono)',
              marginTop: 6, whiteSpace: 'pre-line', lineHeight: 1.5 }}>{c.desc}</div>
          </div>
        ))}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between',
        marginTop: 10, fontSize: 11, color: 'var(--text3)', fontFamily: 'var(--font-mono)' }}>
        <span>← Predicted DOWN</span>
        <span>Predicted UP →</span>
      </div>
    </div>
  );
}

function ExplainRow({ label, value, desc, color }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 14 }}>
      <div style={{ fontSize: 22, fontWeight: 800, color, width: 44,
        fontFamily: 'var(--font-mono)', flexShrink: 0, textAlign: 'right' }}>{value}</div>
      <div>
        <div style={{ fontSize: 13, fontWeight: 700 }}>{label}</div>
        <div style={{ fontSize: 12, color: 'var(--text3)' }}>{desc}</div>
      </div>
    </div>
  );
}

function Finding({ icon, title, text }) {
  return (
    <div style={styles.finding}>
      <div style={{ fontSize: 24, marginBottom: 10 }}>{icon}</div>
      <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 8 }}>{title}</div>
      <div style={{ fontSize: 13, color: 'var(--text2)', lineHeight: 1.7 }}>{text}</div>
    </div>
  );
}

function SectionLabel({ children }) {
  return (
    <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text3)', letterSpacing: '0.1em',
      textTransform: 'uppercase', fontFamily: 'var(--font-mono)', marginBottom: 14 }}>
      {children}
    </div>
  );
}

const tooltipStyle = {
  background: '#0d1420', border: '1px solid #1e2d42',
  borderRadius: 8, color: '#e8edf5', fontSize: 12, fontFamily: 'DM Mono',
};

const styles = {
  header:   { marginBottom: 28 },
  title:    { fontSize: 32, fontWeight: 800, letterSpacing: '-0.03em', marginBottom: 8 },
  subtitle: { color: 'var(--text2)', fontSize: 15 },
  error:    { background: '#ff4d6d15', border: '1px solid #ff4d6d40', borderRadius: 10,
    padding: '14px 18px', color: '#ff4d6d', fontSize: 14, marginBottom: 24 },
  loading:  { color: 'var(--text2)', fontFamily: 'var(--font-mono)', padding: '40px 0' },
  metricGrid: { display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 14 },
  mCard: {
    background: 'var(--bg2)', border: '1px solid var(--border)',
    borderRadius: 14, padding: '20px',
  },
  confGrid: { display: 'flex', gap: 24, alignItems: 'flex-start' },
  confExplain: { flex: 1, paddingTop: 4 },
  accuracyBox: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    background: 'var(--bg3)', border: '1px solid var(--border)',
    borderRadius: 10, padding: '14px 18px', marginTop: 8,
  },
  findingsGrid: { display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 },
  finding: {
    background: 'var(--bg2)', border: '1px solid var(--border)',
    borderRadius: 14, padding: '22px',
  },
};
