import React, { useState } from 'react';
import axios from 'axios';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, LineChart, Line,
} from 'recharts';

const API = 'http://localhost:5001/api';
const QUICK = ['AAPL','MSFT','GOOGL','AMZN','NVDA','TCS','RELIANCE','HDFCBANK'];

export default function Predict() {
  const [ticker,  setTicker]  = useState('');
  const [loading, setLoading] = useState(false);
  const [result,  setResult]  = useState(null);
  const [chart,   setChart]   = useState(null);
  const [error,   setError]   = useState('');

  const run = async (t) => {
    const sym = (t || ticker).toUpperCase().trim();
    if (!sym) return;
    setLoading(true); setError(''); setResult(null); setChart(null);
    try {
      const [pred, ch] = await Promise.all([
        axios.get(`${API}/predict?ticker=${sym}`),
        axios.get(`${API}/chart?ticker=${sym}`),
      ]);
      setResult(pred.data);
      // build chart data
      const d = ch.data;
      const rows = d.dates.map((date, i) => ({
        date: date.slice(5),   // MM-DD
        close:    d.close[i],
        sma20:    d.sma20[i],
        bb_upper: d.bb_upper[i],
        bb_lower: d.bb_lower[i],
        rsi:      d.rsi[i],
        volume:   d.volume[i],
      })).filter(r => r.close !== null);
      setChart(rows);
    } catch(e) {
      setError(e.response?.data?.error || 'Prediction failed. Is the Flask API running?');
    }
    setLoading(false);
  };

  const isUp = result?.direction === 'UP';

  return (
    <div>
      <div style={styles.header}>
        <h1 style={styles.title}>Live Prediction</h1>
        <p style={styles.subtitle}>Enter any stock ticker to get an AI-powered trend forecast</p>
      </div>

      {/* Search bar */}
      <div style={styles.searchWrap}>
        <input
          style={styles.input}
          placeholder="e.g. AAPL, NVDA, TCS ..."
          value={ticker}
          onChange={e => setTicker(e.target.value.toUpperCase())}
          onKeyDown={e => e.key === 'Enter' && run()}
        />
        <button style={styles.btn} onClick={() => run()} disabled={loading}>
          {loading ? '...' : 'Predict →'}
        </button>
      </div>

      {/* Quick picks */}
      <div style={styles.quickRow}>
        {QUICK.map(t => (
          <button key={t} style={styles.chip}
            onClick={() => { setTicker(t); run(t); }}>
            {t}
          </button>
        ))}
      </div>

      {error && <div style={styles.error}>{error}</div>}

      {result && (
        <div style={{ animation: 'fadeIn 0.4s ease' }}>
          {/* Result card */}
          <div style={{
            ...styles.resultCard,
            borderColor: isUp ? '#00e5a040' : '#ff4d6d40',
            background: isUp
              ? 'linear-gradient(135deg, #080c14 60%, #00e5a008)'
              : 'linear-gradient(135deg, #080c14 60%, #ff4d6d08)',
          }}>
            <div style={styles.resultLeft}>
              <div style={styles.tickerLabel}>{result.ticker}</div>
              {result.proxy_used && (
                <span className="tag tag-blue" style={{ marginBottom: 12 }}>
                  Proxy: {result.model_used}
                </span>
              )}
              <div style={{
                fontSize: 80, fontWeight: 800, lineHeight: 1,
                color: isUp ? 'var(--up)' : 'var(--down)',
                letterSpacing: '-0.04em',
              }}>
                {isUp ? '▲' : '▼'}
              </div>
              <div style={{
                fontSize: 36, fontWeight: 800,
                color: isUp ? 'var(--up)' : 'var(--down)',
                marginTop: 4,
              }}>
                {result.direction}
              </div>
              <div style={styles.confText}>
                {result.probability.toFixed(1)}% confidence
              </div>
            </div>

            {/* Confidence gauge */}
            <div style={styles.gaugeWrap}>
              <GaugeBar value={result.probability} isUp={isUp} />
              <div style={styles.gaugeDetails}>
                <Detail label="Model"    value={result.model_used} />
                <Detail label="Lookback" value="30 days" />
                <Detail label="Features" value="28" />
                <Detail label="Type"
                  value={result.proxy_used ? 'OOD Proxy' : 'Direct'} />
              </div>
            </div>
          </div>

          {/* Price chart */}
          {chart && (
            <div style={{ marginTop: 28 }}>
              <SectionLabel>30-Day Price Chart with Bollinger Bands</SectionLabel>
              <div className="card" style={{ padding: '24px 16px' }}>
                <ResponsiveContainer width="100%" height={260}>
                  <AreaChart data={chart}>
                    <defs>
                      <linearGradient id="closeGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%"  stopColor={isUp ? '#00e5a0' : '#ff4d6d'} stopOpacity={0.15}/>
                        <stop offset="95%" stopColor={isUp ? '#00e5a0' : '#ff4d6d'} stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke="#1e2d42" strokeDasharray="3 3" />
                    <XAxis dataKey="date" tick={{ fill: '#4a6180', fontSize: 11, fontFamily: 'DM Mono' }} />
                    <YAxis tick={{ fill: '#4a6180', fontSize: 11, fontFamily: 'DM Mono' }} domain={['auto','auto']} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Area  type="monotone" dataKey="close"    stroke={isUp ? '#00e5a0' : '#ff4d6d'} fill="url(#closeGrad)" strokeWidth={2} dot={false} name="Close" />
                    <Line type="monotone" dataKey="sma20"    stroke="#3d8bff" strokeWidth={1.5} dot={false} strokeDasharray="4 2" name="SMA 20" />
                    <Line type="monotone" dataKey="bb_upper" stroke="#f5c84260" strokeWidth={1} dot={false} name="BB Upper" />
                    <Line type="monotone" dataKey="bb_lower" stroke="#f5c84260" strokeWidth={1} dot={false} name="BB Lower" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              {/* RSI chart */}
              <SectionLabel style={{ marginTop: 24 }}>RSI (14)</SectionLabel>
              <div className="card" style={{ padding: '24px 16px' }}>
                <ResponsiveContainer width="100%" height={120}>
                  <LineChart data={chart}>
                    <CartesianGrid stroke="#1e2d42" strokeDasharray="3 3" />
                    <XAxis dataKey="date" tick={{ fill: '#4a6180', fontSize: 11, fontFamily: 'DM Mono' }} />
                    <YAxis domain={[0, 100]} tick={{ fill: '#4a6180', fontSize: 11, fontFamily: 'DM Mono' }} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <ReferenceLine y={70} stroke="#ff4d6d60" strokeDasharray="3 3" label={{ value: 'OB', fill: '#ff4d6d80', fontSize: 10 }} />
                    <ReferenceLine y={30} stroke="#00e5a060" strokeDasharray="3 3" label={{ value: 'OS', fill: '#00e5a080', fontSize: 10 }} />
                    <Line type="monotone" dataKey="rsi" stroke="#f5c842" strokeWidth={2} dot={false} name="RSI" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      )}

      <style>{`
        @keyframes fadeIn { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
      `}</style>
    </div>
  );
}

function GaugeBar({ value, isUp }) {
  const color = isUp ? 'var(--up)' : 'var(--down)';
  return (
    <div style={{ width: '100%' }}>
      <div style={{ display:'flex', justifyContent:'space-between', marginBottom: 8 }}>
        <span style={{ fontSize: 12, color: 'var(--text3)', fontFamily: 'var(--font-mono)' }}>Confidence</span>
        <span style={{ fontSize: 14, fontWeight: 700, color, fontFamily: 'var(--font-mono)' }}>{value.toFixed(1)}%</span>
      </div>
      <div style={{ height: 8, background: 'var(--bg3)', borderRadius: 4, overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${value}%`,
          background: `linear-gradient(90deg, ${color}80, ${color})`,
          borderRadius: 4,
          transition: 'width 0.8s cubic-bezier(0.4,0,0.2,1)',
        }} />
      </div>
    </div>
  );
}

function Detail({ label, value }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 10 }}>
      <span style={{ fontSize: 12, color: 'var(--text3)', fontFamily: 'var(--font-mono)' }}>{label}</span>
      <span style={{ fontSize: 12, color: 'var(--text2)', fontFamily: 'var(--font-mono)' }}>{value}</span>
    </div>
  );
}

function SectionLabel({ children, style }) {
  return (
    <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text3)', letterSpacing: '0.1em',
      textTransform: 'uppercase', fontFamily: 'var(--font-mono)', marginBottom: 12, ...style }}>
      {children}
    </div>
  );
}

const tooltipStyle = {
  background: '#0d1420', border: '1px solid #1e2d42',
  borderRadius: 8, color: '#e8edf5', fontSize: 12, fontFamily: 'DM Mono',
};

const styles = {
  header:   { marginBottom: 32 },
  title:    { fontSize: 32, fontWeight: 800, letterSpacing: '-0.03em', marginBottom: 8 },
  subtitle: { color: 'var(--text2)', fontSize: 15 },
  searchWrap: { display: 'flex', gap: 12, marginBottom: 16 },
  input: {
    flex: 1, background: 'var(--bg2)', border: '1px solid var(--border2)',
    borderRadius: 10, padding: '14px 18px', fontSize: 16, color: 'var(--text)',
    letterSpacing: '0.05em',
  },
  btn: {
    background: 'var(--accent)', color: '#fff', borderRadius: 10,
    padding: '14px 28px', fontSize: 15, fontWeight: 700,
    transition: 'opacity 0.15s',
  },
  quickRow: { display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 32 },
  chip: {
    background: 'var(--bg3)', border: '1px solid var(--border)',
    color: 'var(--text2)', borderRadius: 8, padding: '6px 14px',
    fontSize: 13, fontFamily: 'var(--font-mono)', fontWeight: 500,
    cursor: 'pointer', transition: 'all 0.15s',
  },
  error: {
    background: '#ff4d6d15', border: '1px solid #ff4d6d40',
    borderRadius: 10, padding: '14px 18px', color: '#ff4d6d',
    fontSize: 14, marginBottom: 24, fontFamily: 'var(--font-mono)',
  },
  resultCard: {
    border: '1px solid', borderRadius: 20, padding: 32,
    display: 'flex', gap: 40, alignItems: 'flex-start',
  },
  resultLeft:  { flex: 1 },
  tickerLabel: { fontSize: 14, fontFamily: 'var(--font-mono)', color: 'var(--text3)', marginBottom: 8, letterSpacing: '0.1em' },
  confText:    { fontSize: 16, color: 'var(--text2)', marginTop: 8, fontFamily: 'var(--font-mono)' },
  gaugeWrap:   { width: 240, flexShrink: 0 },
  gaugeDetails:{ marginTop: 24, padding: '16px', background: 'var(--bg3)', borderRadius: 10, border: '1px solid var(--border)' },
};
