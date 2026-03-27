import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API = 'http://localhost:5001/api';

export default function Overview() {
  const [data,    setData]    = useState(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState('');

  const load = async () => {
    setLoading(true); setError('');
    try {
      const r = await axios.get(`${API}/overview`);
      setData(r.data);
    } catch(e) {
      setError('Could not load overview. Is the Flask API running on port 5001?');
    }
    setLoading(false);
  };

  useEffect(() => { load(); }, []);

  const us     = data?.results.filter(r => !r.proxy_used) || [];
  const indian = data?.results.filter(r =>  r.proxy_used) || [];
  const upCount   = data?.results.filter(r => r.direction === 'UP').length   || 0;
  const downCount = data?.results.filter(r => r.direction === 'DOWN').length || 0;

  return (
    <div>
      <div style={styles.header}>
        <h1 style={styles.title}>All Stocks Overview</h1>
        <p style={styles.subtitle}>AI predictions for all 8 stocks — refreshed on load</p>
      </div>

      <button style={styles.refreshBtn} onClick={load} disabled={loading}>
        {loading ? 'Loading...' : '↺  Refresh Predictions'}
      </button>

      {error && <div style={styles.error}>{error}</div>}

      {data && (
        <>
          {/* Summary bar */}
          <div style={styles.summaryRow}>
            <SummaryCard label="Total Stocks" value={data.total}      color="var(--accent)" />
            <SummaryCard label="Trending UP"  value={upCount}         color="var(--up)"     />
            <SummaryCard label="Trending DOWN" value={downCount}      color="var(--down)"   />
            <SummaryCard label="Updated"
              value={new Date(data.timestamp).toLocaleTimeString()} color="var(--text2)" />
          </div>

          {/* US Stocks */}
          <SectionLabel>US Stocks — Direct Model</SectionLabel>
          <div style={styles.grid}>
            {us.map(r => <StockCard key={r.ticker} r={r} />)}
          </div>

          {/* Indian Stocks */}
          <SectionLabel style={{ marginTop: 36 }}>Indian Stocks — OOD Proxy Model</SectionLabel>
          <div style={styles.grid}>
            {indian.map(r => <StockCard key={r.ticker} r={r} indian />)}
          </div>
        </>
      )}
    </div>
  );
}

function StockCard({ r, indian }) {
  const isUp  = r.direction === 'UP';
  const isErr = r.direction === 'ERROR';
  const color = isErr ? 'var(--text3)' : isUp ? 'var(--up)' : 'var(--down)';

  return (
    <div style={{
      ...styles.card,
      borderColor: isErr ? 'var(--border)' : isUp ? '#00e5a030' : '#ff4d6d30',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div style={{ fontSize: 22, fontWeight: 800, letterSpacing: '-0.02em' }}>{r.ticker}</div>
        {indian && (
          <span className="tag tag-blue" style={{ fontSize: 10 }}>NSE</span>
        )}
      </div>

      {isErr ? (
        <div style={{ color: 'var(--down)', fontSize: 12, marginTop: 12, fontFamily: 'var(--font-mono)' }}>
          Error loading
        </div>
      ) : (
        <>
          <div style={{ fontSize: 40, fontWeight: 800, color, marginTop: 12, lineHeight: 1 }}>
            {isUp ? '▲' : '▼'} {r.direction}
          </div>
          <div style={{ marginTop: 16 }}>
            <ConfBar value={r.probability} isUp={isUp} />
          </div>
          <div style={{ fontSize: 12, color: 'var(--text3)', marginTop: 14,
            fontFamily: 'var(--font-mono)', display: 'flex', justifyContent: 'space-between' }}>
            <span>{r.probability.toFixed(1)}% conf.</span>
            {r.proxy_used && <span>via {r.model_used}</span>}
          </div>
        </>
      )}
    </div>
  );
}

function ConfBar({ value, isUp }) {
  const color = isUp ? 'var(--up)' : 'var(--down)';
  return (
    <div style={{ height: 5, background: 'var(--bg3)', borderRadius: 3, overflow: 'hidden' }}>
      <div style={{
        height: '100%', width: `${value}%`,
        background: color, borderRadius: 3,
      }} />
    </div>
  );
}

function SummaryCard({ label, value, color }) {
  return (
    <div style={styles.summaryCard}>
      <div style={{ fontSize: 28, fontWeight: 800, color }}>{value}</div>
      <div style={{ fontSize: 12, color: 'var(--text3)', fontFamily: 'var(--font-mono)', marginTop: 4 }}>{label}</div>
    </div>
  );
}

function SectionLabel({ children, style }) {
  return (
    <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text3)', letterSpacing: '0.1em',
      textTransform: 'uppercase', fontFamily: 'var(--font-mono)', marginBottom: 16, ...style }}>
      {children}
    </div>
  );
}

const styles = {
  header:     { marginBottom: 28 },
  title:      { fontSize: 32, fontWeight: 800, letterSpacing: '-0.03em', marginBottom: 8 },
  subtitle:   { color: 'var(--text2)', fontSize: 15 },
  refreshBtn: {
    background: 'var(--bg3)', border: '1px solid var(--border2)',
    color: 'var(--text2)', borderRadius: 10, padding: '10px 20px',
    fontSize: 14, fontWeight: 600, marginBottom: 28, cursor: 'pointer',
  },
  error: {
    background: '#ff4d6d15', border: '1px solid #ff4d6d40',
    borderRadius: 10, padding: '14px 18px', color: '#ff4d6d',
    fontSize: 14, marginBottom: 24, fontFamily: 'var(--font-mono)',
  },
  summaryRow: {
    display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)',
    gap: 16, marginBottom: 36,
  },
  summaryCard: {
    background: 'var(--bg2)', border: '1px solid var(--border)',
    borderRadius: 14, padding: '20px 24px',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
    gap: 16,
  },
  card: {
    background: 'var(--bg2)', border: '1px solid',
    borderRadius: 16, padding: '20px',
    transition: 'transform 0.15s',
  },
};
