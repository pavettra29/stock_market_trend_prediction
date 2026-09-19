import React from 'react';
import { BrowserRouter, Routes, Route, NavLink, Navigate } from 'react-router-dom';
import './index.css';
import Predict     from './pages/Predict';
import Overview    from './pages/Overview';
import IndianOOD   from './pages/IndianOOD';
import Performance from './pages/Performance';
import Login       from './pages/Login';
import Signup      from './pages/Signup';
import { AuthProvider, useAuth } from './context/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';

const NAV = [
  { to: '/',            label: 'Live Predict' },
  { to: '/overview',   label: 'All Stocks'   },
  { to: '/indian',     label: 'Indian OOD'   },
  { to: '/performance',label: 'Model Stats'  },
];

function Layout({ children }) {
  const { user, logout } = useAuth();
  return (
    <div style={{ display: 'flex', minHeight: '100vh' }}>
      <aside style={styles.sidebar}>
        <div style={styles.logo}>
          <div style={styles.logoMark}>S</div>
          <div>
            <div style={styles.logoTitle}>StockSense</div>
            <div style={styles.logoSub}>AI Trend Predictor</div>
          </div>
        </div>

        <nav style={{ marginTop: 40 }}>
          {NAV.map(n => (
            <NavLink key={n.to} to={n.to} end={n.to === '/'}
              style={({ isActive }) => ({
                ...styles.navLink,
                ...(isActive ? styles.navLinkActive : {}),
              })}>
              {n.label}
            </NavLink>
          ))}
        </nav>

        <div style={styles.sidebarFooter}>
          {user && (
            <>
              <div style={styles.footerBadge}>Logged in as {user.username}</div>
              <button style={styles.logoutBtn} onClick={logout}>Log out</button>
            </>
          )}
          <div style={{ ...styles.footerText, marginTop: 10 }}>Parul University</div>
          <div style={styles.footerText}>Final Year Project · CSE</div>
          <div style={styles.footerText}>LSTM · 28 features · 8 stocks</div>
        </div>
      </aside>

      <main style={styles.main}>{children}</main>
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/*" element={
            <ProtectedRoute>
              <Layout>
                <Routes>
                  <Route path="/"             element={<Predict />}     />
                  <Route path="/overview"     element={<Overview />}    />
                  <Route path="/indian"       element={<IndianOOD />}   />
                  <Route path="/performance"  element={<Performance />} />
                  <Route path="*"             element={<Navigate to="/" replace />} />
                </Routes>
              </Layout>
            </ProtectedRoute>
          } />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}

const styles = {
  sidebar: {
    width: 230,
    minHeight: '100vh',
    background: 'var(--bg2)',
    borderRight: '1px solid var(--border)',
    padding: '32px 20px',
    display: 'flex',
    flexDirection: 'column',
    position: 'sticky',
    top: 0,
    height: '100vh',
    flexShrink: 0,
  },
  logo: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
  },
  logoMark: {
    width: 40, height: 40,
    background: 'linear-gradient(135deg, #3d8bff, #00e5a0)',
    borderRadius: 10,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontSize: 20, fontWeight: 800, color: '#fff',
    flexShrink: 0,
  },
  logoTitle: {
    fontSize: 18, fontWeight: 800,
    color: 'var(--text)',
    letterSpacing: '-0.02em',
  },
  logoSub: {
    fontSize: 11, color: 'var(--text3)',
    fontFamily: 'var(--font-mono)',
    marginTop: 1,
  },
  navLink: {
    display: 'block',
    padding: '11px 14px',
    borderRadius: 10,
    color: 'var(--text2)',
    textDecoration: 'none',
    fontSize: 14,
    fontWeight: 600,
    marginBottom: 4,
    transition: 'all 0.15s',
  },
  navLinkActive: {
    background: 'var(--accent-dim)',
    color: 'var(--accent)',
    borderLeft: '3px solid var(--accent)',
  },
  main: {
    flex: 1,
    padding: '40px 48px',
    overflowY: 'auto',
  },
  sidebarFooter: {
    marginTop: 'auto',
    padding: '16px 14px',
    background: 'var(--bg3)',
    borderRadius: 10,
    border: '1px solid var(--border)',
  },
  footerBadge: {
    fontSize: 12, fontWeight: 700,
    color: 'var(--gold)',
    marginBottom: 6,
  },
  footerText: {
    fontSize: 11,
    color: 'var(--text3)',
    fontFamily: 'var(--font-mono)',
    marginTop: 3,
  },
  logoutBtn: {
    fontSize: 11,
    padding: '4px 10px',
    borderRadius: 6,
    border: '1px solid var(--border)',
    background: 'transparent',
    color: 'var(--text2)',
    cursor: 'pointer',
  },
};
