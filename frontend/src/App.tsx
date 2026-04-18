import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import Admin from './components/Admin';

const App: React.FC = () => {
  return (
    <Router>
      <div style={{ padding: '1rem', background: '#ffffff', borderBottom: '1px solid var(--surface-border)', display: 'flex', justifyContent: 'flex-end', gap: '1rem' }}>
        <Link to="/" style={{ textDecoration: 'none', color: 'var(--text-color)', fontWeight: 500 }}>Assistant</Link>
        <Link to="/admin" style={{ textDecoration: 'none', color: 'var(--accent-color)', fontWeight: 500 }}>Admin Diagnostics</Link>
      </div>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/admin" element={<Admin />} />
      </Routes>
    </Router>
  );
};

export default App;
