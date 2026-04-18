import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { ShieldAlert, BarChart3, Activity, Beaker, Database, Clock } from 'lucide-react';
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface EvalResult {
  context_precision: number;
  faithfulness: number;
  answer_relevancy: number;
  context_recall: number;
}

interface AnalyticsResult {
  total_queries: number;
  avg_latency_ms: number;
  distribution: { name: string; value: number }[];
  latency_history: { Session: number; Latency_ms: number }[];
}

const COLORS = ['#059669', '#dc2626', '#d97706', '#2563eb'];

const Admin: React.FC = () => {
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [results, setResults] = useState<EvalResult | null>(null);
  const [analytics, setAnalytics] = useState<AnalyticsResult | null>(null);
  const [error, setError] = useState('');

  const fetchAnalytics = async () => {
    try {
      const res = await axios.get('/api/analytics');
      setAnalytics(res.data);
    } catch (err) {
      console.error("Failed to load analytics");
    }
  };

  useEffect(() => {
    fetchAnalytics();
  }, []);

  const runEvaluation = async () => {
    setIsEvaluating(true);
    setError('');
    
    try {
      const res = await axios.post('/api/evaluate');
      setResults(res.data);
    } catch (err: any) {
      setError('Evaluation failed. Please check the backend console.');
    } finally {
      setIsEvaluating(false);
    }
  };

  const data = results ? [
    { name: 'Context Precision', score: results.context_precision * 100 },
    { name: 'Faithfulness', score: results.faithfulness * 100 },
    { name: 'Answer Relevancy', score: results.answer_relevancy * 100 },
    { name: 'Context Recall', score: results.context_recall * 100 },
  ] : [];

  return (
    <div className="container" style={{ maxWidth: '1000px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '2rem' }}>
        <Database size={32} color="var(--accent-color)" />
        <h1>Data Science & System Diagnostics</h1>
      </div>

      {/* Analytics Row */}
      {analytics && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginBottom: '2rem' }}>
          
          <div className="glass-panel" style={{ padding: '2rem' }}>
            <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem', fontSize: '1.25rem' }}>
              <Clock size={20} /> System Latency (ms)
            </h2>
            <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-color)', marginBottom: '1rem' }}>
              {analytics.avg_latency_ms} <span style={{ fontSize: '1rem', color: 'var(--text-muted)' }}>avg</span>
            </div>
            <div style={{ width: '100%', height: 200 }}>
              <ResponsiveContainer>
                <LineChart data={analytics.latency_history}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--surface-border)" />
                  <XAxis dataKey="Session" stroke="var(--text-color)" />
                  <YAxis stroke="var(--text-color)" />
                  <Tooltip contentStyle={{ backgroundColor: 'var(--bg-color)', borderColor: 'var(--surface-border)', color: 'var(--text-color)' }} />
                  <Line type="monotone" dataKey="Latency_ms" stroke="var(--accent-color)" strokeWidth={3} dot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="glass-panel" style={{ padding: '2rem' }}>
            <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem', fontSize: '1.25rem' }}>
              <BarChart3 size={20} /> Query Intent Distribution
            </h2>
            <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-color)', marginBottom: '1rem' }}>
              {analytics.total_queries} <span style={{ fontSize: '1rem', color: 'var(--text-muted)' }}>queries logged</span>
            </div>
            <div style={{ width: '100%', height: 200 }}>
              <ResponsiveContainer>
                <PieChart>
                  <Pie data={analytics.distribution} cx="50%" cy="50%" outerRadius={80} fill="#8884d8" dataKey="value" label>
                    {analytics.distribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ backgroundColor: 'var(--bg-color)', borderColor: 'var(--surface-border)', color: 'var(--text-color)' }} />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

        </div>
      )}

      {/* RAGAS Evaluation Row */}
      <div className="glass-panel" style={{ padding: '2rem', marginBottom: '2rem' }}>
        <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
          <Beaker size={24} /> RAGAS Evaluation Suite
        </h2>
        <p style={{ color: 'var(--text-muted)', marginBottom: '1.5rem', lineHeight: 1.6 }}>
          Run an automated LLM-as-a-judge diagnostic against the Golden Dataset. This measures context retrieval accuracy and prevents medical hallucinations.
        </p>

        <button 
          className="submit-btn" 
          style={{ position: 'relative', top: 0, right: 0, transform: 'none' }}
          onClick={runEvaluation} 
          disabled={isEvaluating}
        >
          {isEvaluating ? 'Running Evaluation (may take 60s)...' : 'Run System Diagnostic'}
        </button>
        {error && <p style={{ color: 'var(--danger-color)', marginTop: '1rem' }}>{error}</p>}
      </div>

      {results && (
        <div className="glass-panel" style={{ padding: '2rem', animation: 'fadeIn 0.5s ease' }}>
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '2rem' }}>
            <Activity size={24} /> RAGAS Quality Metrics (100-point scale)
          </h2>
          
          <div style={{ width: '100%', height: 300 }}>
            <ResponsiveContainer>
              <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--surface-border)" />
                <XAxis dataKey="name" stroke="var(--text-color)" />
                <YAxis stroke="var(--text-color)" domain={[0, 100]} />
                <Tooltip 
                  cursor={{fill: 'rgba(37, 99, 235, 0.1)'}}
                  contentStyle={{ backgroundColor: 'var(--bg-color)', borderColor: 'var(--surface-border)', color: 'var(--text-color)' }}
                />
                <Legend />
                <Bar dataKey="score" fill="var(--success-color)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
};

export default Admin;
