import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Search, ShieldAlert, Sparkles, BookOpen, Mic, MicOff, UploadCloud, History, CheckCircle2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// Types for TypeScript Web Speech API
declare global {
  interface Window {
    SpeechRecognition: any;
    webkitSpeechRecognition: any;
  }
}

interface Chunk {
  id: string;
  text: string;
  metadata: {
    title: string;
    authors: string;
    year: string;
    journal: string;
    source: string;
  };
  rerank_score?: number;
}

interface QueryResponse {
  answer: string;
  chunks: Chunk[];
}

const Dashboard: React.FC = () => {
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState('');

  // New States for dictation and upload
  const [isListening, setIsListening] = useState(false);
  const [searchHistory, setSearchHistory] = useState<string[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState('');
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Initialize Speech Recognition
  const initSpeechRecognition = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Your browser does not support Speech Recognition. Please use Chrome or Edge.");
      return null;
    }
    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      setIsListening(true);
    };

    recognition.onresult = (event: any) => {
      const current = event.resultIndex;
      const transcript = event.results[current][0].transcript;
      setQuery(transcript);
    };

    recognition.onerror = (event: any) => {
      console.error(event.error);
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
    };
    return recognition;
  };

  const toggleListen = () => {
    if (isListening) {
      setIsListening(false); // Can't explicitly stop native API easily without global ref, relying on end event or timeout
    } else {
      const recognition = initSpeechRecognition();
      if (recognition) recognition.start();
    }
  };

  useEffect(() => {
    const hist = localStorage.getItem('medlens_history');
    if (hist) {
      setSearchHistory(JSON.parse(hist));
    }
  }, []);

  const saveToHistory = (q: string) => {
    const updated = [q, ...searchHistory.filter(item => item !== q)].slice(0, 5);
    setSearchHistory(updated);
    localStorage.setItem('medlens_history', JSON.stringify(updated));
  };

  const handleSearch = async (e: React.FormEvent, directQuery?: string) => {
    if (e) e.preventDefault();
    const q = directQuery || query;
    if (!q.trim()) return;

    setShowHistory(false);
    saveToHistory(q);
    setQuery(q);
    setIsLoading(true);
    setError('');

    try {
      setResult({ answer: '', chunks: [] });
      
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q })
      });

      if (!response.ok) {
        if (response.status === 503) {
          setError(`Backend is starting up. Please wait 30 seconds and try again.`);
        } else {
          setError(`Backend Error (${response.status})`);
        }
        setIsLoading(false);
        return;
      }

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let partialData = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        partialData += decoder.decode(value, { stream: true });
        
        // Split by SSE events
        const events = partialData.split('\n\n');
        // Keep the last partial event attached to buffer
        partialData = events.pop() || "";

        for (const event of events) {
          if (event.startsWith('data: ')) {
            const dataStr = event.substring(6);
            if (dataStr === '[DONE]') {
               setIsLoading(false);
               continue;
            }
            try {
              const data = JSON.parse(dataStr);
              if (data.type === 'metadata') {
                setResult(prev => ({ ...prev!, chunks: data.chunks }));
              } else if (data.type === 'token') {
                setResult(prev => ({ ...prev!, answer: prev!.answer + data.content }));
                // Remove primary loading state as soon as first token arrives so UI opens
                setIsLoading(false); 
              } else if (data.type === 'error') {
                setError(data.content);
                setIsLoading(false);
              }
            } catch (e) {
              console.error("Failed to parse SSE JSON", e);
            }
          }
        }
      }
    } catch (err: any) {
      console.error(err);
      setError(`Error: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setError('');
    setUploadSuccess('');
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await axios.post('/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setUploadSuccess(res.data.message);
      // Auto clear success after 5 seconds
      setTimeout(() => setUploadSuccess(''), 5000);
    } catch (err: any) {
      const detail = err.response?.data?.detail || 'Upload failed';
      setError(`Upload Error: ${detail}`);
    } finally {
      setIsUploading(false);
      // reset file input
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  // Close history when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (searchInputRef.current && !searchInputRef.current.contains(e.target as Node)) {
        setShowHistory(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="container">
      {/* Safety Disclaimer Banner */}
      <div className="disclaimer-banner">
        <ShieldAlert size={20} />
        <span>This system provides research summaries, not medical advice. Always consult a healthcare professional for diagnosis and treatment.</span>
      </div>

      {/* Hero Section */}
      <div className="hero">
        <h1>Medical Research Assistant</h1>
        <p>Evidence-based answers grounded in verified medical literature.</p>
      </div>

      {/* Search Input */}
      <form onSubmit={(e) => handleSearch(e)} className="search-container" ref={searchInputRef}>
        <Search className="search-icon" size={24} />
        <input 
          type="text" 
          className="search-input" 
          placeholder="Ask a medical research question..." 
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => setShowHistory(true)}
          disabled={isLoading}
        />
        <button type="button" onClick={toggleListen} className={`mic-btn ${isListening ? 'listening' : ''}`} title="Dictate Query">
          {isListening ? <MicOff size={20} /> : <Mic size={20} />}
        </button>
        <button type="submit" className="submit-btn" disabled={isLoading || !query.trim()}>
          {isLoading ? 'Searching...' : 'Search'}
        </button>

        {/* History Dropdown */}
        <AnimatePresence>
          {showHistory && searchHistory.length > 0 && !isLoading && (
            <motion.div 
              initial={{ opacity: 0, y: -10 }} 
              animate={{ opacity: 1, y: 0 }} 
              exit={{ opacity: 0, y: -10 }}
              className="history-dropdown"
            >
              <div style={{ padding: '0.5rem 1rem', fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600, borderBottom: '1px solid var(--surface-border)' }}>Recent Searches</div>
              {searchHistory.map((item, idx) => (
                <div key={idx} className="history-item" onClick={() => handleSearch(undefined as any, item)}>
                  <History className="history-icon" />
                  <span>{item}</span>
                </div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </form>

      {/* Upload Document Section */}
      <div className="upload-container">
        <input type="file" accept="application/pdf" style={{ display: 'none' }} ref={fileInputRef} onChange={handleFileUpload} />
        <button className="upload-label" onClick={() => fileInputRef.current?.click()} disabled={isUploading}>
          {isUploading ? <div className="loading-spinner" style={{ width: '1.2rem', height: '1.2rem', borderWidth: '2px' }} /> : <UploadCloud size={20}/>}
          {isUploading ? 'Ingesting Document...' : 'Upload Source Document (PDF)'}
        </button>
        {uploadSuccess && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ color: 'var(--success-color)', display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 500 }}>
            <CheckCircle2 size={18} /> {uploadSuccess}
          </motion.div>
        )}
      </div>

      {/* Content Area */}
      <AnimatePresence>
        {isLoading && (
          <motion.div 
            initial={{ opacity: 0 }} 
            animate={{ opacity: 1 }} 
            exit={{ opacity: 0 }}
            className="status-message"
          >
            <div className="loading-spinner"></div>
            <p style={{ marginTop: '1rem' }}>Retrieving and synthesizing research...</p>
          </motion.div>
        )}

        {error && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="status-message" style={{ color: 'var(--danger-color)' }}>
            {error}
          </motion.div>
        )}

        {result && !isLoading && (
          <motion.div 
            className="dashboard-grid"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            {/* Left Column: Generated Answer */}
            <div className="glass-panel answer-section">
              <h2><Sparkles size={24} /> Synthesized Answer</h2>
              <div className="answer-content">
                {result.answer.split('\n').map((paragraph, i) => (
                  <p key={i}>{paragraph}</p>
                ))}
              </div>
            </div>

            {/* Right Column: Retrieved Chunks */}
            <div className="glass-panel chunks-section">
              <h3><BookOpen size={20} /> Retrieved Evidence</h3>
              {result.chunks.length === 0 ? (
                <p className="chunk-text" style={{ marginTop: '1rem' }}>No relevant chunks retrieved.</p>
              ) : (
                result.chunks.map((chunk, idx) => (
                  <motion.div 
                    key={chunk.id} 
                    className="chunk-card"
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.1 }}
                  >
                    <div className="chunk-meta">Source [{idx + 1}]</div>
                    <div className="chunk-title">{chunk.metadata.title}</div>
                    <div className="chunk-text" style={{ fontSize: '0.75rem', marginBottom: '0.5rem' }}>
                      {chunk.metadata.authors} ({chunk.metadata.year}) - {chunk.metadata.journal}
                    </div>
                    <div className="chunk-text">"{chunk.text}"</div>
                  </motion.div>
                ))
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Dashboard;
