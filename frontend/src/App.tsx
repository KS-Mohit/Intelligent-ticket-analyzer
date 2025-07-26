import { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

// --- Type Definitions ---
interface ExtractedEntities {
  product: string | null;
  date: string | null;
  complaint_keywords: string[];
}

interface AnalysisResponse {
  issue_type: string;
  urgency_level: string;
  entities: ExtractedEntities;
}

// --- SVG Icon Components (optional: keep as in your project) ---
const IssueIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z" />
    <path d="m9 12 2 2 4-4" />
  </svg>
);
const UrgencyIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="m21.73 18-8-14a2 2 0 0 0-3.46 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
    <path d="M12 9v4" />
    <path d="M12 17h.01" />
  </svg>
);
const EntitiesIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z" />
    <path d="m15 5 4 4" />
  </svg>
);

const SunIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2" />
    <path d="M12 20v2" />
    <path d="m4.93 4.93 1.41 1.41" />
    <path d="m17.66 17.66 1.41 1.41" />
    <path d="M2 12h2" />
    <path d="M20 12h2" />
    <path d="m6.34 17.66-1.41 1.41" />
    <path d="m19.07 4.93-1.41 1.41" />
  </svg>
);
const MoonIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z" />
  </svg>
);

function App() {
  const [ticketText, setTicketText] = useState<string>('');
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [theme, setTheme] = useState<'light' | 'dark'>('dark');

  const toggleTheme = () => {
    setTheme(currentTheme => (currentTheme === 'dark' ? 'light' : 'dark'));
  };

  useEffect(() => {
    document.body.setAttribute('data-theme', theme);
  }, [theme]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    setAnalysis(null);

    try {
      const response = await axios.post<AnalysisResponse>('http://127.0.0.1:8000/analyze_ticket', {
        text: ticketText,
      });
      setAnalysis(response.data);
    } catch (err) {
      setError('Failed to connect to the API. Is the backend running?');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="main-bg">
      <button onClick={toggleTheme} className="theme-toggle">
        {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
      </button>
      <header className="main-header">
        <h1 className="main-title">Intelligent Ticket Analyzer</h1>
        <p className="main-subtitle">AI-powered classification and entity extraction for support tickets.</p>
      </header>
      <section className="analyzer-box">
        <form onSubmit={handleSubmit} className="form-container">
          <textarea
            value={ticketText}
            onChange={e => setTicketText(e.target.value)}
            placeholder="Paste your support ticket text here..."
            rows={7}
            required
          />
          <button type="submit" disabled={isLoading || !ticketText}>
            {isLoading ? 'Analyzing...' : 'Analyze Ticket'}
          </button>
        </form>
        {error && <p className="error-message">{error}</p>}
        {analysis && (
          <div className="results-grid">
            <div className="result-card">
              <div className="card-header">
                <IssueIcon />
                <h3>Issue Type</h3>
              </div>
              <p className="card-content">{analysis.issue_type}</p>
            </div>
            <div className="result-card">
              <div className="card-header">
                <UrgencyIcon />
                <h3>Urgency Level</h3>
              </div>
              <p className={`card-content urgency-${analysis.urgency_level.toLowerCase()}`}>{analysis.urgency_level}</p>
            </div>
            <div className="result-card full-width">
              <div className="card-header">
                <EntitiesIcon />
                <h3>Extracted Entities</h3>
              </div>
              <ul className="entity-list">
                <li><span>Product:</span> {analysis.entities.product || 'N/A'}</li>
                <li><span>Date:</span> {analysis.entities.date || 'N/A'}</li>
                <li>
                  <span>Keywords:</span>
                  {analysis.entities.complaint_keywords.length > 0
                    ? analysis.entities.complaint_keywords.join(', ')
                    : 'N/A'}
                </li>
              </ul>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

export default App;
