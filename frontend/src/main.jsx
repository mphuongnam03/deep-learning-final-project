import React from 'react';
import { createRoot } from 'react-dom/client';
import { Activity, BarChart3, Clock, Database, History, LogOut, ShieldCheck, UploadCloud, UserPlus } from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import './styles.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

function apiClient(token) {
  async function request(path, options = {}) {
    const headers = new Headers(options.headers || {});
    if (token) headers.set('Authorization', `Bearer ${token}`);
    const response = await fetch(`${API_URL}${path}`, { ...options, headers });
    const text = await response.text();
    const data = text ? JSON.parse(text) : null;
    if (!response.ok) {
      throw new Error(data?.detail || 'Request failed');
    }
    return data;
  }
  return { request };
}

function App() {
  const [token, setToken] = React.useState(() => localStorage.getItem('tb_token'));
  const [user, setUser] = React.useState(() => JSON.parse(localStorage.getItem('tb_user') || 'null'));
  const [view, setView] = React.useState('dashboard');

  const saveSession = (payload) => {
    localStorage.setItem('tb_token', payload.access_token);
    localStorage.setItem('tb_user', JSON.stringify(payload.user));
    setToken(payload.access_token);
    setUser(payload.user);
  };

  const logout = () => {
    localStorage.removeItem('tb_token');
    localStorage.removeItem('tb_user');
    setToken(null);
    setUser(null);
    setView('dashboard');
  };

  if (!token || !user) {
    return <AuthScreen onAuth={saveSession} />;
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Activity size={24} /></div>
          <div>
            <strong>TB AI</strong>
            <span>Diagnosis Console</span>
          </div>
        </div>
        <nav>
          <NavButton icon={<UploadCloud />} label="Diagnosis" active={view === 'dashboard'} onClick={() => setView('dashboard')} />
          <NavButton icon={<History />} label="History" active={view === 'history'} onClick={() => setView('history')} />
          <NavButton icon={<BarChart3 />} label="Analytics" active={view === 'analytics'} onClick={() => setView('analytics')} />
          <NavButton icon={<Database />} label="Training" active={view === 'training'} onClick={() => setView('training')} />
        </nav>
        <div className="user-panel">
          <span>{user.full_name}</span>
          <small>{user.email}</small>
          <button className="ghost-button" onClick={logout}><LogOut size={16} /> Logout</button>
        </div>
      </aside>
      <main className="content">
        {view === 'dashboard' && <Dashboard token={token} />}
        {view === 'history' && <HistoryPage token={token} />}
        {view === 'analytics' && <AnalyticsPage token={token} />}
        {view === 'training' && <TrainingPage token={token} />}
      </main>
    </div>
  );
}

function NavButton({ icon, label, active, onClick }) {
  return <button className={`nav-button ${active ? 'active' : ''}`} onClick={onClick}>{React.cloneElement(icon, { size: 18 })}{label}</button>;
}

function AuthScreen({ onAuth }) {
  const [mode, setMode] = React.useState('login');
  const [email, setEmail] = React.useState('');
  const [fullName, setFullName] = React.useState('');
  const [password, setPassword] = React.useState('');
  const [error, setError] = React.useState('');
  const [loading, setLoading] = React.useState(false);

  const submit = async (event) => {
    event.preventDefault();
    setError('');
    setLoading(true);
    try {
      const body = mode === 'register' ? { email, full_name: fullName, password } : { email, password };
      const payload = await apiClient().request(`/auth/${mode}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      onAuth(payload);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-screen">
      <section className="auth-hero">
        <div className="hero-copy">
          <div className="pill"><ShieldCheck size={16} /> Secure AI workflow</div>
          <h1>TB AI Diagnosis Console</h1>
          <p>Upload chest X-ray images, run the two-stage AI pipeline, review annotated results, and track predictions in PostgreSQL.</p>
        </div>
      </section>
      <form className="auth-card" onSubmit={submit}>
        <UserPlus size={28} />
        <h2>{mode === 'login' ? 'Sign in' : 'Create account'}</h2>
        {mode === 'register' && <input value={fullName} onChange={(e) => setFullName(e.target.value)} placeholder="Full name" required />}
        <input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="Email" type="email" required />
        <input value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Password" type="password" minLength={8} required />
        {error && <div className="error">{error}</div>}
        <button className="primary-button" disabled={loading}>{loading ? 'Please wait...' : mode === 'login' ? 'Login' : 'Register'}</button>
        <button type="button" className="link-button" onClick={() => setMode(mode === 'login' ? 'register' : 'login')}>
          {mode === 'login' ? 'Need an account? Register' : 'Already have an account? Login'}
        </button>
      </form>
    </div>
  );
}

function Dashboard({ token }) {
  const [file, setFile] = React.useState(null);
  const [preview, setPreview] = React.useState(null);
  const [threshold, setThreshold] = React.useState(0.25);
  const [result, setResult] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState('');

  const onFile = (event) => {
    const selected = event.target.files?.[0];
    setFile(selected || null);
    setResult(null);
    setError('');
    if (selected) setPreview(URL.createObjectURL(selected));
  };

  const submit = async () => {
    if (!file) return;
    setLoading(true);
    setError('');
    try {
      const form = new FormData();
      form.append('file', file);
      const payload = await apiClient(token).request(`/predict?conf_threshold=${threshold}`, { method: 'POST', body: form });
      setResult(payload);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Page title="AI Diagnosis" subtitle="Run detection-first TB analysis on a chest X-ray image.">
      <div className="grid two">
        <section className="panel">
          <h3>Upload X-ray</h3>
          <label className="dropzone">
            <UploadCloud size={28} />
            <span>{file ? file.name : 'Choose PNG/JPG image'}</span>
            <input type="file" accept="image/*" onChange={onFile} />
          </label>
          <label className="slider-label">Detection confidence: {threshold.toFixed(2)}</label>
          <input type="range" min="0.05" max="0.95" step="0.05" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} />
          <button className="primary-button" onClick={submit} disabled={!file || loading}>{loading ? 'Analyzing...' : 'Analyze image'}</button>
          {error && <div className="error">{error}</div>}
          {preview && <img className="image-preview" src={preview} alt="Original preview" />}
        </section>
        <section className="panel">
          <h3>Annotated Result</h3>
          {result?.annotated_image_base64 ? (
            <img className="image-preview" src={`data:image/jpeg;base64,${result.annotated_image_base64}`} alt="Annotated result" />
          ) : (
            <div className="empty-state">Result image will appear here.</div>
          )}
        </section>
      </div>
      {result && <PredictionDetails result={result} />}
    </Page>
  );
}

function PredictionDetails({ result }) {
  return (
    <div className="result-stack">
      <div className="stats-row">
        <Stat label="Diagnosis" value={result.predicted_class} />
        <Stat label="Confidence" value={`${(result.confidence * 100).toFixed(1)}%`} />
        <Stat label="Kept boxes" value={result.kept_detection_count} />
        <Stat label="Time" value={`${result.processing_time_ms.toFixed(0)} ms`} />
      </div>
      <section className="panel">
        <h3>Class probabilities</h3>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={result.probabilities}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="class_name" />
            <YAxis tickFormatter={(v) => `${Math.round(v * 100)}%`} />
            <Tooltip formatter={(v) => `${(v * 100).toFixed(1)}%`} />
            <Bar dataKey="probability" fill="#2563eb" radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </section>
      <section className="panel">
        <h3>Bounding boxes</h3>
        <DataTable rows={result.boxes} />
      </section>
    </div>
  );
}

function HistoryPage({ token }) {
  const { data, error, loading, refresh } = useApi(token, '/predictions');
  return (
    <Page title="Prediction History" subtitle="Review saved diagnoses from PostgreSQL.">
      <button className="secondary-button" onClick={refresh}>Refresh</button>
      {loading && <div className="empty-state">Loading history...</div>}
      {error && <div className="error">{error}</div>}
      <div className="history-list">
        {(data || []).map((item) => (
          <article className="history-item" key={item.id}>
            <div>
              <strong>{item.filename}</strong>
              <span>{new Date(item.created_at).toLocaleString()}</span>
            </div>
            <div className="badge">{item.predicted_class}</div>
            <div>{(item.confidence * 100).toFixed(1)}%</div>
            <div><Clock size={14} /> {item.processing_time_ms.toFixed(0)} ms</div>
          </article>
        ))}
      </div>
    </Page>
  );
}

function AnalyticsPage({ token }) {
  const { data, error, loading } = useApi(token, '/analytics/dataset');
  const classRows = Object.entries(data?.class_distribution || {}).map(([name, count]) => ({ name, count }));
  const bboxRows = Object.entries(data?.bbox_distribution || {}).map(([name, count]) => ({ name, count }));
  return (
    <Page title="Dataset Analytics" subtitle="Summary generated from tbx11k-simplified/data.csv.">
      {loading && <div className="empty-state">Loading analytics...</div>}
      {error && <div className="error">{error}</div>}
      {data && (
        <>
          <div className="stats-row">
            <Stat label="Rows" value={data.total_rows} />
            <Stat label="Columns" value={data.columns.length} />
            <Stat label="Classes" value={Object.keys(data.class_distribution).length} />
          </div>
          <div className="grid two">
            <ChartPanel title="Four-class distribution" rows={classRows} />
            <ChartPanel title="BBox availability" rows={bboxRows} />
          </div>
        </>
      )}
    </Page>
  );
}

function TrainingPage({ token }) {
  const { data, error, loading, refresh } = useApi(token, '/training-metrics');
  const [importStatus, setImportStatus] = React.useState('');
  const detection = (data || []).filter((item) => item.model_type === 'detection');
  const classification = (data || []).filter((item) => item.model_type === 'classification');
  const importMetrics = async () => {
    setImportStatus('Importing metrics...');
    try {
      const payload = await apiClient(token).request('/training-metrics/import', { method: 'POST' });
      setImportStatus(`Imported ${payload.inserted} rows`);
      refresh();
    } catch (err) {
      setImportStatus(err.message);
    }
  };
  return (
    <Page title="Training Metrics" subtitle="Visualize imported YOLO result CSV metrics.">
      <div className="action-row">
        <button className="secondary-button" onClick={refresh}>Refresh</button>
        <button className="secondary-button" onClick={importMetrics}>Import metrics</button>
        {importStatus && <span>{importStatus}</span>}
      </div>
      {loading && <div className="empty-state">Loading metrics...</div>}
      {error && <div className="error">{error}</div>}
      <div className="grid two">
        <LinePanel title="Detection mAP/Recall" rows={detection} lines={['precision', 'recall', 'map50', 'map50_95']} />
        <LinePanel title="Classification accuracy" rows={classification} lines={['accuracy_top1', 'accuracy_top5']} />
      </div>
    </Page>
  );
}

function ChartPanel({ title, rows }) {
  return (
    <section className="panel">
      <h3>{title}</h3>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={rows}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="count" fill="#0f766e" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </section>
  );
}

function LinePanel({ title, rows, lines }) {
  return (
    <section className="panel">
      <h3>{title}</h3>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={rows}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="epoch" />
          <YAxis />
          <Tooltip />
          <Legend />
          {lines.map((line, index) => <Line key={line} dataKey={line} stroke={['#2563eb', '#dc2626', '#0f766e', '#f59e0b'][index]} dot={false} />)}
        </LineChart>
      </ResponsiveContainer>
    </section>
  );
}

function DataTable({ rows }) {
  if (!rows?.length) return <div className="empty-state">No bounding boxes returned.</div>;
  return (
    <table>
      <thead><tr><th>Status</th><th>ROI</th><th>ROI Conf</th><th>Detection</th><th>Det Conf</th><th>BBox</th><th>Reason</th></tr></thead>
      <tbody>
        {rows.map((row, index) => (
          <tr key={index}>
            <td>{row.kept ? 'Kept' : 'Dropped'}</td>
            <td>{row.roi_class || '-'}</td>
            <td>{row.roi_conf == null ? '-' : `${(row.roi_conf * 100).toFixed(1)}%`}</td>
            <td>{row.det_class}</td>
            <td>{(row.det_conf * 100).toFixed(1)}%</td>
            <td>{row.bbox.join(', ')}</td>
            <td>{row.reason || '-'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Page({ title, subtitle, children }) {
  return (
    <>
      <header className="page-header">
        <div>
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
      </header>
      {children}
    </>
  );
}

function Stat({ label, value }) {
  return <div className="stat"><span>{label}</span><strong>{value}</strong></div>;
}

function useApi(token, path) {
  const [data, setData] = React.useState(null);
  const [error, setError] = React.useState('');
  const [loading, setLoading] = React.useState(true);
  const refresh = React.useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      setData(await apiClient(token).request(path));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [token, path]);
  React.useEffect(() => { refresh(); }, [refresh]);
  return { data, error, loading, refresh };
}

createRoot(document.getElementById('root')).render(<App />);
