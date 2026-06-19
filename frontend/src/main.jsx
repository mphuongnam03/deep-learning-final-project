import React from 'react';
import { createRoot } from 'react-dom/client';
import { Activity, ArrowLeft, BarChart3, Clock, Database, Edit, FileText, History, Image, LogOut, Plus, Search, ShieldCheck, Trash2, UploadCloud, UserPlus, Users } from 'lucide-react';
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
          <NavButton icon={<Users />} label="Patients" active={view === 'patients'} onClick={() => setView('patients')} />
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
        {view === 'patients' && <PatientsPage token={token} />}
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
      <div className="notice">
        Quick upload is not linked to a patient record. Use <strong>Patients</strong> to select a patient, upload X-ray images, and keep diagnoses in the medical history.
      </div>
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
      {result && <PredictionDetails result={result} token={token} />}
    </Page>
  );
}

function PredictionDetails({ result, token }) {
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
      <MedicalReportPanel predictionId={result.id} token={token} />
    </div>
  );
}

function PatientsPage({ token }) {
  const [search, setSearch] = React.useState('');
  const [patients, setPatients] = React.useState([]);
  const [selected, setSelected] = React.useState(null);
  const [editing, setEditing] = React.useState(null);
  const [showForm, setShowForm] = React.useState(false);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState('');

  const loadPatients = React.useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const qs = search.trim() ? `?search=${encodeURIComponent(search.trim())}` : '';
      setPatients(await apiClient(token).request(`/patients${qs}`));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [search, token]);

  React.useEffect(() => { loadPatients(); }, [loadPatients]);

  const savePatient = async (payload) => {
    const path = editing ? `/patients/${editing.id}` : '/patients';
    const method = editing ? 'PUT' : 'POST';
    const patient = await apiClient(token).request(path, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    setShowForm(false);
    setEditing(null);
    setSelected(patient);
    await loadPatients();
  };

  const removePatient = async (patient) => {
    if (!confirm(`Deactivate patient ${patient.full_name}?`)) return;
    const updated = await apiClient(token).request(`/patients/${patient.id}`, { method: 'DELETE' });
    if (selected?.id === patient.id) setSelected(updated);
    await loadPatients();
  };

  if (selected) {
    if (showForm) {
      return (
        <Page title="Edit Patient" subtitle={`${selected.patient_code} · ${selected.full_name}`}>
          <PatientForm
            initial={editing || selected}
            onCancel={() => { setShowForm(false); setEditing(null); }}
            onSave={savePatient}
          />
        </Page>
      );
    }
    return (
      <PatientDetail
        token={token}
        patient={selected}
        onBack={() => setSelected(null)}
        onEdit={() => { setEditing(selected); setShowForm(true); }}
      />
    );
  }

  return (
    <Page title="Patients" subtitle="Create patient profiles, manage X-ray studies, and run linked AI diagnoses.">
      <div className="toolbar">
        <div className="search-box"><Search size={16} /><input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search name, code, phone" /></div>
        <button className="secondary-button" onClick={() => { setEditing(null); setShowForm(true); }}><Plus size={16} /> Create Patient</button>
      </div>
      {error && <div className="error">{error}</div>}
      {loading && <div className="empty-state">Loading patients...</div>}
      {showForm && (
        <PatientForm
          initial={editing}
          onCancel={() => { setShowForm(false); setEditing(null); }}
          onSave={savePatient}
        />
      )}
      <div className="patient-grid">
        {patients.map((patient) => (
          <article className="patient-card" key={patient.id}>
            <div>
              <strong>{patient.full_name}</strong>
              <span>{patient.patient_code}</span>
            </div>
            <div className="patient-meta">
              <span>{patient.gender || 'Gender: -'}</span>
              <span>{patient.phone || 'Phone: -'}</span>
            </div>
            <div className="patient-card-actions">
              <button className="compact-button" onClick={() => setSelected(patient)}><Image size={14} /> Open</button>
              <button className="compact-button" onClick={() => { setEditing(patient); setShowForm(true); }}><Edit size={14} /> Edit</button>
              <button className="compact-button danger" onClick={() => removePatient(patient)}><Trash2 size={14} /> Deactivate</button>
            </div>
          </article>
        ))}
      </div>
    </Page>
  );
}

function PatientForm({ initial, onSave, onCancel }) {
  const [form, setForm] = React.useState(() => ({
    patient_code: initial?.patient_code || '',
    full_name: initial?.full_name || '',
    gender: initial?.gender || '',
    date_of_birth: initial?.date_of_birth || '',
    phone: initial?.phone || '',
    address: initial?.address || '',
    national_id: initial?.national_id || '',
    insurance_id: initial?.insurance_id || '',
    medical_history: initial?.medical_history || '',
    allergy_history: initial?.allergy_history || '',
    current_symptoms: initial?.current_symptoms || '',
    notes: initial?.notes || '',
  }));
  const [error, setError] = React.useState('');
  const [saving, setSaving] = React.useState(false);

  const setField = (key, value) => setForm((old) => ({ ...old, [key]: value }));
  const submit = async (event) => {
    event.preventDefault();
    setSaving(true);
    setError('');
    try {
      const payload = Object.fromEntries(Object.entries(form).map(([key, value]) => [key, value === '' ? null : value]));
      await onSave(payload);
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form className="panel patient-form" onSubmit={submit}>
      <div className="panel-title-row">
        <h3>{initial ? 'Edit patient' : 'Create patient'}</h3>
        <button type="button" className="compact-button" onClick={onCancel}>Cancel</button>
      </div>
      <div className="form-grid">
        <input value={form.patient_code} onChange={(e) => setField('patient_code', e.target.value)} placeholder="Patient code (auto if empty)" />
        <input value={form.full_name} onChange={(e) => setField('full_name', e.target.value)} placeholder="Full name" required />
        <select value={form.gender} onChange={(e) => setField('gender', e.target.value)}>
          <option value="">Gender</option>
          <option value="male">Male</option>
          <option value="female">Female</option>
          <option value="other">Other</option>
        </select>
        <input value={form.date_of_birth} onChange={(e) => setField('date_of_birth', e.target.value)} type="date" />
        <input value={form.phone} onChange={(e) => setField('phone', e.target.value)} placeholder="Phone" />
        <input value={form.national_id} onChange={(e) => setField('national_id', e.target.value)} placeholder="National ID" />
        <input value={form.insurance_id} onChange={(e) => setField('insurance_id', e.target.value)} placeholder="Insurance ID" />
        <input value={form.address} onChange={(e) => setField('address', e.target.value)} placeholder="Address" />
      </div>
      <textarea value={form.current_symptoms} onChange={(e) => setField('current_symptoms', e.target.value)} placeholder="Current symptoms" />
      <textarea value={form.medical_history} onChange={(e) => setField('medical_history', e.target.value)} placeholder="Medical history" />
      <textarea value={form.allergy_history} onChange={(e) => setField('allergy_history', e.target.value)} placeholder="Allergy history" />
      <textarea value={form.notes} onChange={(e) => setField('notes', e.target.value)} placeholder="Notes" />
      {error && <div className="error">{error}</div>}
      <button className="primary-button" disabled={saving}>{saving ? 'Saving...' : 'Save patient'}</button>
    </form>
  );
}

function PatientDetail({ token, patient, onBack, onEdit }) {
  const [studies, setStudies] = React.useState([]);
  const [file, setFile] = React.useState(null);
  const [threshold, setThreshold] = React.useState(0.25);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState('');
  const [selectedResult, setSelectedResult] = React.useState(null);

  const loadStudies = React.useCallback(async () => {
    setError('');
    try {
      setStudies(await apiClient(token).request(`/patients/${patient.id}/xray-studies`));
    } catch (err) {
      setError(err.message);
    }
  }, [patient.id, token]);

  React.useEffect(() => { loadStudies(); }, [loadStudies]);

  const diagnose = async () => {
    if (!file) return;
    setLoading(true);
    setError('');
    try {
      const form = new FormData();
      form.append('file', file);
      const study = await apiClient(token).request(`/patients/${patient.id}/xray-studies?conf_threshold=${threshold}`, {
        method: 'POST',
        body: form,
      });
      setSelectedResult(study.prediction);
      setFile(null);
      await loadStudies();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Page title={patient.full_name} subtitle={`${patient.patient_code} · ${patient.gender || 'gender not set'}`}>
      <div className="toolbar">
        <button className="secondary-button" onClick={onBack}><ArrowLeft size={16} /> Back</button>
        <button className="secondary-button" onClick={onEdit}><Edit size={16} /> Edit profile</button>
      </div>
      <div className="grid two">
        <section className="panel profile-panel">
          <h3>Patient Profile</h3>
          <Info label="Date of birth" value={patient.date_of_birth || '-'} />
          <Info label="Phone" value={patient.phone || '-'} />
          <Info label="Symptoms" value={patient.current_symptoms || '-'} />
          <Info label="Medical history" value={patient.medical_history || '-'} />
          <Info label="Allergy history" value={patient.allergy_history || '-'} />
        </section>
        <section className="panel">
          <h3>Upload X-ray & Diagnose</h3>
          <label className="dropzone">
            <UploadCloud size={28} />
            <span>{file ? file.name : 'Choose patient X-ray image'}</span>
            <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files?.[0] || null)} />
          </label>
          <label className="slider-label">Detection confidence: {threshold.toFixed(2)}</label>
          <input type="range" min="0.05" max="0.95" step="0.05" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} />
          <button className="primary-button" onClick={diagnose} disabled={!file || loading}>{loading ? 'Analyzing...' : 'Upload & diagnose'}</button>
          {error && <div className="error">{error}</div>}
        </section>
      </div>
      {selectedResult && <PredictionDetails result={selectedResult} token={token} />}
      <section className="panel">
        <h3>X-ray Studies</h3>
        <div className="study-list">
          {studies.map((study) => (
            <article className="study-card" key={study.id}>
              <div>
                <strong>{study.original_filename}</strong>
                <span>{new Date(study.created_at).toLocaleString()}</span>
              </div>
              <span className="badge">{study.study_status}</span>
              {study.prediction && (
                <>
                  <span>{study.prediction.predicted_class}</span>
                  <span>{(study.prediction.confidence * 100).toFixed(1)}%</span>
                  <button className="compact-button" onClick={() => setSelectedResult(study.prediction)}>Open diagnosis</button>
                </>
              )}
            </article>
          ))}
          {!studies.length && <div className="empty-state">No X-ray studies for this patient yet.</div>}
        </div>
      </section>
    </Page>
  );
}

function Info({ label, value }) {
  return <div className="info-row"><span>{label}</span><strong>{value}</strong></div>;
}

function HistoryPage({ token }) {
  const [patientId, setPatientId] = React.useState('');
  const patientsApi = useApi(token, '/patients?limit=100');
  const path = patientId ? `/predictions?patient_id=${patientId}` : '/predictions';
  const { data, error, loading, refresh } = useApi(token, path);
  const [selected, setSelected] = React.useState(null);
  return (
    <Page title="Prediction History" subtitle="Review saved diagnoses from PostgreSQL.">
      <div className="toolbar">
        <select value={patientId} onChange={(e) => setPatientId(e.target.value)}>
          <option value="">All patients and quick uploads</option>
          {(patientsApi.data || []).map((patient) => (
            <option key={patient.id} value={patient.id}>{patient.patient_code} - {patient.full_name}</option>
          ))}
        </select>
        <button className="secondary-button" onClick={refresh}>Refresh</button>
      </div>
      {loading && <div className="empty-state">Loading history...</div>}
      {error && <div className="error">{error}</div>}
      <div className="history-list">
        {(data || []).map((item) => (
          <article className="history-item" key={item.id}>
            <div>
              <strong>{item.filename}</strong>
              <span>{item.patient ? `${item.patient.patient_code} · ${item.patient.full_name}` : 'Quick upload'}</span>
              <span>{new Date(item.created_at).toLocaleString()}</span>
            </div>
            <div className="badge">{item.predicted_class}</div>
            <div>{(item.confidence * 100).toFixed(1)}%</div>
            <div><Clock size={14} /> {item.processing_time_ms.toFixed(0)} ms</div>
            <button className="compact-button" onClick={() => setSelected(selected === item.id ? null : item.id)}>
              <FileText size={14} /> Report
            </button>
            {selected === item.id && (
              <div className="history-report">
                <MedicalReportPanel predictionId={item.id} token={token} />
              </div>
            )}
          </article>
        ))}
      </div>
    </Page>
  );
}

function MedicalReportPanel({ predictionId, token }) {
  const [report, setReport] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState('');

  const loadExisting = React.useCallback(async () => {
    setError('');
    try {
      const payload = await apiClient(token).request(`/predictions/${predictionId}/medical-report`);
      setReport(payload);
    } catch (err) {
      if (!String(err.message).includes('not found')) {
        setError(err.message);
      }
    }
  }, [predictionId, token]);

  React.useEffect(() => {
    loadExisting();
  }, [loadExisting]);

  const generate = async (force = false) => {
    setLoading(true);
    setError('');
    try {
      const payload = await apiClient(token).request(`/predictions/${predictionId}/medical-report${force ? '?force=true' : ''}`, {
        method: 'POST',
      });
      setReport(payload);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="panel medical-report-panel">
      <div className="panel-title-row">
        <h3>Medical Report</h3>
        <div className="button-row">
          <button className="secondary-button" onClick={() => generate(false)} disabled={loading}>
            <FileText size={16} /> {report ? 'Load report' : 'Generate report'}
          </button>
          {report && (
            <button className="secondary-button" onClick={() => generate(true)} disabled={loading}>
              Retry
            </button>
          )}
        </div>
      </div>
      {loading && <div className="empty-state">Generating medical report with Gemini...</div>}
      {error && <div className="error">{error}</div>}
      {!loading && !report && !error && (
        <div className="empty-state">Generate a structured medical report for this prediction.</div>
      )}
      {report?.report && <MedicalReportView report={report} />}
    </section>
  );
}

function MedicalReportView({ report }) {
  const content = report.report;
  const sections = [
    ['Clinical summary', [content.clinical_summary]],
    ['Imaging findings', content.imaging_findings],
    ['AI interpretation', [content.ai_interpretation]],
    ['Risk level', [content.risk_level]],
    ['Recommendations', content.recommendations],
    ['Patient advice', content.patient_advice],
    ['Red flags', content.red_flags],
    ['Limitations', content.limitations],
    ['Next steps', content.next_steps],
    ['Disclaimer', [content.disclaimer]],
  ];
  return (
    <article className="medical-report-view">
      <div className="report-meta">
        <span>Status: {report.status}</span>
        <span>Model: {report.model_name}</span>
      </div>
      {sections.map(([title, items]) => (
        <section className="report-section" key={title}>
          <h4>{title}</h4>
          {items.length === 1 ? <p>{items[0]}</p> : (
            <ul>{items.map((item, index) => <li key={index}>{item}</li>)}</ul>
          )}
        </section>
      ))}
    </article>
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
