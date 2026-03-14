import React, { useState, useRef } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleDrag = function(e) {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const processFile = (selectedFile) => {
    if (selectedFile && selectedFile.type.startsWith('image/')) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    } else {
      setError("Please select a valid image file.");
    }
  };

  const handleDrop = function(e) {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = function(e) {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const onUploadClick = () => {
    fileInputRef.current.click();
  };

  const resetState = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if(fileInputRef.current) fileInputRef.current.value = "";
  };

  const analyzeImage = async () => {
    if (!file) return;
    
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      if(data.error) {
        setError(data.error);
      } else {
        setResult(data);
      }
    } catch (err) {
      console.error("Analysis failed:", err);
      setError("Failed to connect to the backend server. Make sure FastAPI is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  const formatClassName = (name) => {
    if(!name) return "";
    return name.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>NeuroAI</h1>
        <p>Advanced Brain Tumor Classification</p>
      </header>

      <main className="main-content">
        {/* Upload Panel */}
        <div className="glass-panel">
          <div 
            className={`upload-area ${dragActive ? "active" : ""}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={!preview ? onUploadClick : undefined}
          >
            <input 
              ref={fileInputRef}
              type="file" 
              className="file-input" 
              accept="image/*" 
              onChange={handleChange} 
            />
            
            {preview ? (
              <img src={preview} alt="MRI Preview" className="preview-image" />
            ) : (
              <>
                <div className="upload-icon">🔬</div>
                <div className="upload-text">
                  <strong>Drag & Drop</strong> your MRI scan here<br />
                  or click to browse
                </div>
              </>
            )}
          </div>

          {preview && (
            <button className="btn-secondary" onClick={resetState} style={{marginTop: '1rem', alignSelf: 'center'}}>
              Remove Image
            </button>
          )}

          {error && <div style={{color: '#ff4b4b', marginTop: '1rem', textAlign: 'center'}}>{error}</div>}

          <button 
            className="btn-primary" 
            onClick={analyzeImage} 
            disabled={!file || loading}
          >
            {loading ? "Analyzing..." : "Analyze Scan"}
          </button>
        </div>

        {/* Results Panel */}
        {(loading || result) && (
          <div className="glass-panel" style={{animationDelay: '0.2s'}}>
            {loading ? (
              <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%'}}>
                <div className="loader"></div>
                <p style={{color: 'var(--accent-color)'}}>Processing MRI scan via Deep Learning...</p>
              </div>
            ) : (
              result && (
                <div className="results-container">
                  <div className="result-card">
                    <div className="result-title">Detection Result</div>
                    <div className="result-value">
                      {formatClassName(result.prediction)}
                    </div>
                    {result.confidence && (
                      <div style={{marginTop: '1rem', textAlign: 'left'}}>
                        <div style={{display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem'}}>
                          <span>Confidence</span>
                          <span>{(result.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div className="confidence-bar-container">
                          <div 
                            className="confidence-bar" 
                            style={{width: `${result.confidence * 100}%`}}
                          ></div>
                        </div>
                      </div>
                    )}
                  </div>

                  {result.gradcam_base64 && (
                    <div className="gradcam-container">
                      <div className="result-title" style={{marginBottom: '1rem'}}>Grad-CAM Activation</div>
                      <img 
                        src={`data:image/jpeg;base64,${result.gradcam_base64}`} 
                        alt="Grad-CAM Visualization" 
                        className="gradcam-image" 
                      />
                    </div>
                  )}

                  {result.is_mock && (
                    <div className="mock-warning">
                      <strong>Mock Mode Active</strong><br/>
                      {result.message}
                    </div>
                  )}
                </div>
              )
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
