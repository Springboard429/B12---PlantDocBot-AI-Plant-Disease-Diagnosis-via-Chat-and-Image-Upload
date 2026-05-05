import React, { useState } from 'react';
import { UploadCloud, Settings, Search, ZoomIn, Maximize2, Leaf } from 'lucide-react';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('IMAGE');
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState(null);
  const [previewSrc, setPreviewSrc] = useState("https://images.unsplash.com/photo-1595054178556-91ed82ec0573?auto=format&fit=crop&w=600&h=400&q=80");
  const [isPredicting, setIsPredicting] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [textInput, setTextInput] = useState("");
  const [searchQuery, setSearchQuery] = useState("");

  const PLANT_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
  ];

  const filteredClasses = PLANT_CLASSES.filter(cls => 
    cls.replace(/_/g, ' ').toLowerCase().includes(searchQuery.toLowerCase())
  );

  const groupedClasses = filteredClasses.reduce((acc, cls) => {
    const parts = cls.split('___');
    const plantName = parts[0].replace(/_/g, ' ');
    const diseaseName = parts[1] ? parts[1].replace(/_/g, ' ') : '';
    
    if (!acc[plantName]) acc[plantName] = [];
    acc[plantName].push(diseaseName || cls);
    return acc;
  }, {});

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const selectedFile = e.dataTransfer.files[0];
      setFile(selectedFile);
      setPreviewSrc(URL.createObjectURL(selectedFile));
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreviewSrc(URL.createObjectURL(selectedFile));
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreviewSrc("https://images.unsplash.com/photo-1595054178556-91ed82ec0573?auto=format&fit=crop&w=600&h=400&q=80");
    setPrediction(null);
    setErrorMsg(null);
  };

  const handlePredict = async () => {
    setErrorMsg(null);
    setIsPredicting(true);

    if (activeTab === 'IMAGE') {
      if (!file) {
        setErrorMsg("Please upload an image first.");
        setIsPredicting(false);
        return;
      }
      
      const formData = new FormData();
      formData.append("file", file);

      try {
        const response = await fetch("http://localhost:8000/predict/image", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) throw new Error(`Error: ${response.statusText}`);

        const data = await response.json();
        setPrediction({
          predicted_class: data.predicted_class,
          confidence: data.confidence,
          class_id: data.class_id,
          model: "simplecnn"
        });
      } catch (err) {
        console.error(err);
        setErrorMsg("Failed to connect to the backend. Is it running?");
      } finally {
        setIsPredicting(false);
      }
    } else if (activeTab === 'TEXT') {
      if (!textInput.trim()) {
        setErrorMsg("Please enter text symptoms first.");
        setIsPredicting(false);
        return;
      }

      try {
        const response = await fetch("http://localhost:8000/predict/text", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: textInput }),
        });

        if (!response.ok) throw new Error(`Error: ${response.statusText}`);

        const data = await response.json();
        setPrediction({
          predicted_class: data.predicted_disease || data.predicted_class || 'Unknown',
          confidence: data.confidence || 0,
          class_id: "N/A",
          model: "HF Text Model"
        });
      } catch (err) {
        console.error(err);
        setErrorMsg("Failed to connect to the backend. Is it running?");
      } finally {
        setIsPredicting(false);
      }
    }
  };

  return (
    <div className="app-container">
      {/* Navbar */}
      <nav className="navbar animate-fade-in">
        <div className="nav-left">
          <Leaf className="logo-icon" />
          <div className="logo-text">
            <h1>PLANTDOC</h1>
            <p>AI Dashboard</p>
          </div>
        </div>
        
        <div className="nav-right">
          <div className="status-badge">
            <span className="status-dot"></span>
            MODELS READY
          </div>
          <div className="status-badge" style={{ borderColor: 'rgba(56, 189, 248, 0.3)', background: 'rgba(56, 189, 248, 0.1)', color: '#38bdf8' }}>
            API CONNECTED
          </div>
        </div>
      </nav>

      <main className="main-grid">
        {/* Left Panel */}
        <div className="left-panel animate-fade-in" style={{ animationDelay: '0.1s' }}>
          <div className="glass-panel">
            <div className="tabs">
              <div 
                className={`tab ${activeTab === 'IMAGE' ? 'active' : ''}`}
                onClick={() => setActiveTab('IMAGE')}
              >
                IMAGE
              </div>
              <div 
                className={`tab ${activeTab === 'TEXT' ? 'active' : ''}`}
                onClick={() => setActiveTab('TEXT')}
              >
                TEXT
              </div>
            </div>

            {activeTab === 'IMAGE' && (
              <>
                <label 
                  className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                >
                  <input 
                    type="file" 
                    style={{ display: 'none' }} 
                    onChange={handleChange}
                    accept="image/*"
                  />
                  <UploadCloud className="upload-icon" />
                  <div className="upload-text">
                    Drag & Drop or Click to Upload Leaf Image
                  </div>
                </label>

                {file && (
                  <div className="file-info">
                    Original filename: <span>{file.name}</span>
                  </div>
                )}
                {!file && (
                  <div className="file-info">
                    Original filename: <span>0bb8fb61-d561...FrgE.S 2849.JPG</span>
                  </div>
                )}
              </>
            )}

            {activeTab === 'TEXT' && (
              <div className="text-input-zone">
                <textarea 
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  placeholder="Describe the symptoms of the plant (e.g. 'My tomato leaf has black spots and yellowing edges.')"
                  className="symptom-textarea"
                />
              </div>
            )}

            <div className="action-buttons">
              <button className="btn btn-primary" onClick={handlePredict} disabled={isPredicting}>
                {isPredicting ? "Predicting..." : "Predict Disease"}
              </button>
              <button className="btn btn-secondary" onClick={handleReset} disabled={isPredicting}>Reset</button>
            </div>
            
            {errorMsg && (
              <div style={{ color: "var(--danger)", marginTop: "1rem", fontSize: "0.9rem", textAlign: "center" }}>
                {errorMsg}
              </div>
            )}
          </div>

          <div className="glass-panel">
            <div className="preview-header">PREVIEW</div>
            <div className="preview-image-container">
              <img 
                src={previewSrc}
                alt="Leaf Preview" 
                className="preview-image" 
              />
              <div className="preview-actions">
                <button className="preview-btn">
                  <Maximize2 size={16} />
                </button>
                <button className="preview-btn">
                  <ZoomIn size={16} />
                </button>
              </div>
            </div>
            <p className="preview-caption">
              Clean caption the image on its triggers zoom and magnifier image.
            </p>
          </div>
        </div>

        {/* Right Panel */}
        <div className="right-panel animate-fade-in" style={{ animationDelay: '0.2s' }}>
          <div className="glass-panel">
            <div className="results-header">PREDICTION RESULTS</div>
            
            {!prediction ? (
              <div style={{ padding: '2rem 0', textAlign: 'center', color: 'var(--text-muted)' }}>
                Upload an image and click Predict Disease to see results.
              </div>
            ) : (
              <>
                <h2 className="disease-title text-gradient">{prediction.predicted_class}</h2>
                <div className="model-info">Model: {prediction.model} (ID: {prediction.class_id})</div>

                <div className="confidence-container">
                  <div className="confidence-bar">
                    {[...Array(10)].map((_, i) => (
                      <div key={i} className={`confidence-segment ${i < Math.round(prediction.confidence * 10) ? 'filled' : ''}`}></div>
                    ))}
                  </div>
                  <div className="confidence-text">Confidence: {(prediction.confidence * 100).toFixed(2)}%</div>
                </div>
              </>
            )}
          </div>

          <div className="glass-panel">
            <div className="classes-header">
              <div className="classes-title">KNOWN DISEASE CLASSES ({PLANT_CLASSES.length})</div>
            </div>
            
            <div className="search-container">
              <Search className="search-icon" />
              <input 
                type="text" 
                className="search-input" 
                placeholder="Filter by name..." 
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>

            <div style={{ maxHeight: '400px', overflowY: 'auto', paddingRight: '4px' }}>
              {Object.entries(groupedClasses).map(([plant, diseases]) => (
                <div className="category-group" key={plant}>
                  <div className="category-title">{plant}</div>
                  <div className="badges-container">
                    {diseases.map((disease, i) => (
                      <span key={i} className="disease-badge">{disease}</span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
