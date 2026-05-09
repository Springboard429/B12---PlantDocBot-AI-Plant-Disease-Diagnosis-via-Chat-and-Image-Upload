import React, { useState, useRef } from "react";
import "./App.css";

import { predictImage, predictText } from "./api";

function App() {
  const [activeTab, setActiveTab] = useState("image");
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [text, setText] = useState("");
  const [fileName, setFileName] = useState("");

  const fileRef = useRef(null);

  // Upload click
  const handleClickUpload = () => {
    fileRef.current?.click();
  };

  // File select
  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) setFileName(file.name);
  };

  // Add symptom tag
  const addTag = (tag) => {
    setText((prev) => (prev ? prev + ", " + tag : tag));
  };

  // =========================
  // MAIN SCAN FUNCTION
  // =========================
  const handleScan = async () => {
    setLoading(true);
    setPrediction(null);

    try {
      // ================= IMAGE =================
      if (activeTab === "image") {
        const file = fileRef.current?.files?.[0];

        if (!file) {
          alert("Upload image first");
          setLoading(false);
          return;
        }

        const data = await predictImage(file);

        setPrediction({
          label: data.disease,
          confidence: data.confidence,
          model: "CNN",
          top_3: data.top_3 || null,
        });
      }

      // ================= TEXT =================
      else {
        if (!text.trim()) {
          alert("Enter symptoms first");
          setLoading(false);
          return;
        }

        const data = await predictText(text);

        setPrediction({
          label: data.disease,
          confidence: data.confidence,
          model: "DistilBERT",
          top_3: data.top_3 || null,
        });
      }
    } catch (err) {
      console.error("API ERROR:", err);
      alert("Backend error. Check console.");
    }

    setLoading(false);
  };

  return (
    <div className="app-container">

      {/* HEADER */}
      <header className="header">
        <h1>
          🌿 Plant <span>Doc</span> Bot
        </h1>
        <p className="subtitle">AI-POWERED PLANT DISEASE DETECTION</p>
      </header>

      {/* TABS */}
      <div className="tabs">
        <button
          className={activeTab === "image" ? "active" : ""}
          onClick={() => setActiveTab("image")}
        >
          IMAGE SCAN
        </button>

        <button
          className={activeTab === "symptom" ? "active" : ""}
          onClick={() => setActiveTab("symptom")}
        >
          SYMPTOM ANALYSIS
        </button>
      </div>

      {/* MAIN */}
      <div className="main">

        {/* LEFT PANEL */}
        <div className="card">
          {activeTab === "image" ? (
            <>
              <h3>Upload Leaf Image</h3>

              <div className="drop-zone" onClick={handleClickUpload}>
                <p>Click to upload image</p>
                <span>{fileName || "No file selected"}</span>
              </div>

              <input
                type="file"
                ref={fileRef}
                onChange={handleFileChange}
                hidden
              />

              <button onClick={handleScan}>
                {loading ? "Scanning..." : "Upload & Scan"}
              </button>
            </>
          ) : (
            <>
              <h3>Describe Symptoms</h3>

              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="e.g. yellow spots on tomato leaves"
              />

              <div className="tags">
                {["Yellow spots", "Brown spots", "Wilting", "Powdery"].map(
                  (tag) => (
                    <span key={tag} onClick={() => addTag(tag)}>
                      {tag}
                    </span>
                  )
                )}
              </div>

              <button onClick={handleScan}>
                {loading ? "Analyzing..." : "Analyse Symptoms"}
              </button>
            </>
          )}
        </div>

        {/* RIGHT PANEL */}
        <div className="card">

          {!prediction ? (
            <div className="placeholder">
              {loading ? "Analyzing..." : "Awaiting input..."}
            </div>
          ) : (
            <>
              <h2 className="result-title">{prediction.label}</h2>

              <p className="model-text">
                Model: {prediction.model}
              </p>

              {/* CONFIDENCE BAR */}
              <div className="confidence-box">
                <div
                  className="bar"
                  style={{ width: `${prediction.confidence}%` }}
                ></div>
              </div>

              <p className="confidence-text">
                Confidence: {prediction.confidence}%
              </p>

              {/* TOP 3 PREDICTIONS */}
              {prediction.top_3 && (
                <div style={{ marginTop: "15px" }}>
                  <h4>Top Predictions</h4>

                  {prediction.top_3.map((item, i) => (
                    <p key={i}>
                      {item.disease} — {item.confidence}%
                    </p>
                  ))}
                </div>
              )}
            </>
          )}
        </div>

      </div>
    </div>
  );
}

export default App;