import React, { useState } from "react";
import axios from "axios";
import bg from "./bg.avif";   

function App() {
  const [activeTab, setActiveTab] = useState("image");

  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null); // 🔥 NEW
  const [imageResult, setImageResult] = useState("");

  const [text, setText] = useState("");
  const [textResult, setTextResult] = useState("");

  // Image upload
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setImageResult(""); // clear old result

    if (file) {
      setPreview(URL.createObjectURL(file)); // 🔥 preview
    }
  };

  const predictImage = async () => {
    if (!image) return alert("Upload image first");

    const formData = new FormData();
    formData.append("file", image);

    try {
      const res = await axios.post(
        "http://127.0.0.1:8000/predict-image",
        formData
      );
      setImageResult(res.data.predicted_disease);
    } catch (err) {
      alert("Error predicting image");
    }
  };

  // Text prediction
  const predictText = async () => {
    if (!text) return alert("Enter description");

    const formData = new FormData();
    formData.append("description", text);

    try {
      const res = await axios.post(
        "http://127.0.0.1:8000/predict-text",
        formData
      );
      setTextResult(res.data.predicted_disease);
    } catch (err) {
      alert("Error predicting text");
    }
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>🌿 Plant Disease Detector</h1>

      {/* Tabs */}
      <div style={styles.tabs}>
        <button
          style={activeTab === "image" ? styles.activeTab : styles.tab}
          onClick={() => setActiveTab("image")}
        >
          Image
        </button>
        <button
          style={activeTab === "text" ? styles.activeTab : styles.tab}
          onClick={() => setActiveTab("text")}
        >
          Text
        </button>
      </div>

      {/* CARD */}
      <div style={styles.card}>
        {activeTab === "image" && (
          <>
            <h2>Upload Leaf Image</h2>
            <input type="file" onChange={handleImageUpload} />

            {/* 🔥 IMAGE PREVIEW */}
            {preview && (
              <img
                src={preview}
                alt="preview"
                style={styles.imagePreview}
              />
            )}

            <button style={styles.button} onClick={predictImage}>
              Predict Image
            </button>

            {imageResult && (
              <p style={styles.result}>Result: {imageResult}</p>
            )}
          </>
        )}

        {activeTab === "text" && (
          <>
            <h2>Enter Leaf Description</h2>
            <textarea
              rows="4"
              style={styles.textarea}
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
            <button style={styles.buttonBlue} onClick={predictText}>
              Predict Text
            </button>

            {textResult && (
              <p style={styles.resultBlue}>Result: {textResult}</p>
            )}
          </>
        )}
      </div>
    </div>
  );
}

const styles = {
  container: {
  minHeight: "100vh",
  backgroundImage: `url(${bg})`, // ✅ correct
  backgroundSize: "cover",
  backgroundPosition: "center",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  padding: "40px",
  fontFamily: "Arial",
},
  title: {
    marginBottom: "30px",
    color: "white",
    fontSize: "36px",
    textShadow: "2px 2px 5px rgba(0,0,0,0.7)", // 🔥 glow effect
  },
  tabs: {
    marginBottom: "20px",
  },
  tab: {
    padding: "12px 25px",
    margin: "5px",
    border: "none",
    background: "rgba(255,255,255,0.6)",
    cursor: "pointer",
    borderRadius: "8px",
    fontSize: "16px",
  },
  activeTab: {
    padding: "12px 25px",
    margin: "5px",
    border: "none",
    background: "rgba(46,125,50,0.8)",
    color: "white",
    cursor: "pointer",
    borderRadius: "8px",
    fontSize: "16px",
  },

  // 🔥 GLASS EFFECT CARD
  card: {
    background: "rgba(255, 255, 255, 0.2)", // transparent
    backdropFilter: "blur(10px)",          // blur effect
    WebkitBackdropFilter: "blur(10px)",    // for Safari
    padding: "30px",
    borderRadius: "20px",
    boxShadow: "0 8px 32px rgba(0,0,0,0.3)",
    width: "90%",
    maxWidth: "600px",
    textAlign: "center",
    color: "white",
  },

  imagePreview: {
    width: "224px",
    height: "224px",
    marginTop: "15px",
    borderRadius: "10px",
    objectFit: "cover",
    border: "2px solid white",
    display: "block",
    marginLeft: "auto",
    marginRight: "auto",
  },

  button: {
    marginTop: "15px",
    padding: "12px",
    width: "100%",
    background: "#2e7d32",
    color: "white",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    fontSize: "16px",
  },
  buttonBlue: {
    marginTop: "15px",
    padding: "12px",
    width: "100%",
    background: "#1565c0",
    color: "white",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    fontSize: "16px",
  },
  textarea: {
    width: "100%",
    padding: "12px",
    borderRadius: "8px",
    border: "none",
    outline: "none",
  },
  result: {
    marginTop: "20px",
    color: "#00ffcc",
    fontWeight: "bold",
    fontSize: "18px",
  },
  resultBlue: {
    marginTop: "20px",
    color: "#00ccff",
    fontWeight: "bold",
    fontSize: "18px",
  },
};

export default App;