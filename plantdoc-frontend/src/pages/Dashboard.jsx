import { useState } from "react";

function Dashboard() {
  const [mode, setMode] = useState("image");
  const [image, setImage] = useState(null);
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);

  const handlePredict = async () => {
  try {
    // IMAGE MODE
    if (mode === "image") {
      if (!image) return alert("Please upload an image");

      const formData = new FormData();
      formData.append("file", image);

      const res = await fetch("http://127.0.0.1:8000/predict-image", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      setResult({
  prediction: data?.prediction || data?.disease || "Unknown",
  confidence: data?.confidence,
});
    }

    // TEXT MODE
    else if (mode === "text") {
      if (!text) return alert("Please enter text");

      const res = await fetch("http://127.0.0.1:8000/predict-text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      const data = await res.json();

      setResult({
        prediction: data?.prediction || data?.disease || "Unknown",
        confidence: data?.confidence || 0.99,
      });
    }

    // 🚀 COMBINED MODE
    else if (mode === "combined") {
      if (!image || !text) {
        return alert("Upload image + enter text");
      }

      // image request
      const formData = new FormData();
      formData.append("file", image);

      const imgRes = await fetch("http://127.0.0.1:8000/predict-image", {
        method: "POST",
        body: formData,
      });

      const imgData = await imgRes.json();

      // text request
      const txtRes = await fetch("http://127.0.0.1:8000/predict-text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      const txtData = await txtRes.json();

      // ✅ SAFE prediction fix (IMPORTANT)
      const finalPrediction =
  imgData?.prediction ||
  imgData?.disease ||
  txtData?.prediction ||
  "Unknown";

      // confidence combine
      const finalConfidence =
        (imgData?.confidence ? parseFloat(imgData.confidence) : 50) * 0.7 +
        (txtData?.confidence ? txtData.confidence * 100 : 50) * 0.3;

      setResult({
        prediction: finalPrediction,
        confidence: finalConfidence,
      });
    }
  } catch (err) {
    console.error(err);
    alert("Something went wrong");
  }
};
  const handleClear = () => {
    setImage(null);
    setText("");
    setResult(null);
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.header}>Leaf Disease Detection</h2>

      <div style={styles.main}>
        {/* LEFT SIDE */}
        <div>
          {/* Tabs */}
          <div style={styles.tabs}>
            {["image", "text", "combined"].map((m) => (
              <button
                key={m}
                onClick={() => {
  setMode(m);

  // clear unwanted data when switching
  if (m === "text") {
    setImage(null);
  }

  if (m === "image") {
    setText("");
  }
}}
                style={{
                  ...styles.tabBtn,
                  background: mode === m ? "#2E7D32" : "#e0e0e0",
                  color: mode === m ? "white" : "#333",
                }}
              >
                {m.toUpperCase()}
              </button>
            ))}
          </div>

          {/* Input Box */}
          <div style={styles.box}>
            {mode === "image" && (
              <>
                <p style={styles.label}>Upload Image</p>
                <input
                  type="file"
                  onChange={(e) => setImage(e.target.files[0])}
                />
              </>
            )}

            {mode === "text" && (
              <>
                <p style={styles.label}>Enter Symptoms</p>
                <input
                  type="text"
                  placeholder="Describe disease..."
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  style={styles.input}
                />
              </>
            )}

            {mode === "combined" && (
              <>
                <p style={styles.label}>Upload + Describe</p>
                <input
                  type="file"
                  onChange={(e) => setImage(e.target.files[0])}
                />
                <input
                  type="text"
                  placeholder="Describe disease..."
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  style={styles.input}
                />
              </>
            )}

            {/* Image Preview */}
            {image && (
              <img
                src={URL.createObjectURL(image)}
                alt="preview"
                style={styles.preview}
              />
            )}

            {/* Buttons */}
            <div style={styles.buttonRow}>
              <button style={styles.predictBtn} onClick={handlePredict}>
                Predict
              </button>
              <button style={styles.clearBtn} onClick={handleClear}>
                Clear
              </button>
            </div>
          </div>
        </div>

        {/* RIGHT SIDE */}
        <div style={styles.resultBox}>
          <h3>Prediction Result</h3>

          {result ? (
            <div>
              <p>
  <strong>Disease:</strong>{" "}
  {result && result.prediction
    ? result.prediction.replace(/_/g, " ")
    : "No prediction"}
</p>
               <p>
  Confidence:{" "}
  {result?.confidence
    ? (result.confidence > 1
        ? result.confidence.toFixed(2)
        : (result.confidence * 100).toFixed(2)
      ) + "%"
    : "N/A"}
</p>
            </div>
          ) : (
            <p style={styles.noResult}>
              Upload data and click <b>Predict</b> to see results
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

const styles = {
container: {
  minHeight: "100vh",
  padding: "80px 40px 40px",  // 👈 TOP MORE, BOTTOM LESS
},

header: {
  textAlign: "center",
  color: "#1B5E20",
  marginBottom: "50px",
  fontSize: "2.8rem",
  fontWeight: "700",
},

main: {
  display: "grid",
  gridTemplateColumns: "1fr 1fr",
  gap: "30px",
  maxWidth: "1200px",
  margin: "0 auto",
},

  tabs: {
    display: "flex",
    gap: "10px",
    marginBottom: "15px",
  },

  tabBtn: {
    padding: "8px 14px",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    fontWeight: "500",
  },

  box: {
    background: "white",
    padding: "30px",
    borderRadius: "15px",
    boxShadow: "0 10px 25px rgba(0,0,0,0.08)",
    minHeight: "250px",
  },

  resultBox: {
    background: "white",
    padding: "30px",
    borderRadius: "15px",
    boxShadow: "0 10px 25px rgba(0,0,0,0.08)",
    minHeight: "300px",
  },

  label: {
    marginBottom: "10px",
    fontWeight: "500",
  },

  input: {
    width: "100%",
    padding: "10px",
    marginTop: "10px",
    borderRadius: "8px",
    border: "1px solid #ccc",
    background: "#f9f9f9",
  },

  buttonRow: {
    marginTop: "15px",
  },

  predictBtn: {
    padding: "10px 20px",
    background: "linear-gradient(135deg, #2E7D32, #43A047)",
    color: "white",
    borderRadius: "8px",
    marginRight: "10px",
    border: "none",
    boxShadow: "0 5px 15px rgba(46,125,50,0.3)",
    cursor: "pointer",
  },

  clearBtn: {
    padding: "10px 20px",
    background: "#e0e0e0",
    borderRadius: "8px",
    border: "none",
    cursor: "pointer",
  },

  preview: {
    width: "120px",
    marginTop: "15px",
    borderRadius: "10px",
  },

  noResult: {
    color: "#777",
    marginTop: "20px",
  },
};

export default Dashboard;