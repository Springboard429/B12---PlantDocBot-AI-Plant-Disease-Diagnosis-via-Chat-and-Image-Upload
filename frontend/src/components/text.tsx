import { useState } from "react";

function TextPrediction() {
  const [description, setDescription] = useState("");
  const [result, setResult] = useState("");

  const handlePredict = async () => {
    if (!description) {
      alert("Please enter a description!");
      return;
    }

    const formData = new FormData();
    formData.append("description", description);

    try {
      const res = await fetch("http://127.0.0.1:8000/predict-text", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      setResult(data.predicted_disease);
    } catch (err) {
      console.error(err);
      setResult("Error predicting text");
    }
  };

  return (
    <div>
      <h2>Text Prediction</h2>
      <textarea
        placeholder="Describe the leaf..."
        value={description}
        onChange={(e) => setDescription(e.target.value)}
      ></textarea>
      <button onClick={handlePredict}>Predict Text</button>
      <p>Result: {result}</p>
    </div>
  );
}

export default TextPrediction;