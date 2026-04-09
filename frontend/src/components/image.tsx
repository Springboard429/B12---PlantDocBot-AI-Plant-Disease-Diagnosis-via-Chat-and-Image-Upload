import { useState } from "react";

function ImagePrediction() {
  const [result, setResult] = useState("");

  const handlePredict = async () => {
    const fileInput = document.querySelector<HTMLInputElement>("input[type=file]");
    if (!fileInput?.files?.length) {
      alert("Please select a file!");
      return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
      const res = await fetch("http://127.0.0.1:8000/predict-image", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      setResult(data.predicted_disease);
    } catch (err) {
      console.error(err);
      setResult("Error predicting image");
    }
  };

  return (
    <div>
      <h2>Image Prediction</h2>
      <input type="file" />
      <button onClick={handlePredict}>Predict Image</button>
      <p>Result: {result}</p>
    </div>
  );
}

export default ImagePrediction;