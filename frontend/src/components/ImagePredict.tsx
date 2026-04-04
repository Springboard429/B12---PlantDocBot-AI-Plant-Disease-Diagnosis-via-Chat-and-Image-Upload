// image prediction component

import { useState } from "react";
import { predictImage } from "../services/api";
import ResultCard from "./ResultCard";

function ImagePredict() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    if (!file) return;

    setLoading(true);

    try {
      const data = await predictImage(file);
      setResult(data);
    } catch (err) {
      console.error(err);
    }

    setLoading(false);
  };

  return (
    <div className="predict-box">

      <input
        type="file"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
      />

      <button onClick={handlePredict}>analyze</button>

      {loading && <p>predicting...</p>}

      {result && <ResultCard data={result} title="image prediction" />}

    </div>
  );
}

export default ImagePredict;