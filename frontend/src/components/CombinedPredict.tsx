// combined prediction component

import { useState } from "react";
import { predictCombined } from "../services/api";
import ResultCard from "./ResultCard";

function CombinedPredict() {
  const [file, setFile] = useState<File | null>(null);
  const [text, setText] = useState("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    if (!file && !text) return;

    setLoading(true);

    try {
      const data = await predictCombined(text, file);
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

      <textarea
        placeholder="describe symptoms..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <button onClick={handlePredict}>analyze</button>

      {loading && <p>predicting...</p>}

      {result && (
        <div style={{ width: "100%" }}>

          {/* image result */}
          {result.image_prediction && (
            <ResultCard
              data={result.image_prediction}
              title="image prediction"
            />
          )}

          {/* text result */}
          {result.text_prediction && (
            <ResultCard
              data={result.text_prediction}
              title="text prediction"
            />
          )}

        </div>
      )}

    </div>
  );
}

export default CombinedPredict;