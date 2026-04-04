// text prediction component

import { useState } from "react";
import { predictText } from "../services/api";
import ResultCard from "./ResultCard";

function TextPredict() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    if (!text) return;

    setLoading(true);

    try {
      const data = await predictText(text);
      setResult(data);
    } catch (err) {
      console.error(err);
    }

    setLoading(false);
  };

  return (
    <div className="predict-box">

      <textarea
        placeholder="describe plant symptoms..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <button onClick={handlePredict}>analyze</button>

      {loading && <p>predicting...</p>}

      {result && <ResultCard data={result} title="text prediction" />}

    </div>
  );
}

export default TextPredict;