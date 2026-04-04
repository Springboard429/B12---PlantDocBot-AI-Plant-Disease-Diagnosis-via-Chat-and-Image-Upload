// result display component

function ResultCard({ data, title }: any) {

  const disease = data.disease || data.prediction;

  const isInvalid =
    disease.toLowerCase().includes("invalid") ||
    disease.toLowerCase().includes("not related");

  return (
    <div className="result-card">
      <h3>{title}</h3>

      <p><strong>Prediction:</strong> {disease}</p>

      {!isInvalid && (
        <p><strong>Confidence:</strong> {data.confidence}%</p>
      )}
    </div>
  );
}

export default ResultCard;