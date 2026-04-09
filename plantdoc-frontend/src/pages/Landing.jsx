import { useNavigate } from "react-router-dom";

function Landing() {
  const navigate = useNavigate();

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <h1 style={styles.title}>Leaf Disease Analyzer</h1>

        <p style={styles.subtitle}>
          Upload images, enter symptoms, or combine both to detect Leaf diseases using AI.
        </p>

        <button style={styles.button} onClick={() => navigate("/app")}>
          Start Analysis →
        </button>
      </div>
    </div>
  );
}

const styles = {
  container: {
    height: "100vh",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    background: "linear-gradient(135deg, #d4f5dc, #b8e6c1)",
  },

  card: {
    background: "white",
    padding: "45px",
    borderRadius: "20px",
    boxShadow: "0 20px 50px rgba(0,0,0,0.1)",
    textAlign: "center",
    width: "420px",
  },

  title: {
    color: "#1B5E20",
    fontSize: "2.6rem",
    marginBottom: "12px",
    fontWeight: "600",
  },

  subtitle: {
    color: "#4CAF50",
    marginBottom: "30px",
    fontSize: "1.05rem",
  },

  button: {
    padding: "12px 28px",
    fontSize: "1rem",
    background: "linear-gradient(135deg, #2E7D32, #43A047)",
    color: "white",
    border: "none",
    borderRadius: "10px",
    cursor: "pointer",
    boxShadow: "0 8px 20px rgba(46,125,50,0.3)",
    transition: "0.3s",
  },
};

export default Landing;