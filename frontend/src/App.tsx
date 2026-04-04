// main layout with correct mode mapping

import { useState } from "react";
import ImagePredict from "./components/ImagePredict";
import TextPredict from "./components/TextPredict";
import CombinedPredict from "./components/CombinedPredict";
import "./App.css";

function App() {
  const [tab, setTab] = useState<"image" | "text" | "combined">("combined");
  const [dark, setDark] = useState(false);

  return (
    <div className={dark ? "app dark" : "app"}>

      <div className="top-bar">
        <h1>🌿 PlantDocBot</h1>

        <button onClick={() => setDark(!dark)}>
          {dark ? "🌞" : "🌙"}
        </button>
      </div>

      <div className="center-container">

        <div className="description">
          <p>
            PlantDocBot is an AI-powered tool that helps identify plant diseases using image and text inputs. 
            Users can upload a leaf image, describe the symptoms, or use both for getting results.
          </p>
        </div>

        <div className="mode-buttons">
          <button onClick={() => setTab("image")} className={tab==="image"?"active":""}>
            Image Prediction
          </button>
          <button onClick={() => setTab("text")} className={tab==="text"?"active":""}>
            Text Prediction
          </button>
          <button onClick={() => setTab("combined")} className={tab==="combined"?"active":""}>
            Combined Prediction
          </button>
        </div>

        <div className="content">
          {tab === "image" && <ImagePredict />}
          {tab === "text" && <TextPredict />}
          {tab === "combined" && <CombinedPredict />}
        </div>

      </div>
    </div>
  );
}

export default App;