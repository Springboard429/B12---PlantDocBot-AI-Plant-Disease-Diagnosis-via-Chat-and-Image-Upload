import './App.css';
import ImagePrediction from './components/image';
import TextPrediction from './components/text';
//import CombinedPrediction from './components/combined';

function App() {
  return (
    <div>
      <h1>Plant Disease Diagnosis</h1>

      <ImagePrediction />
      <TextPrediction />
       
    </div>
  );
}

export default App;
