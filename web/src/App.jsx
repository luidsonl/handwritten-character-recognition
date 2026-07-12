import { useState, useRef, useCallback } from 'react';
import Canvas from './components/Canvas';
import { useModel } from './hooks/useModel';
import { segmentAndPredict } from './lib/predict';
import './index.css';

function App() {
  const { model, loading, error } = useModel();
  const [predictions, setPredictions] = useState([]);
  const [recognizing, setRecognizing] = useState(false);
  const canvasRef = useRef(null);

  const handleCanvasReady = useCallback((canvas) => {
    canvasRef.current = canvas;
  }, []);

  const handleRecognize = useCallback(() => {
    if (!model || !canvasRef.current) return;
    setRecognizing(true);
    setPredictions([]);

    setTimeout(() => {
      try {
        const results = segmentAndPredict(canvasRef.current, model);
        setPredictions(results);
      } catch (err) {
        console.error('Prediction error:', err);
      }
      setRecognizing(false);
    }, 50);
  }, [model]);

  const recognizedWord = predictions.map((p) => p.char).join('');

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-300 flex flex-col items-center px-4 py-10">
      <header className="text-center mb-8">
        <h1 className="text-2xl font-semibold text-zinc-100">
          Reconhecimento de Caracteres Manuscritos
        </h1>
        <p className="text-sm text-zinc-500 mt-1">EMNIST Balanced &middot; CNN + TensorFlow.js</p>
      </header>

      <section className="flex flex-col items-center gap-3 mb-6">
        <Canvas onCanvasReady={handleCanvasReady} />
        <p className="text-xs text-zinc-500">Escreva um caractere ou palavra no canvas acima</p>
      </section>

      <section className="flex items-center gap-3 mb-8">
        <button
          onClick={handleRecognize}
          disabled={loading || recognizing || !window.opencvReady}
          className="px-6 py-2.5 rounded-lg bg-indigo-600 text-white font-semibold text-sm
                     hover:bg-indigo-500 active:scale-[0.97] transition-all
                     disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {recognizing ? 'Reconhecendo...' : 'Reconhecer'}
        </button>
        {!window.opencvReady && !loading && (
          <span className="text-xs text-zinc-500">Carregando OpenCV...</span>
        )}
      </section>

      {loading && (
        <div className="flex items-center gap-2 text-sm text-zinc-400">
          <div className="w-4 h-4 border-2 border-zinc-600 border-t-indigo-500 rounded-full animate-spin" />
          Carregando modelo...
        </div>
      )}

      {error && (
        <div className="w-full max-w-md p-3 rounded-lg bg-red-950 border border-red-800 text-red-400 text-sm">
          Erro: {error}
        </div>
      )}

      {predictions.length > 0 && (
        <section className="w-full max-w-md flex flex-col gap-3">
          <div className="flex items-baseline gap-3 p-4 rounded-lg bg-zinc-900 border border-zinc-800">
            <span className="text-xs text-zinc-500 uppercase tracking-wide">Resultado</span>
            <span className="text-3xl font-bold font-mono text-zinc-100 tracking-widest">
              {recognizedWord}
            </span>
          </div>

          <div className="flex flex-wrap gap-2 justify-center">
            {predictions.map((p, i) => (
              <div
                key={i}
                className="flex flex-col items-center px-3 py-2 rounded-md bg-zinc-900 border border-zinc-800 min-w-[48px]"
              >
                <span className="text-xl font-bold font-mono text-zinc-100">{p.char}</span>
                <span className="text-[10px] text-zinc-500">{(p.confidence * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </section>
      )}

      <footer className="mt-auto pt-10 text-center">
        <p className="text-xs text-zinc-600">Modelo treinado no dataset EMNIST Balanced (47 classes)</p>
      </footer>
    </div>
  );
}

export default App;
