import { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';

export function useModel() {
  const [model, setModel] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const modelRef = useRef(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const m = await tf.loadLayersModel('/model/model.json');
        if (!cancelled) {
          modelRef.current = m;
          setModel(m);
          setLoading(false);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message || 'Falha ao carregar o modelo');
          setLoading(false);
        }
      }
    }

    load();

    return () => {
      cancelled = true;
      if (modelRef.current) {
        modelRef.current.dispose();
      }
    };
  }, []);

  return { model, loading, error };
}
