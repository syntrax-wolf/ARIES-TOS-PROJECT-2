import { useState } from "react";

export default function Sidebar({ onTrain }) {
  const [dataset, setDataset] = useState("iris");
  const [learningRate, setLearningRate] = useState(0.1);
  const [epochs, setEpochs] = useState(100);
  const [regularization, setRegularization] = useState(0.01);

  const handleTrain = () => {
    onTrain({
      dataset,
      learningRate,
      epochs,
      regularization,
    });
  };

  return (
    <div className="w-72 bg-slate-900 border-r border-slate-800 p-6">
      <h2 className="text-white text-2xl font-bold mb-8">
        Configuration
      </h2>

      {/* Dataset */}

      <div className="mb-6">
        <label className="text-slate-400 text-sm block mb-2">
          Dataset
        </label>

        <select
          value={dataset}
          onChange={(e) => setDataset(e.target.value)}
          className="w-full bg-slate-800 text-white rounded-lg p-3"
        >
          <option value="iris">🌸 Iris</option>
          <option value="linear">📈 Linear</option>
          <option value="blobs">🔵 Blobs</option>
          <option value="moons">🌙 Moons</option>
          <option value="circles">⭕ Circles</option>
        </select>
      </div>

      {/* Learning Rate */}

      <div className="mb-6">
        <label className="text-slate-400 text-sm block mb-2">
          Learning Rate
        </label>

        <input
          type="range"
          min="0.001"
          max="1"
          step="0.001"
          value={learningRate}
          onChange={(e) => setLearningRate(Number(e.target.value))}
          className="w-full"
        />

        <p className="text-cyan-400 mt-1">
          {learningRate.toFixed(3)}
        </p>
      </div>

      {/* Epochs */}

      <div className="mb-6">
        <label className="text-slate-400 text-sm block mb-2">
          Epochs
        </label>

        <input
          type="range"
          min="10"
          max="500"
          value={epochs}
          onChange={(e) => setEpochs(Number(e.target.value))}
          className="w-full"
        />

        <p className="text-cyan-400 mt-1">
          {epochs}
        </p>
      </div>

      {/* Regularization (future feature) */}

      <div className="mb-8">
        <label className="text-slate-400 text-sm block mb-2">
          Regularization (Coming Soon)
        </label>

        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={regularization}
          onChange={(e) => setRegularization(Number(e.target.value))}
          className="w-full"
          disabled
        />

        <p className="text-cyan-400 mt-1">
          {regularization.toFixed(2)}
        </p>
      </div>

      <button
        onClick={handleTrain}
        className="w-full bg-gradient-to-r from-cyan-500 to-purple-600 rounded-lg p-3 font-semibold text-white hover:scale-105 transition"
      >
        Train Model
      </button>
    </div>
  );
}