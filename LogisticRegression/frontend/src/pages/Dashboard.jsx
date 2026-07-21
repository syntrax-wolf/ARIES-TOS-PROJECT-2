import { useEffect, useRef, useState } from "react";
import axios from "axios";

import Navbar from "../components/Navbar";
import Sidebar from "../components/Sidebar";
import DatasetPlot from "../components/DatasetPlot";
import AnimationControls from "../components/AnimationControls";
import ProgressBar from "../components/ProgressBar";
import SpeedSlider from "../components/SpeedSlider";
import EpochSlider from "../components/EpochSlider";
import LossChart from "../components/LossChart";
import MetricsPanel from "../components/MetricsPanel";

export default function Dashboard() {
  const [data, setData] = useState([]);
  const [history, setHistory] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);

  const [isPlaying, setIsPlaying] = useState(false);
  const [animationSpeed, setAnimationSpeed] = useState(50);

  const [learningRate, setLearningRate] = useState(0.1);

  const intervalRef = useRef(null);

  const currentStep =
    history.length > 0 ? history[currentIndex] : null;

  // ==========================================
  // Load Default Dataset
  // ==========================================

  useEffect(() => {
    axios
      .get("http://127.0.0.1:5000/dataset?name=iris")
      .then((res) => setData(res.data))
      .catch(console.error);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  // ==========================================
  // Animation
  // ==========================================

  useEffect(() => {
    if (!isPlaying) return;
    if (history.length === 0) return;

    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    intervalRef.current = setInterval(() => {
      setCurrentIndex((prev) => {
        if (prev >= history.length - 1) {
          clearInterval(intervalRef.current);
          setIsPlaying(false);
          return prev;
        }

        return prev + 1;
      });
    }, animationSpeed);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isPlaying, history, animationSpeed]);

  // ==========================================
  // Train Model
  // ==========================================

  const trainModel = ({
    dataset,
    learningRate,
    epochs,
  }) => {
    setLearningRate(learningRate);

    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    setHistory([]);
    setCurrentIndex(0);
    setIsPlaying(false);

    axios
      .get(`http://127.0.0.1:5000/dataset?name=${dataset}`)
      .then((res) => {
        setData(res.data);

        return axios.post("http://127.0.0.1:5000/train", {
          dataset,
          learning_rate: learningRate,
          epochs,
        });
      })
      .then((res) => {
        setHistory(res.data);
        setCurrentIndex(0);
        setIsPlaying(true);
      })
      .catch(console.error);
  };

  // ==========================================
  // Controls
  // ==========================================

  const handlePlay = () => {
    if (history.length > 0) {
      setIsPlaying(true);
    }
  };

  const handlePause = () => {
    setIsPlaying(false);
  };

  const handleRestart = () => {
    if (!history.length) return;

    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    setCurrentIndex(0);
    setIsPlaying(true);
  };

  return (
    <div className="min-h-screen bg-slate-950">
      <Navbar />

      <div className="flex">
        <Sidebar onTrain={trainModel} />

        <div className="flex-1 p-6">

          {/* Decision Boundary */}

          <div className="rounded-2xl bg-slate-900 border border-slate-800 h-[650px] p-2 flex">
            <DatasetPlot
              data={data}
              currentStep={currentStep}
            />
          </div>

          {/* Animation Controls */}

          <AnimationControls
            isPlaying={isPlaying}
            onPlay={handlePlay}
            onPause={handlePause}
            onRestart={handleRestart}
          />

          {/* Progress Bar */}

          <ProgressBar
            currentIndex={currentIndex}
            totalEpochs={history.length}
            isPlaying={isPlaying}
          />

          {/* Speed Controls */}

          <SpeedSlider
            animationSpeed={animationSpeed}
            setAnimationSpeed={setAnimationSpeed}
          />

          {/* Epoch Slider */}

          <EpochSlider
            currentIndex={currentIndex}
            maxIndex={Math.max(history.length - 1, 0)}
            setCurrentIndex={setCurrentIndex}
            setIsPlaying={setIsPlaying}
          />

          {/* Loss Graph */}

          <LossChart
            history={history}
            currentIndex={currentIndex}
          />

          {/* Metrics */}

          <MetricsPanel
            currentStep={currentStep}
            learningRate={learningRate}
            sampleCount={data.length}
            featureCount={2}
            isPlaying={isPlaying}
            totalEpochs={history.length}
          />

        </div>
      </div>
    </div>
  );
}