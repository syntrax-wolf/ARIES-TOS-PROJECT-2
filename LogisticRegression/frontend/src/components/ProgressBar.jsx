export default function ProgressBar({
  currentIndex,
  totalEpochs,
  isPlaying,
}) {
  const progress =
    totalEpochs > 1
      ? (currentIndex / (totalEpochs - 1)) * 100
      : 0;

  return (
    <div className="mt-6 bg-slate-900 border border-slate-800 rounded-xl p-5">

      <div className="flex justify-between items-center mb-3">
        <h2 className="text-white font-semibold">
          Training Progress
        </h2>

        <span
          className={`text-sm font-medium ${
            isPlaying
              ? "text-green-400"
              : "text-slate-400"
          }`}
        >
          {isPlaying ? "● Training..." : "● Paused"}
        </span>
      </div>

      <div className="w-full bg-slate-800 rounded-full h-3 overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-cyan-500 to-purple-600 transition-all duration-200"
          style={{
            width: `${progress}%`,
          }}
        />
      </div>

      <div className="flex justify-between mt-3 text-sm text-slate-400">
        <span>
          Epoch {currentIndex}
        </span>

        <span>
          {progress.toFixed(1)}%
        </span>

        <span>
          {Math.max(totalEpochs - 1, 0)} Epochs
        </span>
      </div>

    </div>
  );
}