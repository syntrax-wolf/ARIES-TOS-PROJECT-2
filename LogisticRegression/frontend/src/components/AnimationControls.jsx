export default function AnimationControls({
  isPlaying,
  onPlay,
  onPause,
  onRestart,
}) {
  return (
    <div className="flex gap-4 mt-6">

      <button
        onClick={onPlay}
        className="bg-green-600 hover:bg-green-700 px-5 py-2 rounded-lg text-white font-semibold"
      >
        ▶ Play
      </button>

      <button
        onClick={onPause}
        className="bg-yellow-500 hover:bg-yellow-600 px-5 py-2 rounded-lg text-white font-semibold"
      >
        ⏸ Pause
      </button>

      <button
        onClick={onRestart}
        className="bg-cyan-600 hover:bg-cyan-700 px-5 py-2 rounded-lg text-white font-semibold"
      >
        🔄 Restart
      </button>

      <div className="flex items-center text-white ml-4">
        Status:
        <span
          className={`ml-2 font-bold ${
            isPlaying ? "text-green-400" : "text-red-400"
          }`}
        >
          {isPlaying ? "Playing" : "Paused"}
        </span>
      </div>

    </div>
  );
}