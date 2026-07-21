export default function EpochSlider({
  currentIndex,
  maxIndex,
  setCurrentIndex,
  setIsPlaying,
}) {
  const handleChange = (e) => {
    setIsPlaying(false);
    setCurrentIndex(Number(e.target.value));
  };

  return (
    <div className="mt-6 bg-slate-900 border border-slate-700 rounded-xl p-4">
      <div className="flex justify-between items-center mb-3">
        <h2 className="text-white text-lg font-semibold">
          Epoch Timeline
        </h2>

        <span className="text-cyan-400 font-bold">
          Epoch {currentIndex}
        </span>
      </div>

      <input
        type="range"
        min="0"
        max={Math.max(maxIndex, 0)}
        value={currentIndex}
        onChange={handleChange}
        className="w-full accent-cyan-500 cursor-pointer"
      />

      <div className="flex justify-between text-slate-400 text-sm mt-2">
        <span>0</span>
        <span>{maxIndex}</span>
      </div>
    </div>
  );
}