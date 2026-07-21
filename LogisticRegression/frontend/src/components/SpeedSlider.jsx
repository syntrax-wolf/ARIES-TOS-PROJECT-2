export default function SpeedSlider({
  animationSpeed,
  setAnimationSpeed,
}) {
  const speeds = [
    { label: "0.5x", value: 100 },
    { label: "1x", value: 50 },
    { label: "2x", value: 25 },
    { label: "5x", value: 10 },
  ];

  return (
    <div className="mt-6 bg-slate-900 border border-slate-700 rounded-xl p-4">
      <h2 className="text-white font-semibold text-lg mb-4">
        Animation Speed
      </h2>

      <div className="flex gap-3">
        {speeds.map((speed) => (
          <button
            key={speed.value}
            onClick={() => setAnimationSpeed(speed.value)}
            className={`px-4 py-2 rounded-lg transition-all font-semibold
              ${
                animationSpeed === speed.value
                  ? "bg-cyan-500 text-white"
                  : "bg-slate-700 text-slate-200 hover:bg-slate-600"
              }`}
          >
            {speed.label}
          </button>
        ))}
      </div>
    </div>
  );
}