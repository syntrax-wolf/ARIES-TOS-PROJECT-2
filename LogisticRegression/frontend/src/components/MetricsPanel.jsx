export default function MetricsPanel({
  currentStep,
  learningRate,
  sampleCount,
  featureCount,
  isPlaying,
  totalEpochs,
}) {
  const cards = [
    {
      title: "Epoch",
      value: currentStep ? currentStep.epoch : "-",
      color: "text-cyan-400",
    },
    {
      title: "Loss",
      value: currentStep
        ? currentStep.loss.toFixed(5)
        : "-",
      color: "text-green-400",
    },
    {
      title: "Accuracy",
      value: currentStep
        ? `${(currentStep.accuracy * 100).toFixed(2)}%`
        : "-",
      color: "text-purple-400",
    },
    {
      title: "Learning Rate",
      value: learningRate,
      color: "text-yellow-400",
    },
    {
      title: "Samples",
      value: sampleCount,
      color: "text-pink-400",
    },
    {
      title: "Features",
      value: featureCount,
      color: "text-orange-400",
    },
    {
      title: "Status",
      value: isPlaying ? "Playing" : "Paused",
      color: isPlaying
        ? "text-green-400"
        : "text-red-400",
    },
    {
      title: "Epochs",
      value: totalEpochs,
      color: "text-cyan-300",
    },
  ];

  return (
    <div className="mt-6">

      <h2 className="text-white text-2xl font-bold mb-4">
        Training Analytics
      </h2>

      <div className="grid grid-cols-4 gap-4">

        {cards.map((card) => (
          <div
            key={card.title}
            className="bg-slate-900 border border-slate-700 rounded-xl p-5"
          >
            <p className="text-slate-400 text-sm">
              {card.title}
            </p>

            <p className={`text-2xl font-bold mt-2 ${card.color}`}>
              {card.value}
            </p>
          </div>
        ))}

      </div>

    </div>
  );
}