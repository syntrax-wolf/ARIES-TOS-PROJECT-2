import Plot from "react-plotly.js";

export default function LossChart({ history, currentIndex }) {
  if (history.length === 0) {
    return (
      <div className="mt-6 text-white text-center">
        No training history yet.
      </div>
    );
  }

  const visibleHistory = history.slice(0, currentIndex + 1);

  const epochs = visibleHistory.map((step) => step.epoch);
  const losses = visibleHistory.map((step) => step.loss);

  return (
    <div className="mt-6">
      <Plot
        data={[
          {
            x: epochs,
            y: losses,
            type: "scatter",
            mode: "lines",
            name: "Loss",
            line: {
              color: "#38bdf8",
              width: 3,
            },
          },
        ]}
        layout={{
          title: {
            text: "Training Loss",
            font: { color: "white" },
          },
          paper_bgcolor: "#0f172a",
          plot_bgcolor: "#0f172a",

          font: {
            color: "white",
          },

          xaxis: {
            title: "Epoch",
            gridcolor: "#334155",
            zerolinecolor: "#334155",
          },

          yaxis: {
            title: "Loss",
            gridcolor: "#334155",
            zerolinecolor: "#334155",
          },

          margin: {
            l: 60,
            r: 30,
            t: 50,
            b: 60,
          },

          height: 350,
        }}
        config={{
          responsive: true,
          displayModeBar: false,
        }}
        style={{
          width: "100%",
        }}
      />
    </div>
  );
}