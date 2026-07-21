import Plot from "react-plotly.js";

export default function DatasetPlot({ data, currentStep }) {
  const colors = ["#22d3ee", "#a855f7", "#ec4899"];

  const traces = [];

  // ===========================
  // Probability Background
  // ===========================
  if (currentStep && data.length > 0) {
    const w = currentStep.weights;
    const b = currentStep.bias;

    const xMin = Math.min(...data.map((p) => p.x)) - 0.5;
    const xMax = Math.max(...data.map((p) => p.x)) + 0.5;

    const yMin = Math.min(...data.map((p) => p.y)) - 0.5;
    const yMax = Math.max(...data.map((p) => p.y)) + 0.5;

    const resolution = 60;

    const xs = [];
    const ys = [];
    const z = [];

    for (let i = 0; i < resolution; i++) {
      xs.push(xMin + (i * (xMax - xMin)) / (resolution - 1));
      ys.push(yMin + (i * (yMax - yMin)) / (resolution - 1));
    }

    for (let j = 0; j < resolution; j++) {
      const row = [];

      for (let i = 0; i < resolution; i++) {
        const x = xs[i];
        const y = ys[j];

        const value = w[0] * x + w[1] * y + b;

        const probability = 1 / (1 + Math.exp(-value));

        row.push(probability);
      }

      z.push(row);
    }

    traces.push({
      x: xs,
      y: ys,
      z: z,
      type: "contour",

      contours: {
        coloring: "heatmap",
        showlines: false,
      },

      colorscale: [
        [0, "#2563eb"],
        [0.5, "#ffffff"],
        [1, "#dc2626"],
      ],

      opacity: 0.45,

      showscale: true,

      hoverinfo: "skip",
    });
  }

  // ===========================
  // Dataset Points
  // ===========================
  traces.push({
    x: data.map((p) => p.x),
    y: data.map((p) => p.y),

    mode: "markers",
    type: "scatter",

    marker: {
      size: 10,
      color: data.map((p) => colors[p.class]),
      line: {
        color: "white",
        width: 1,
      },
    },

    name: "Dataset",
  });

  // ===========================
  // Decision Boundary
  // ===========================
  if (currentStep) {
    const w = currentStep.weights;
    const b = currentStep.bias;

    if (Math.abs(w[1]) > 1e-8) {
      const xMin = Math.min(...data.map((p) => p.x));
      const xMax = Math.max(...data.map((p) => p.x));

      const xLine = [xMin, xMax];

      const yLine = xLine.map(
        (x) => -(w[0] * x + b) / w[1]
      );

      traces.push({
        x: xLine,
        y: yLine,

        mode: "lines",
        type: "scatter",

        line: {
          color: "#ffffff",
          width: 4,
        },

        name: "Decision Boundary",
      });
    }
  }

  return (
    <Plot
      data={traces}
      layout={{
        autosize: true,

        paper_bgcolor: "#0f172a",
        plot_bgcolor: "#0f172a",

        font: {
          color: "white",
        },

        margin: {
          l: 60,
          r: 20,
          t: 20,
          b: 60,
        },

        xaxis: {
          title: "Sepal Length",
          gridcolor: "#334155",
          zeroline: false,
        },

        yaxis: {
          title: "Sepal Width",
          gridcolor: "#334155",
          zeroline: false,
        },

        showlegend: false,
      }}
      style={{
        width: "100%",
        height: "100%",
      }}
      useResizeHandler={true}
      config={{
        responsive: true,
        displayModeBar: false,
      }}
    />
  );
}