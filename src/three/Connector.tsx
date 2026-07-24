import { Line } from "@react-three/drei";

interface ConnectorProps {
  fromZ: number;
  toZ: number;
  y?: number;
  revealed: boolean;
}

export function Connector({ fromZ, toZ, y = 0, revealed }: ConnectorProps) {
  if (!revealed) return null;
  return (
    <Line
      points={[
        [0, y, fromZ],
        [0, y, toZ],
      ]}
      color="#b9c2b2"
      lineWidth={1.5}
      dashed
      dashSize={0.4}
      gapSize={0.35}
      transparent
      opacity={0.7}
    />
  );
}
