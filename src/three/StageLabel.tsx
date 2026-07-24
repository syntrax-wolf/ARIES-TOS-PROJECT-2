import { Html } from "@react-three/drei";

interface StageLabelProps {
  x?: number;
  y: number;
  z: number;
  label: string;
  active: boolean;
}

export function StageLabel({ x = 0, y, z, label, active }: StageLabelProps) {
  return (
    <Html center position={[x, y, z]} distanceFactor={16} style={{ pointerEvents: "none" }}>
      <span
        style={{
          fontSize: 13,
          fontWeight: 600,
          letterSpacing: "0.02em",
          whiteSpace: "nowrap",
          color: active ? "#2f6b4f" : "#b7bdad",
          transition: "color 0.4s ease",
        }}
      >
        {label}
      </span>
    </Html>
  );
}
