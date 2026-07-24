import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import { easeOutBack } from "./cnnGeometry";

const POP_DURATION = 0.4;
const MAX_STAGGER = 0.35;

interface NodeSpec {
  radius: number;
  color: THREE.Color;
  opacity: number;
}

interface NodeColumn3DProps {
  values: number[];
  stageZ: number;
  revealed: boolean;
  spacing?: number;
  nodeSpec: (value: number, index: number) => NodeSpec;
  labelFn?: (index: number) => string;
  highlightIndex?: number;
  captionFn?: (index: number) => string | null;
}

function Node({ index, x, y, z, spec, revealed, label, highlighted, caption }: {
  index: number;
  x: number;
  y: number;
  z: number;
  spec: NodeSpec;
  revealed: boolean;
  label?: string;
  highlighted: boolean;
  caption?: string | null;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const revealTimeRef = useRef<number | null>(null);
  const doneRef = useRef(false);

  useEffect(() => {
    if (revealed && revealTimeRef.current === null) revealTimeRef.current = performance.now();
  }, [revealed]);

  useFrame(() => {
    const group = groupRef.current;
    if (!group) return;
    if (!revealed) {
      group.scale.setScalar(0.0001);
      return;
    }
    if (doneRef.current) return;
    const started = revealTimeRef.current ?? performance.now();
    const stagger = (index / 10) * MAX_STAGGER;
    const elapsed = (performance.now() - started) / 1000;
    const local = Math.max(0, Math.min(1, (elapsed - stagger) / POP_DURATION));
    const eased = easeOutBack(local);
    group.scale.setScalar(Math.max(eased, 0.0001));
    if (local >= 1) doneRef.current = true;
  });

  return (
    <group ref={groupRef} position={[x, y, z]} scale={0.0001}>
      <mesh>
        <sphereGeometry args={[spec.radius, 24, 24]} />
        <meshStandardMaterial color={spec.color} transparent opacity={spec.opacity} roughness={0.4} metalness={0.1} />
      </mesh>
      {highlighted && revealed && (
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[spec.radius + 0.5, 0.06, 12, 48]} />
          <meshBasicMaterial color="#2f6b4f" />
        </mesh>
      )}
      {label && (
        <Html center distanceFactor={14} style={{ pointerEvents: "none" }}>
          <span style={{ fontSize: 13, fontWeight: 700, color: "#ffffff", textShadow: "0 1px 2px rgba(0,0,0,0.35)" }}>{label}</span>
        </Html>
      )}
      {caption && (
        <Html center distanceFactor={14} position={[0, -spec.radius - 0.9, 0]} style={{ pointerEvents: "none" }}>
          <span
            style={{
              fontSize: 13,
              fontWeight: 700,
              color: "#2f6b4f",
              whiteSpace: "nowrap",
              background: "rgba(255,255,255,0.85)",
              padding: "2px 8px",
              borderRadius: 999,
            }}
          >
            {caption}
          </span>
        </Html>
      )}
    </group>
  );
}

export function NodeColumn3D({ values, stageZ, revealed, spacing = 1.1, nodeSpec, labelFn, highlightIndex, captionFn }: NodeColumn3DProps) {
  const totalH = (values.length - 1) * spacing;
  const positions = useMemo(
    () => values.map((_, i) => totalH / 2 - i * spacing),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [values.length, spacing, totalH]
  );

  return (
    <group>
      {values.map((v, i) => (
        <Node
          key={i}
          index={i}
          x={0}
          y={positions[i]}
          z={stageZ}
          spec={nodeSpec(v, i)}
          revealed={revealed}
          label={labelFn?.(i)}
          highlighted={highlightIndex === i}
          caption={captionFn?.(i) ?? null}
        />
      ))}
    </group>
  );
}
