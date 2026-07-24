import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

interface KernelProbeProps {
  z: number;
  footprint?: number;
  size?: number;
}

/** A wandering wireframe cube standing in for the kernel sliding across the previous layer during a convolve/pool transition — pure flourish, not literally tied to the real sliding window. */
export function KernelProbe({ z, footprint = 6, size = 1.4 }: KernelProbeProps) {
  const ref = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    const mesh = ref.current;
    if (!mesh) return;
    const t = clock.getElapsedTime();
    mesh.position.set(Math.sin(t * 2.1) * footprint * 0.42, Math.cos(t * 1.7) * 1.6, z + Math.sin(t * 1.3) * 0.7);
    mesh.rotation.set(t * 0.6, t * 0.4, 0);
  });

  return (
    <mesh ref={ref}>
      <boxGeometry args={[size, size, size]} />
      <meshBasicMaterial color="#2f6b4f" wireframe transparent opacity={0.6} />
    </mesh>
  );
}
