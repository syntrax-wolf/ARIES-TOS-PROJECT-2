import { Suspense, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { Grid as DreiGrid, OrbitControls } from "@react-three/drei";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { PAD, type CnnForward } from "../lib/cnn";
import { GridLayer } from "./GridLayer";
import { NodeColumn3D } from "./NodeColumn3D";
import { StageLabel } from "./StageLabel";
import { Connector } from "./Connector";
import { KernelProbe } from "./KernelProbe";
import { CameraRig, TOP_OFFSET } from "./CameraRig";
import { activationColor, classThreeColor, reshapeToGrid, STAGE_GAP, STAGE_LABELS, toSingleChannelGrid } from "./cnnGeometry";

const LAST = STAGE_LABELS.length - 1;
const MID_Z = (-STAGE_GAP * LAST) / 2;

// Matches CameraRig's initial (top-view, stage-0) target + offset, so the
// very first frame doesn't flash a different framing before the rig takes over.
const INITIAL_CAMERA_POS: [number, number, number] = [TOP_OFFSET.x, 2 + TOP_OFFSET.y, TOP_OFFSET.z];

function z(i: number): number {
  return -i * STAGE_GAP;
}

interface CnnScene3DProps {
  fwd: CnnForward;
  stage: number;
}

export function CnnScene3D({ fwd, stage }: CnnScene3DProps) {
  const flatGrid = reshapeToGrid(fwd.pool2.flat(2), 12);
  const maxDense = Math.max(...fwd.dense1, 1e-9);
  const controlsRef = useRef<OrbitControlsImpl>(null);

  return (
    <Canvas dpr={[1, 1.8]} camera={{ position: INITIAL_CAMERA_POS, fov: 45, near: 0.5, far: 160 }} gl={{ antialias: true }}>
      <color attach="background" args={["#f7f7f1"]} />
      <fog attach="fog" args={["#f7f7f1", 16, 55]} />
      <ambientLight intensity={0.7} />
      <directionalLight position={[24, 30, 12]} intensity={1.1} />
      <directionalLight position={[-18, 12, -24]} intensity={0.3} color="#dce8ff" />
      <hemisphereLight args={["#eef4ef", "#ece2cd", 0.45]} />

      <Suspense fallback={null}>
        <DreiGrid
          position={[0, -6, MID_Z]}
          args={[120, 120]}
          cellSize={2}
          cellColor="#e2e4d8"
          sectionSize={14}
          sectionColor="#c9cdbd"
          fadeDistance={100}
          infiniteGrid
        />

        {STAGE_LABELS.map((label, i) => (
          <StageLabel key={i} y={-6.8} z={z(i)} label={label} active={stage >= i} />
        ))}

        {STAGE_LABELS.slice(0, -1).map((_, i) => (
          <Connector key={i} fromZ={z(i)} toZ={z(i + 1)} revealed={stage > i} />
        ))}

        <GridLayer grid={toSingleChannelGrid(fwd.input)} stageZ={z(0)} revealed={stage >= 0} variant="input" />
        <GridLayer grid={fwd.padded1} stageZ={z(1)} revealed={stage >= 1} variant="pad" innerVariant="input" pad={PAD} />
        <GridLayer grid={fwd.conv1} stageZ={z(2)} revealed={stage >= 2} variant="activation" />
        <GridLayer grid={fwd.pool1} stageZ={z(3)} revealed={stage >= 3} variant="activation" />
        <GridLayer grid={fwd.padded2} stageZ={z(4)} revealed={stage >= 4} variant="pad" innerVariant="activation" pad={PAD} />
        <GridLayer grid={fwd.conv2} stageZ={z(5)} revealed={stage >= 5} variant="activation" />
        <GridLayer grid={fwd.pool2} stageZ={z(6)} revealed={stage >= 6} variant="activation" />
        <GridLayer grid={flatGrid} stageZ={z(7)} revealed={stage >= 7} variant="activation" />

        <NodeColumn3D
          values={fwd.dense1}
          stageZ={z(8)}
          revealed={stage >= 8}
          spacing={1}
          nodeSpec={(v) => ({ radius: 0.32 + (v / maxDense) * 0.28, color: activationColor(v / maxDense), opacity: 1 })}
        />

        <NodeColumn3D
          values={fwd.probs}
          stageZ={z(9)}
          revealed={stage >= 9}
          spacing={1.7}
          nodeSpec={(v, i) => ({ radius: 0.5 + v * 0.9, color: classThreeColor(i), opacity: 0.35 + v * 0.65 })}
          labelFn={(i) => String(i)}
          highlightIndex={fwd.predicted}
          captionFn={(i) => (i === fwd.predicted ? `guesses ${fwd.predicted}` : null)}
        />

        {stage === 2 && <KernelProbe z={z(1)} footprint={7} />}
        {stage === 3 && <KernelProbe z={z(2)} footprint={5} size={1.1} />}
        {stage === 5 && <KernelProbe z={z(4)} footprint={5} />}
        {stage === 6 && <KernelProbe z={z(5)} footprint={4} size={0.9} />}
      </Suspense>

      <CameraRig stage={stage} controlsRef={controlsRef} />

      <OrbitControls
        ref={controlsRef}
        makeDefault
        enablePan={false}
        minDistance={9}
        maxDistance={42}
        target={[0, 2, 0]}
      />
    </Canvas>
  );
}
