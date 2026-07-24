import { useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { FLATTEN_STAGE_INDEX, STAGE_GAP } from "./cnnGeometry";

const FOLLOW_LAMBDA = 3.2;
const BLEND_LAMBDA = 1.8;
const TARGET_Y = 2;

// Steep, mostly-overhead angle for reading the 2D feature maps while
// conv/pool "filters" are being applied to the previous layer.
export const TOP_OFFSET = new THREE.Vector3(5, 21, 10);
// Level, side-on angle for reading the flatten/dense/output columns once the
// grids have collapsed into a vector.
export const SIDE_OFFSET = new THREE.Vector3(21, 7, 3);

interface CameraRigProps {
  stage: number;
  controlsRef: React.RefObject<OrbitControlsImpl | null>;
}

/**
 * Drives the camera down the -Z tunnel to track the stage currently being
 * revealed, and swings between a top-down shot (while convolution/pooling
 * "filters" are at work) and a side profile (once flatten/dense/output take
 * over) — both damped so the moves read as camera pans, not cuts.
 */
export function CameraRig({ stage, controlsRef }: CameraRigProps) {
  const { camera } = useThree();
  const trackedZ = useRef(0);
  const viewBlend = useRef(0); // 0 = top view, 1 = side view

  useFrame((_, delta) => {
    const controls = controlsRef.current;
    if (!controls) return;

    const desiredZ = -Math.max(stage, 0) * STAGE_GAP;
    trackedZ.current = THREE.MathUtils.damp(trackedZ.current, desiredZ, FOLLOW_LAMBDA, delta);

    const desiredBlend = stage >= FLATTEN_STAGE_INDEX ? 1 : 0;
    viewBlend.current = THREE.MathUtils.damp(viewBlend.current, desiredBlend, BLEND_LAMBDA, delta);

    const target = new THREE.Vector3(0, TARGET_Y, trackedZ.current);
    const offset = TOP_OFFSET.clone().lerp(SIDE_OFFSET, viewBlend.current);

    camera.position.copy(target).add(offset);
    controls.target.copy(target);
    controls.update();
  });

  return null;
}
