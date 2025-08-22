export type V = { index: number; x: number; y: number; color: number; visible: boolean };
export type E = { u: number; v: number; visible: boolean };
export type GState = { vertices: V[]; edges: E[]; periphery: number[]; meta: any };