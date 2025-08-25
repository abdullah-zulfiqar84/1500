export type V = { index: number; x: number; y: number; color: number; visible: boolean };
export type E = { u: number; v: number; visible: boolean };

export interface GraphMeta {
  total_vertices: number;
  edges: number;
  periphery_size: number;
}

export interface GState { 
  vertices: V[]; 
  edges: E[]; 
  periphery: number[]; 
  meta: GraphMeta;
  labelMode?: number;
}

export interface GraphData {
  vertices: V[];
  edges: E[];
  periphery: number[];
  meta: GraphMeta;
  labelMode?: number;
}

export interface AddVertexInfo {
  index: number;
  spawn_pos?: { x: number; y: number };
  final_pos?: { x: number; y: number };
}

export interface AddVertexResult {
  ok: boolean;
  index: number;
  info?: AddVertexInfo;
  state: GState;
}

export interface DeclutterResult {
  changed: boolean;
  state: GState;
}

export interface WorkerMessage {
  cmd: string;
  payload?: unknown;
  rid: number;
}

export interface WorkerResponse {
  rid: number;
  ok: boolean;
  data?: unknown;
  error?: unknown;
}