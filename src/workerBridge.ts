import { WorkerMessage, WorkerResponse } from './types'

export class PyWorkerBridge {
  private w: Worker;
  private nextRid = 1;
  private inflight = new Map<number, (data: unknown, ok: boolean)=>void>();

  constructor() {
    this.w = new Worker('/py-worker.js');
    this.w.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const { rid, ok, data, error } = e.data || {};
      const cb = this.inflight.get(rid);
      if (cb) {
        this.inflight.delete(rid);
        cb(ok ? data : error, !!ok);
      }
    };
  }

  call<T = unknown>(cmd: string, payload?: unknown): Promise<T> {
    const rid = this.nextRid++;
    return new Promise<T>((resolve, reject) => {
      this.inflight.set(rid, (res, ok) => ok ? resolve(res as T) : reject(res));
      this.w.postMessage({ cmd, payload, rid } as WorkerMessage);
    });
  }
}