export class PyWorkerBridge {
  private w: Worker;
  private nextRid = 1;
  private inflight = new Map<number, (data: any, ok: boolean)=>void>();

  constructor() {
    this.w = new Worker('/py-worker.js');
    this.w.onmessage = (e: MessageEvent) => {
      const { rid, ok, data, error } = e.data || {};
      const cb = this.inflight.get(rid);
      if (cb) {
        this.inflight.delete(rid);
        cb(ok ? data : error, !!ok);
      }
    };
  }

  call<T = any>(cmd: string, payload?: any): Promise<T> {
    const rid = this.nextRid++;
    return new Promise<T>((resolve, reject) => {
      this.inflight.set(rid, (res, ok) => ok ? resolve(res as T) : reject(res));
      this.w.postMessage({ cmd, payload, rid });
    });
  }
}