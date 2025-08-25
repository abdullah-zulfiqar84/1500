self.languagePluginUrl = 'https://cdn.jsdelivr.net/pyodide/v0.26.2/full/';
importScripts('https://cdn.jsdelivr.net/pyodide/v0.26.2/full/pyodide.js');

let pyodideReady = (async () => {
  const pyodide = await loadPyodide({ indexURL: self.languagePluginUrl });
  const files = [
    "qtcore_shim.py","utils_geom.py","vertex.py","edge.py","periphery.py","graph.py","bridge.py"
  ];
  for (const f of files) {
    const res = await fetch(`/py/${f}`);
    const txt = await res.text();
    pyodide.FS.writeFile(f, txt);
  }
  await pyodide.runPythonAsync("import bridge");
  return pyodide;
})();

self.onmessage = async (e) => {
  const { cmd, payload, rid } = e.data || {};
  const reply = (obj) => self.postMessage({ rid, ...obj });
  try {
    const pyodide = await pyodideReady;
    switch (cmd) {
      case "start":
        reply({ ok: true, data: await pyodide.runPythonAsync(`bridge.start_basic(${payload?.n || 3})`) }); break;
      case "add_random":
        reply({ ok: true, data: await pyodide.runPythonAsync("bridge.add_random()") }); break;
      case "add_by_selection":
        reply({ ok: true, data: await pyodide.runPythonAsync(`bridge.add_by_selection(${payload.a}, ${payload.b})`) }); break;
      case "redraw":
        reply({ ok: true, data: await pyodide.runPythonAsync("bridge.redraw()") }); break;
      case "get_state":
        reply({ ok: true, data: await pyodide.runPythonAsync("bridge.get_state()") }); break;
      case "save":
        reply({ ok: true, data: await pyodide.runPythonAsync("bridge.save_json_string()") }); break;
      case "load":
        pyodide.globals.set("___in", payload.json);
        reply({ ok: true, data: await pyodide.runPythonAsync("bridge.load_json_string(___in)") }); break;
      case "go_to":
        reply({ ok: true, data: await pyodide.runPythonAsync(`bridge.go_to(${payload.m})`) }); break;
      case "set_auto_tutte":
        reply({ ok: true, data: await pyodide.runPythonAsync(`bridge.set_auto_tutte(${Boolean(payload.on)})`) }); break;
      case "set_finalize":
        reply({ ok: true, data: await pyodide.runPythonAsync(`bridge.set_finalize(${Boolean(payload.on)})`) }); break;
      case "set_label_mode":
        reply({ ok: true, data: await pyodide.runPythonAsync(`bridge.set_label_mode(${payload.mode})`) }); break;
      case "declutter":
        reply({ ok: true, data: await pyodide.runPythonAsync(`bridge.declutter(${payload?.intensity ?? 1.0})`) }); break;
      default:
        reply({ ok: false, error: `Unknown cmd ${cmd}` });
    }
  } catch (err) {
    reply({ ok: false, error: String(err) });
  }
};
