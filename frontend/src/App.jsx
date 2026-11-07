import { useRef, useState } from "react";

// Un service par approche (ports du docker-compose)
const APIS = {
  cnn: import.meta.env.VITE_API_CNN || "http://localhost:8001",
  vit: import.meta.env.VITE_API_VIT || "http://localhost:8002",
  ocr: import.meta.env.VITE_API_OCR || "http://localhost:8003",
};

export default function App() {
  const [model, setModel] = useState("cnn");
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const video = useRef(null);

  async function send(blob) {
    setError(null);
    setResult(null);
    setPreview(URL.createObjectURL(blob));
    const body = new FormData();
    body.append("file", blob, "document.png");
    try {
      const r = await fetch(`${APIS[model]}/predict?model=${model}`, { method: "POST", body });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      setResult(await r.json());
    } catch (e) {
      setError(String(e.message || e));
    }
  }

  async function startWebcam() {
    video.current.srcObject = await navigator.mediaDevices.getUserMedia({ video: true });
  }

  function snapshot() {
    const v = video.current;
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext("2d").drawImage(v, 0, 0);
    c.toBlob(send, "image/png");
  }

  return (
    <main style={{ maxWidth: 720, margin: "2rem auto", fontFamily: "system-ui" }}>
      <h1>DocGuard-AI</h1>
      <label>
        Modele :{" "}
        <select value={model} onChange={(e) => setModel(e.target.value)}>
          <option value="cnn">CNN (Grad-CAM)</option>
          <option value="vit">Vision Transformer</option>
          <option value="ocr">OCR + texte</option>
        </select>
      </label>
      <p>
        <input type="file" accept="image/*" onChange={(e) => e.target.files[0] && send(e.target.files[0])} />
      </p>
      <p>
        <button onClick={startWebcam}>Activer la webcam</button>{" "}
        <button onClick={snapshot}>Analyser la capture</button>
      </p>
      <video ref={video} autoPlay playsInline width={320} />
      {preview && <img src={preview} alt="document" style={{ maxWidth: "100%" }} />}
      {error && <p style={{ color: "crimson" }}>{error}</p>}
      {result && (
        <section>
          <h2>{result.label === "scam" ? "Document suspect" : "Document sain"} ({(result.score * 100).toFixed(1)} %)</h2>
          <small>{result.model}</small>
          {result.heatmap && <img src={`data:image/png;base64,${result.heatmap}`} alt="heatmap" style={{ maxWidth: "100%" }} />}
          {result.ocr && (
            <ul>
              {result.ocr.map((o, i) => (
                <li key={i} style={{ color: o.suspect ? "crimson" : "inherit" }}>{o.text}</li>
              ))}
            </ul>
          )}
        </section>
      )}
    </main>
  );
}
