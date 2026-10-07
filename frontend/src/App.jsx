import { useState } from "react";
import "./App.css";

function App() {
  const [video, setVideo] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleVideoChange = (event) => {
    const file = event.target.files[0];

    if (!file) return;

    setVideo(file);
    setResult(null);
    setError("");
  };

  const predictShot = async () => {
    if (!video) return;

    setLoading(true);
    setResult(null);
    setError("");

    const formData = new FormData();
    formData.append("file", video);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Backend error: ${response.status}`);
      }

      const data = await response.json();

      setResult(data);
    } catch (err) {
      console.error(err);
      setError(
        "Could not connect to the backend. Make sure FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="navbar">
        <div className="logo">
          <span className="logo-mark">🏸</span>
          <span>Badminton AI</span>
        </div>

        <nav>
          <a href="#home">Home</a>
          <a href="#analyze">Analyze</a>
          <a href="#about">About</a>
        </nav>
      </header>

      <main>
        <section className="hero" id="home">
          <div className="hero-content">
            <p className="eyebrow">AI-POWERED SPORTS ANALYTICS</p>

            <h1>
              Understand Every
              <span> Badminton Shot.</span>
            </h1>

            <p className="hero-description">
              Upload a badminton video and let our CNN-LSTM model recognize
              the player's stroke automatically.
            </p>

            <a href="#analyze" className="primary-button">
              Analyze Video
            </a>
          </div>

          <div className="hero-card">
            <div className="court">
              <div className="net"></div>
              <div className="player player-one"></div>
              <div className="player player-two"></div>
              <div className="shuttle">●</div>
            </div>
          </div>
        </section>

        <section className="analyze-section" id="analyze">
          <div className="section-heading">
            <p className="eyebrow">SHOT RECOGNITION</p>
            <h2>Analyze your video</h2>
            <p>
              Upload a badminton clip to classify the player's shot.
            </p>
          </div>

          <div className="upload-card">
            <div className="upload-icon">↑</div>

            <h3>
              {video ? video.name : "Upload badminton video"}
            </h3>

            <p>
              {video
                ? `${(video.size / (1024 * 1024)).toFixed(2)} MB`
                : "MP4, AVI, MOV or MKV"}
            </p>

            <label className="upload-button">
              Choose Video
              <input
                type="file"
                accept="video/*"
                onChange={handleVideoChange}
              />
            </label>

            {video && (
              <button
                className="analyze-button"
                onClick={predictShot}
                disabled={loading}
              >
                {loading ? "Analyzing..." : "Predict Shot"}
              </button>
            )}

            {error && <p className="error-message">{error}</p>}

            {result && (
              <div className="result-card">
                <p className="result-label">PREDICTION</p>

                <h2>{result.prediction}</h2>

                <p className="confidence">
                  Confidence:{" "}
                  <strong>
                    {(result.confidence * 100).toFixed(1)}%
                  </strong>
                </p>

                {result.probabilities && (
                  <div className="probabilities">
                    {Object.entries(result.probabilities).map(
                      ([shot, probability]) => (
                        <div className="probability-row" key={shot}>
                          <span>{shot}</span>

                          <span>
                            {(probability * 100).toFixed(1)}%
                          </span>
                        </div>
                      )
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </section>

        <section className="classes-section" id="about">
          <div className="section-heading">
            <p className="eyebrow">SUPPORTED SHOTS</p>
            <h2>Five badminton strokes</h2>
          </div>

          <div className="shot-grid">
            <div className="shot-card">
              <span>01</span>
              <h3>Clear</h3>
              <p>High defensive or attacking shot to the back court.</p>
            </div>

            <div className="shot-card">
              <span>02</span>
              <h3>Drive</h3>
              <p>Fast and relatively flat shot across the court.</p>
            </div>

            <div className="shot-card">
              <span>03</span>
              <h3>Drop</h3>
              <p>Controlled shot that falls close to the opponent's net.</p>
            </div>

            <div className="shot-card">
              <span>04</span>
              <h3>Net Shot</h3>
              <p>Delicate shot played close to the net.</p>
            </div>

            <div className="shot-card">
              <span>05</span>
              <h3>Smash</h3>
              <p>Powerful downward attacking shot.</p>
            </div>
          </div>
        </section>
      </main>

      <footer>
        <p>Badminton AI · CNN-LSTM Shot Recognition</p>
      </footer>
    </div>
  );
}

export default App;