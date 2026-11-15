import React, { useState, useRef, useEffect } from 'react';

export default function DiagnosisDashboard() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [confThreshold, setConfThreshold] = useState(0.35);

  const imageRef = useRef(null);
  const canvasRef = useRef(null);

  // Handle file selection
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  // Send image to FastAPI v2.0 Backend
  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(
        `http://localhost:8000/api/v2/diagnose?conf_threshold=${confThreshold}`,
        {
          method: 'POST',
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        // Catches HTTP 422 (Blur rejection) or 400 (Invalid image)
        throw new Error(data.detail || 'Diagnosis failed.');
      }

      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Draw scaled bounding boxes whenever results or window size change
  useEffect(() => {
    if (!result || !imageRef.current || !canvasRef.current) return;

    const img = imageRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Match canvas resolution to rendered image dimensions
    const displayWidth = img.clientWidth;
    const displayHeight = img.clientHeight;
    canvas.width = displayWidth;
    canvas.height = displayHeight;

    // Calculate scale factors from original image coordinates to screen coordinates
    const scaleX = displayWidth / img.naturalWidth;
    const scaleY = displayHeight / img.naturalHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    result.detections.forEach((det) => {
      const x = det.box.x1 * scaleX;
      const y = det.box.y1 * scaleY;
      const w = det.box.width * scaleX;
      const h = det.box.height * scaleY;

      // Color-code: Red for pathogens, Amber for abiotic stress, Emerald for healthy
      let strokeColor = '#10B981'; // Emerald (Healthy)
      if (det.is_pathogen) {
        strokeColor = '#EF4444'; // Red (Pathogen)
      } else if (det.class_name === 'abiotic_chlorosis') {
        strokeColor = '#F59E0B'; // Amber (Nutrient/Water Stress)
      }

      // Draw Bounding Box
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w, h);

      // Draw Label Pill Background
      const label = `${det.class_name.replace('_', ' ')} (${Math.round(det.confidence * 100)}%)`;
      ctx.font = '600 13px Inter, sans-serif';
      const textWidth = ctx.measureText(label).width;

      ctx.fillStyle = strokeColor;
      ctx.fillRect(x, y > 24 ? y - 24 : y, textWidth + 12, 24);

      // Draw Label Text
      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(label, x + 6, y > 24 ? y - 7 : y + 17);
    });
  }, [result]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 font-sans">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <header className="border-b border-slate-800 pb-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-emerald-400">PhytoScan AI v2.0</h1>
            <p className="text-sm text-slate-400">
              Multi-Lesion Object Detection & Abiotic Chlorosis Differentiator
            </p>
          </div>

          {/* Confidence Slider */}
          <div className="flex items-center gap-3 bg-slate-900 px-4 py-2 rounded-lg border border-slate-800">
            <label className="text-xs text-slate-400">Confidence Gate:</label>
            <input
              type="range"
              min="0.15"
              max="0.85"
              step="0.05"
              value={confThreshold}
              onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
              className="accent-emerald-500 cursor-pointer"
            />
            <span className="text-xs font-mono text-emerald-400 w-10">
              {Math.round(confThreshold * 100)}%
            </span>
          </div>
        </header>

        {/* Main Workspace */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: Image Canvas & Upload */}
          <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col justify-between">
            <div className="relative flex items-center justify-center bg-slate-950 rounded-lg border border-dashed border-slate-700 min-h-[380px] overflow-hidden">
              {previewUrl ? (
                <div className="relative inline-block">
                  <img
                    ref={imageRef}
                    src={previewUrl}
                    alt="Uploaded leaf"
                    className="max-h-[460px] w-auto object-contain block"
                  />
                  <canvas
                    ref={canvasRef}
                    className="absolute top-0 left-0 pointer-events-none"
                  />
                </div>
              ) : (
                <p className="text-slate-500 text-sm">
                  Upload a leaf photograph to inspect lesions and chlorosis
                </p>
              )}
            </div>

            {/* Action Bar */}
            <div className="mt-4 flex flex-wrap items-center gap-3">
              <label className="cursor-pointer bg-slate-800 hover:bg-slate-700 text-sm px-4 py-2.5 rounded-lg border border-slate-700 transition">
                Select Leaf Image
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="hidden"
                />
              </label>

              <button
                onClick={handleAnalyze}
                disabled={!selectedFile || loading}
                className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white font-medium text-sm px-6 py-2.5 rounded-lg transition cursor-pointer"
              >
                {loading ? 'Running ONNX Inference...' : 'Diagnose Foliage'}
              </button>
            </div>

            {/* Quality Gate Error Display */}
            {error && (
              <div className="mt-4 p-3.5 bg-red-950/60 border border-red-800 rounded-lg text-sm text-red-300">
                <strong>Gate Rejected:</strong> {error}
              </div>
            )}
          </div>

          {/* Right Column: Telemetry & Prescriptions */}
          <div className="lg:col-span-5 space-y-4">
            {result ? (
              <>
                {/* Telemetry Card */}
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs uppercase tracking-wider text-slate-400">
                      Severity Assessment
                    </span>
                    <span className="text-xs font-mono text-slate-400">
                      {result.processing_time_ms} ms
                    </span>
                  </div>
                  <p className="text-lg font-semibold text-white">{result.severity_index}</p>
                  <p className="text-sm text-slate-300">{result.advice}</p>

                  <div className="pt-2 border-t border-slate-800 flex gap-4 text-xs text-slate-400">
                    <span>Blur Score: {result.quality_score.blur_variance}</span>
                    <span>Detections: {result.total_detections}</span>
                  </div>
                </div>

                {/* Treatment Prescription Cards */}
                <div className="space-y-3 max-h-[420px] overflow-y-auto pr-1">
                  {Object.entries(result.prescriptions).map(([diseaseKey, plan]) => (
                    <div
                      key={diseaseKey}
                      className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-2"
                    >
                      <div className="flex items-center justify-between">
                        <h3 className="font-semibold text-emerald-400 capitalize">
                          {diseaseKey.replace('_', ' ')}
                        </h3>
                        <span
                          className={`text-xs px-2 py-0.5 rounded font-medium ${
                            plan.urgency === 'Critical' || plan.urgency === 'High'
                              ? 'bg-red-950 text-red-400 border border-red-800'
                              : 'bg-amber-950 text-amber-400 border border-amber-800'
                          }`}
                        >
                          Urgency: {plan.urgency}
                        </span>
                      </div>
                      <p className="text-xs text-slate-400">{plan.pathogen_type}</p>

                      <div className="text-xs space-y-1.5 pt-2 text-slate-300">
                        <p>
                          <strong className="text-slate-100">Organic:</strong>{' '}
                          {plan.organic_treatment}
                        </p>
                        <p>
                          <strong className="text-slate-100">Chemical:</strong>{' '}
                          {plan.chemical_treatment}
                        </p>
                        <p>
                          <strong className="text-slate-100">Cultural:</strong>{' '}
                          {plan.cultural_practice}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-sm text-slate-400 h-full flex flex-col justify-center">
                <p className="font-medium text-slate-200 mb-2">How this UI validates your ML concepts:</p>
                <ul className="list-disc pl-5 space-y-2">
                  <li>
                    <strong>Coordinate Rescaling:</strong> Maps raw $640 \times 640$ ONNX coordinates back to responsive screen pixels.
                  </li>
                  <li>
                    <strong>Visual Disentanglement:</strong> Draws <span className="text-red-400">Red</span> boxes on active fungal/bacterial lesions and <span className="text-amber-400">Amber</span> boxes on harmless nutrient yellowing.
                  </li>
                  <li>
                    <strong>Interactive Thresholding:</strong> Lets you adjust the confidence slider live to see how Precision vs. Recall trade-offs behave.
                  </li>
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
