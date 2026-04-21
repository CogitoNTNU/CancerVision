(() => {
  const MODALITIES = ["flair", "t1", "t1ce", "t2"];
  const state = {
    weights: null,
    flair: null,
    t1: null,
    t1ce: null,
    t2: null,
  };

  const runButton = document.getElementById("run-button");
  const statusEl = document.getElementById("status");
  const resultEl = document.getElementById("result");
  const resultSummary = document.getElementById("result-summary");
  const downloadLink = document.getElementById("download-link");
  const architectureSelect = document.getElementById("architecture");

  function detectModality(filename) {
    const lower = filename.toLowerCase();
    if (lower.includes("flair")) return "flair";
    if (lower.includes("t1ce") || lower.includes("t1c")) return "t1ce";
    if (lower.includes("t2")) return "t2";
    if (lower.includes("t1")) return "t1";
    return null;
  }

  function updateZoneDisplay(slot) {
    const zone = document.querySelector(`.dropzone[data-slot="${slot}"]`);
    if (!zone) return;
    const statusSpan = zone.querySelector(".dz-status");
    const file = state[slot];
    if (file) {
      zone.classList.add("filled");
      const size = (file.size / (1024 * 1024)).toFixed(1);
      statusSpan.textContent = `${file.name} (${size} MB)`;
    } else {
      zone.classList.remove("filled");
      if (slot === "weights") {
        statusSpan.innerHTML = "Drop <code>.pth</code> / <code>.pt</code> here or click to browse";
      } else {
        statusSpan.textContent = "Drop .nii(.gz)";
      }
    }
  }

  function refreshRunButton() {
    const hasAll =
      state.weights &&
      MODALITIES.every((m) => state[m] !== null);
    runButton.disabled = !hasAll;
  }

  function assignFile(slot, file) {
    if (slot === "weights") {
      state.weights = file;
      updateZoneDisplay("weights");
      refreshRunButton();
      return;
    }

    // Route modality drops by filename when possible, falling back to the slot itself.
    const detected = detectModality(file.name);
    const targetSlot = MODALITIES.includes(detected) ? detected : slot;
    state[targetSlot] = file;
    updateZoneDisplay(targetSlot);
    refreshRunButton();
  }

  function distributeModalityFiles(fallbackSlot, files) {
    const list = Array.from(files);
    for (const file of list) {
      const detected = detectModality(file.name);
      const target = MODALITIES.includes(detected) ? detected : fallbackSlot;
      state[target] = file;
      updateZoneDisplay(target);
    }
    refreshRunButton();
  }

  function initDropzones() {
    document.querySelectorAll(".dropzone").forEach((zone) => {
      const slot = zone.dataset.slot;
      const input = zone.querySelector("input[type=file]");

      zone.addEventListener("click", () => input.click());

      zone.addEventListener("dragover", (event) => {
        event.preventDefault();
        zone.classList.add("dragover");
      });
      zone.addEventListener("dragleave", () => {
        zone.classList.remove("dragover");
      });

      zone.addEventListener("drop", (event) => {
        event.preventDefault();
        zone.classList.remove("dragover");
        const files = event.dataTransfer?.files;
        if (!files || files.length === 0) return;

        if (slot === "weights") {
          assignFile("weights", files[0]);
        } else if (files.length > 1) {
          distributeModalityFiles(slot, files);
        } else {
          assignFile(slot, files[0]);
        }
      });

      input.addEventListener("change", (event) => {
        const files = event.target.files;
        if (!files || files.length === 0) return;
        if (slot === "weights") {
          assignFile("weights", files[0]);
        } else {
          distributeModalityFiles(slot, files);
        }
        input.value = "";
      });
    });
  }

  function setStatus(message, kind) {
    statusEl.hidden = !message;
    statusEl.textContent = message || "";
    statusEl.className = "status";
    if (kind) statusEl.classList.add(kind);
  }

  function renderResult(payload) {
    resultSummary.innerHTML = "";
    const rows = [
      ["Architecture", payload.architecture],
      ["Device", payload.device],
      ["Threshold", payload.threshold],
      ["ROI size", payload.roi_size.join(" × ")],
      ["Output file", payload.output_filename],
      ["Tumor core voxels", payload.label_counts.tc_voxels.toLocaleString()],
      ["Whole tumor voxels", payload.label_counts.wt_voxels.toLocaleString()],
      ["Enhancing voxels", payload.label_counts.et_voxels.toLocaleString()],
    ];
    for (const [label, value] of rows) {
      const dt = document.createElement("dt");
      dt.textContent = label;
      const dd = document.createElement("dd");
      dd.textContent = value;
      resultSummary.appendChild(dt);
      resultSummary.appendChild(dd);
    }
    downloadLink.href = payload.download_url;
    downloadLink.setAttribute("download", payload.output_filename);
    resultEl.hidden = false;
  }

  async function loadArchitectures() {
    try {
      const response = await fetch("/api/architectures");
      const data = await response.json();
      const architectures = data.architectures || [];
      architectureSelect.innerHTML = "";
      for (const arch of architectures) {
        const option = document.createElement("option");
        option.value = arch;
        option.textContent = arch;
        architectureSelect.appendChild(option);
      }
      if (architectures.length === 0) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "(no architectures registered)";
        architectureSelect.appendChild(option);
      }
    } catch (err) {
      setStatus(`Failed to load architectures: ${err.message}`, "error");
    }
  }

  async function runInference() {
    runButton.disabled = true;
    resultEl.hidden = true;
    setStatus("Running inference — this can take several minutes on CPU…");

    const formData = new FormData();
    formData.append("weights", state.weights);
    formData.append("flair", state.flair);
    formData.append("t1", state.t1);
    formData.append("t1ce", state.t1ce);
    formData.append("t2", state.t2);
    formData.append("architecture", architectureSelect.value);
    formData.append("device", document.getElementById("device").value);
    formData.append("threshold", document.getElementById("threshold").value);
    formData.append("roi_size", document.getElementById("roi").value);

    try {
      const response = await fetch("/api/infer", {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || `Request failed (${response.status})`);
      }
      setStatus("Inference complete.", "success");
      renderResult(payload);
    } catch (err) {
      setStatus(`Inference failed: ${err.message}`, "error");
    } finally {
      refreshRunButton();
    }
  }

  initDropzones();
  loadArchitectures();
  runButton.addEventListener("click", runInference);
})();
