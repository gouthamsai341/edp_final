// ---------- Nav toggle (mobile) ----------
document.addEventListener("DOMContentLoaded", () => {
  const toggle = document.getElementById("navToggle");
  const links = document.getElementById("navLinks");
  if (toggle && links) {
    toggle.addEventListener("click", () => links.classList.toggle("open"));
  }

  initDetectPage();
  initHistoryPage();
  initLibraryPage();
});

// ---------- Detection page ----------
function initDetectPage() {
  const uploadBox = document.getElementById("uploadBox");
  const fileInput = document.getElementById("fileInput");
  const preview = document.getElementById("imagePreview");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const statusMsg = document.getElementById("statusMessage");
  const spinner = document.getElementById("spinner");

  if (!uploadBox || !fileInput) return; // not on this page

  let selectedFile = null;

  function showPreview(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      preview.src = e.target.result;
      preview.style.display = "block";
    };
    reader.readAsDataURL(file);
    analyzeBtn.disabled = false;
    setStatus("", "");
  }

  uploadBox.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) showPreview(fileInput.files[0]);
  });

  ["dragover", "dragenter"].forEach((evt) =>
    uploadBox.addEventListener(evt, (e) => {
      e.preventDefault();
      uploadBox.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    uploadBox.addEventListener(evt, (e) => {
      e.preventDefault();
      uploadBox.classList.remove("dragover");
    })
  );
  uploadBox.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) showPreview(file);
  });

  function setStatus(text, kind) {
    statusMsg.textContent = text;
    statusMsg.className = "status-message" + (kind ? " " + kind : "");
  }

  analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    analyzeBtn.disabled = true;
    spinner.style.display = "block";
    setStatus("Analyzing your plant...", "");

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const res = await fetch("/api/predict", { method: "POST", body: formData });
      const data = await res.json();
      spinner.style.display = "none";

      if (data.status === "success") {
        sessionStorage.setItem("lastResult", JSON.stringify(data));
        sessionStorage.setItem("lastImagePreview", preview.src);
        window.location.href = "/result";
      } else if (data.status === "uncertain") {
        setStatus(data.message || "Prediction uncertain.", "uncertain");
        analyzeBtn.disabled = false;
      } else {
        setStatus(data.message || "Something went wrong. Please try again.", "error");
        analyzeBtn.disabled = false;
      }
    } catch (err) {
      spinner.style.display = "none";
      setStatus("Network error. Please check your connection and try again.", "error");
      analyzeBtn.disabled = false;
    }
  });
}

// ---------- Result page ----------
if (document.getElementById("resultRoot")) {
  const raw = sessionStorage.getItem("lastResult");
  const imgSrc = sessionStorage.getItem("lastImagePreview");
  const root = document.getElementById("resultRoot");

  if (!raw) {
    root.innerHTML = `<div class="card"><p>No recent analysis found. Please analyze an image first.</p>
      <a class="btn" href="/detect">Go to Detection</a></div>`;
  } else {
    const data = JSON.parse(raw);
    const confidencePct = (data.confidence * 100).toFixed(1);
    document.getElementById("resultPlant").textContent = data.plant;
    document.getElementById("resultDisease").textContent = data.disease;
    document.getElementById("resultConfidenceText").textContent = confidencePct + "%";
    document.getElementById("confidenceBarFill").style.width = confidencePct + "%";
    if (imgSrc) document.getElementById("resultImage").src = imgSrc;

    if (data.class_name) {
      fetch(`/api/diseases/${encodeURIComponent(data.class_name)}`)
        .then((r) => (r.ok ? r.json() : null))
        .then((info) => {
          if (!info) return;
          fillList("symptomsList", info.symptoms);
          fillList("managementList", info.management);
          fillList("preventionList", info.prevention);
          document.getElementById("descriptionText").textContent = info.description || "";
        })
        .catch(() => {});
    }
  }
}

function fillList(elementId, items) {
  const el = document.getElementById(elementId);
  if (!el) return;
  if (!items || items.length === 0) {
    el.innerHTML = "<li>No additional information available.</li>";
    return;
  }
  el.innerHTML = items.map((i) => `<li>${i}</li>`).join("");
}

// ---------- History page ----------
function initHistoryPage() {
  const tbody = document.getElementById("historyTableBody");
  if (!tbody) return;

  loadHistory();

  const clearBtn = document.getElementById("clearHistoryBtn");
  if (clearBtn) {
    clearBtn.addEventListener("click", async () => {
      if (!confirm("Clear all prediction history?")) return;
      await fetch("/api/history", { method: "DELETE" });
      loadHistory();
    });
  }

  async function loadHistory() {
    tbody.innerHTML = `<tr><td colspan="5">Loading...</td></tr>`;
    try {
      const res = await fetch("/api/history");
      const data = await res.json();
      if (!data.success || data.history.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5">No predictions yet. Analyze a leaf image to get started.</td></tr>`;
        return;
      }
      tbody.innerHTML = data.history
        .map((item) => {
          const date = new Date(item.created_at).toLocaleString();
          const conf = item.confidence != null ? (item.confidence * 100).toFixed(1) + "%" : "-";
          const badgeClass =
            item.status === "success" ? "badge-success" : item.status === "uncertain" ? "badge-uncertain" : "badge-error";
          return `<tr>
            <td>${date}</td>
            <td>${item.plant || "-"}</td>
            <td>${item.disease || "-"}</td>
            <td>${conf}</td>
            <td><span class="badge ${badgeClass}">${item.status}</span>
              <button class="btn-link" onclick="deleteHistoryItem(${item.id})">Delete</button></td>
          </tr>`;
        })
        .join("");
    } catch (err) {
      tbody.innerHTML = `<tr><td colspan="5">Could not load history.</td></tr>`;
    }
  }

  window.deleteHistoryItem = async (id) => {
    await fetch(`/api/history/${id}`, { method: "DELETE" });
    loadHistory();
  };
}

// ---------- Disease library page ----------
function initLibraryPage() {
  const searchInput = document.getElementById("librarySearch");
  if (!searchInput) return;

  searchInput.addEventListener("input", () => {
    const query = searchInput.value.toLowerCase();
    document.querySelectorAll(".disease-card-wrap").forEach((card) => {
      const text = card.dataset.search || "";
      card.style.display = text.includes(query) ? "" : "none";
    });
  });
}
