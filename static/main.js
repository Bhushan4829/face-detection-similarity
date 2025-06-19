// static/main.js

window.addEventListener("DOMContentLoaded", () => {
  const fileInput  = document.getElementById("fileInput");
  const fpsInput   = document.getElementById("fpsInput");
  const extractBtn = document.getElementById("extractBtn");
  const searchBtn  = document.getElementById("searchBtn");
  const statusEl   = document.getElementById("status");
  const framesEl   = document.getElementById("frames");
  const resultsEl  = document.getElementById("results");

  let extractedFrames = [];

  // Clear previous frames/results when a new file is selected
  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file && file.type.startsWith("image/")) {
      extractedFrames = [];
      framesEl.innerHTML = "";
      resultsEl.innerHTML = "";
      statusEl.textContent = "";
    }
  });

  extractBtn.addEventListener("click", () => {
    const file = fileInput.files[0];
    const fps  = parseFloat(fpsInput.value);
    statusEl.textContent = "";
    framesEl.innerHTML = "";
    resultsEl.innerHTML = "";
    extractedFrames = [];

    if (!file || !file.type.startsWith("video/")) {
      statusEl.textContent = "⚠️ Please select a video file.";
      return;
    }
    if (!fps || fps <= 0) {
      statusEl.textContent = "⚠️ Enter a valid FPS.";
      return;
    }

    statusEl.textContent = "⏳ Extracting frames...";
    const url   = URL.createObjectURL(file);
    const video = document.createElement("video");
    video.src   = url;
    video.muted = true;
    video.playsInline = true;

    video.addEventListener("loadedmetadata", () => {
      const duration = video.duration;
      const interval = 1 / fps;
      let currentTime = 0;

      const canvas = document.createElement("canvas");
      canvas.width  = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");

      const captureFrame = () => {
        video.currentTime = currentTime;
      };

      video.addEventListener("seeked", () => {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(blob => {
          const fname = `frame_${extractedFrames.length}.jpg`;
          extractedFrames.push({ blob, filename: fname });

          // thumbnail
          const thumb = document.createElement("img");
          thumb.src    = URL.createObjectURL(blob);
          thumb.width  = 120;
          framesEl.appendChild(thumb);

          currentTime += interval;
          if (currentTime <= duration && extractedFrames.length < 30) {
            captureFrame();
          } else {
            statusEl.textContent = `✅ Extracted ${extractedFrames.length} frames.`;
            URL.revokeObjectURL(url);
          }
        }, "image/jpeg");
      });

      // start extraction
      captureFrame();
    });
  });

  searchBtn.addEventListener("click", async () => {
    statusEl.textContent = "";
    resultsEl.innerHTML = "";

    const form = new FormData();

    if (extractedFrames.length > 0) {
      // batch-search frames
      form.delete; // ensure fresh
      endpoint = "/search_batch";
      statusEl.textContent = "🔍 Searching frames...";
      extractedFrames.forEach(f => form.append("files", f.blob, f.filename));
    } else {
      // single-image search
      const file = fileInput.files[0];
      if (!file || !file.type.startsWith("image/")) {
        statusEl.textContent = "⚠️ Select an image or extract frames first.";
        return;
      }
      // clear any old frames if searching new image
      extractedFrames = [];
      framesEl.innerHTML = "";

      endpoint = "/search";
      statusEl.textContent = "🔍 Searching image...";
      form.append("file", file, file.name);
    }

    try {
      const res = await fetch(endpoint, { method: "POST", body: form });
      if (!res.ok) {
        throw new Error(await res.text() || res.statusText);
      }
      const data = await res.json();

      const entries = data.batch_results ?? [data];
      statusEl.textContent = "";

      entries.forEach(entry => {
        const card = document.createElement("div");
        card.className = "card";

        const title = document.createElement("h4");
        title.textContent = entry.filename;
        card.appendChild(title);

        if (entry.results && entry.results.length) {
          entry.results.slice(0,5).forEach(m => {
            const img = document.createElement("img");
            img.src   = `/images/${m.filename}`;
            img.width = 120;
            const sim = document.createElement("p");
            sim.textContent = `${m.filename} (${m.similarity.toFixed(2)})`;
            card.appendChild(img);
            card.appendChild(sim);
          });
        } else {
          const err = document.createElement("p");
          err.textContent = entry.error || "No results";
          card.appendChild(err);
        }

        resultsEl.appendChild(card);
      });
    } catch (err) {
      statusEl.textContent = `❌ ${err.message}`;
    }
  });
});
