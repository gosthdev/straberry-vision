// StrawberryVision - Single Image Detection
document.addEventListener("DOMContentLoaded", () => {
    // Elementos del DOM
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const selectBtn = document.getElementById("select-btn");
    const changeBtn = document.getElementById("change-btn");
    const analyzeBtn = document.getElementById("analyze-btn");
    const previewImg = document.getElementById("preview-image");
    const dropPlaceholder = document.getElementById("drop-placeholder");
    const analyzeActions = document.getElementById("analyze-actions");
    const progressContainer = document.getElementById("progress-container");
    const progressBarFill = document.getElementById("progress-bar-fill");
    const progressPercent = document.getElementById("progress-percent");
    const resultsSection = document.getElementById("results-section");
    const classCountersList = document.getElementById("class-counters-list");
    const classScoresList = document.getElementById("class-scores-list");
    const originalImage = document.getElementById("original-image");
    const annotatedImage = document.getElementById("annotated-image");
    const newAnalysisBtn = document.getElementById("new-analysis-btn");
    const saveHistoryBtn = document.getElementById("save-history-btn");
    const debugSection = document.getElementById("debug-section");
    const jsonOutput = document.getElementById("json-output");

    // Mapeo de clases a nombres en español
    const CLASS_LABELS = {
        'flowering': 'Floración 🌸',
        'growing_g': 'Creciendo (verde) 🌱',
        'growing_w': 'Creciendo (blanco) ⚪',
        'nearly_m': 'Casi maduro 🍊',
        'mature': 'Maduro 🍓'
    };

    // Mapeo de clases a CSS class
    const CLASS_CSS = {
        'flowering': 'flowering',
        'growing_g': 'growing-g',
        'growing_w': 'growing-w',
        'nearly_m': 'nearly-m',
        'mature': 'mature'
    };

    let currentFile = null;
    let lastResult = null;

    // Funciones de utilidad
    function showElement(el) {
        el.classList.remove("hidden");
    }

    function hideElement(el) {
        el.classList.add("hidden");
    }

    function setProgress(percent) {
        progressBarFill.style.width = `${percent}%`;
        progressPercent.textContent = `${Math.round(percent)}%`;
        if (percent >= 100) {
            progressBarFill.classList.add("completed");
        } else {
            progressBarFill.classList.remove("completed");
        }
    }

    function simulateProgress(callback) {
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress >= 90) {
                clearInterval(interval);
                progress = 90;
            }
            setProgress(progress);
        }, 200);
        return interval;
    }

    // Manejar selección de archivo
    function handleFile(file) {
        if (!file || !file.type.startsWith("image/")) {
            alert("Por favor, selecciona un archivo de imagen válido.");
            return;
        }

        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            showElement(previewImg);
            hideElement(dropPlaceholder);
            showElement(analyzeActions);
        };
        reader.readAsDataURL(file);
    }

    // Resetear a estado inicial
    function resetToInitial() {
        currentFile = null;
        lastResult = null;
        previewImg.src = "";
        hideElement(previewImg);
        showElement(dropPlaceholder);
        hideElement(analyzeActions);
        hideElement(progressContainer);
        hideElement(resultsSection);
        hideElement(debugSection);
        setProgress(0);
        fileInput.value = "";
    }

    // Construir UI de resultados
    function buildResults(data) {
        // Limpiar contenedores
        classCountersList.innerHTML = "";
        classScoresList.innerHTML = "";

        const summary = data.summary_by_class || {};
        const allClasses = ['flowering', 'growing_g', 'growing_w', 'nearly_m', 'mature'];

        // Contadores por clase
        allClasses.forEach(cls => {
            const info = summary[cls] || { count: 0, best_score: 0 };
            const cssClass = CLASS_CSS[cls];
            const label = CLASS_LABELS[cls];
            
            const item = document.createElement("div");
            item.className = "sv-class-item";
            item.innerHTML = `
                <div class="sv-class-label">
                    <span class="sv-class-dot ${cssClass}"></span>
                    <span class="sv-class-name">${label}</span>
                </div>
                <span class="sv-class-count">${info.count}</span>
            `;
            classCountersList.appendChild(item);
        });

        // Scores máximos por clase
        allClasses.forEach(cls => {
            const info = summary[cls] || { count: 0, best_score: 0 };
            if (info.count === 0) return;
            
            const cssClass = CLASS_CSS[cls];
            const label = CLASS_LABELS[cls];
            const score = (info.best_score * 100).toFixed(1);
            const scoreClass = score >= 80 ? 'high' : score >= 50 ? 'medium' : 'low';
            
            const item = document.createElement("div");
            item.className = "sv-score-item";
            item.innerHTML = `
                <div class="sv-score-header">
                    <span class="sv-score-label">${label}</span>
                    <span class="sv-score-value ${scoreClass}">${score}%</span>
                </div>
                <div class="sv-score-bar">
                    <div class="sv-score-bar-fill ${cssClass}" style="width: ${score}%"></div>
                </div>
            `;
            classScoresList.appendChild(item);
        });

        // Si no hay detecciones, mostrar mensaje
        if (data.num_detections === 0) {
            classScoresList.innerHTML = '<p style="color: var(--text-secondary); font-size: 0.9rem;">No se detectaron fresas en la imagen.</p>';
        }

        // Imágenes
        if (previewImg.src) {
            originalImage.src = previewImg.src;
        }
        if (data.annotated_image_base64) {
            annotatedImage.src = `data:image/png;base64,${data.annotated_image_base64}`;
        }

        // JSON debug
        jsonOutput.textContent = JSON.stringify(data, null, 2);
    }

    // Analizar imagen
    async function analyzeImage() {
        if (!currentFile) return;

        hideElement(analyzeActions);
        showElement(progressContainer);
        setProgress(0);
        
        const progressInterval = simulateProgress();

        try {
            const formData = new FormData();
            formData.append("file", currentFile);

            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            clearInterval(progressInterval);

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || `Error HTTP ${response.status}`);
            }

            setProgress(100);
            const data = await response.json();
            lastResult = data;

            // Esperar un momento para mostrar el 100%
            setTimeout(() => {
                hideElement(progressContainer);
                buildResults(data);
                showElement(resultsSection);
                showElement(debugSection);
            }, 500);

        } catch (err) {
            clearInterval(progressInterval);
            hideElement(progressContainer);
            showElement(analyzeActions);
            alert(`Error al analizar la imagen: ${err.message}`);
            console.error(err);
        }
    }

    // Guardar en historial (localStorage)
    function saveToHistory() {
        if (!lastResult) return;

        const history = JSON.parse(localStorage.getItem("sv_history") || "[]");
        const entry = {
            id: `single-${Date.now()}`,
            type: "single",
            created_at: new Date().toISOString(),
            filename: currentFile?.name || "imagen",
            num_detections: lastResult.num_detections,
            summary_by_class: lastResult.summary_by_class,
            max_confidence: lastResult.max_confidence,
            thumbnail: previewImg.src.substring(0, 200) + "..." // Guardar solo inicio
        };

        history.unshift(entry);
        // Limitar a 50 entradas
        if (history.length > 50) history.pop();

        localStorage.setItem("sv_history", JSON.stringify(history));
        alert("¡Resultado guardado en el historial!");
    }

    // Event Listeners

    // Drag & Drop
    ["dragenter", "dragover"].forEach(event => {
        dropzone.addEventListener(event, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(event => {
        dropzone.addEventListener(event, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        const files = e.dataTransfer?.files;
        if (files?.length > 0) {
            handleFile(files[0]);
        }
    });

    // Click en dropzone
    dropzone.addEventListener("click", (e) => {
        if (e.target === analyzeBtn || e.target === changeBtn || e.target === selectBtn) return;
        if (!currentFile) {
            fileInput.click();
        }
    });

    selectBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files?.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    changeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        resetToInitial();
    });

    analyzeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        analyzeImage();
    });

    newAnalysisBtn.addEventListener("click", resetToInitial);
    saveHistoryBtn.addEventListener("click", saveToHistory);
});
