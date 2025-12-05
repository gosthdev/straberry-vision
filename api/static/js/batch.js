// StrawberryVision - Batch Processing
document.addEventListener("DOMContentLoaded", () => {
    // Elementos del DOM
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const selectBtn = document.getElementById("select-btn");
    const dropPlaceholder = document.getElementById("drop-placeholder");
    const uploadSection = document.getElementById("upload-section");
    const metaForm = document.getElementById("meta-form");
    const progressBarFill = document.getElementById("progress-bar-fill");
    const progressPercent = document.getElementById("progress-percent");
    const progressText = document.getElementById("progress-text");
    const filesTableBody = document.getElementById("files-table-body");
    const cancelBtn = document.getElementById("cancel-btn");
    const viewResultsBtn = document.getElementById("view-results-btn");
    
    // Estado
    let selectedFiles = [];
    let fileStatuses = {}; // {filename: {status: 'pending'|'processing'|'completed'|'error', progress: 0-100, result: null}}
    let batchResults = [];
    let isProcessing = false;
    let abortController = null;

    // Mapeo de clases
    const CLASS_LABELS = {
        'flowering': 'Floración',
        'growing_g': 'Creciendo (verde)',
        'growing_w': 'Creciendo (blanco)',
        'nearly_m': 'Casi maduro',
        'mature': 'Maduro'
    };

    // Utilidades
    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    function showElement(el) {
        el.classList.remove("hidden");
    }

    function hideElement(el) {
        el.classList.add("hidden");
    }

    function getStatusBadge(status, progress = 0) {
        switch (status) {
            case 'completed':
                return `<span class="sv-status-badge completed"><i class="fas fa-check"></i> Completado</span>`;
            case 'processing':
                return `
                    <div class="sv-mini-progress">
                        <div class="sv-mini-progress-bar">
                            <div class="sv-mini-progress-fill" style="width: ${progress}%"></div>
                        </div>
                        <span class="sv-mini-progress-text">${progress}%</span>
                    </div>
                `;
            case 'error':
                return `<span class="sv-status-badge error"><i class="fas fa-times"></i> Error</span>`;
            default:
                return `<span class="sv-status-badge pending">Pendiente</span>`;
        }
    }

    // Renderizar tabla de archivos
    function renderFilesTable() {
        filesTableBody.innerHTML = "";
        
        selectedFiles.forEach((file, index) => {
            const status = fileStatuses[file.name] || { status: 'pending', progress: 0 };
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>
                    <div class="sv-file-info">
                        <i class="fas fa-image sv-file-icon"></i>
                        <span>${file.name}</span>
                    </div>
                </td>
                <td class="sv-file-size">${formatFileSize(file.size)}</td>
                <td id="status-${index}">${getStatusBadge(status.status, status.progress)}</td>
            `;
            filesTableBody.appendChild(tr);
        });
    }

    // Actualizar estado de un archivo
    function updateFileStatus(filename, status, progress = 0) {
        fileStatuses[filename] = { status, progress };
        
        const index = selectedFiles.findIndex(f => f.name === filename);
        if (index >= 0) {
            const statusCell = document.getElementById(`status-${index}`);
            if (statusCell) {
                statusCell.innerHTML = getStatusBadge(status, progress);
            }
        }
        
        updateOverallProgress();
    }

    // Actualizar progreso total
    function updateOverallProgress() {
        const total = selectedFiles.length;
        const completed = Object.values(fileStatuses).filter(s => s.status === 'completed').length;
        const errors = Object.values(fileStatuses).filter(s => s.status === 'error').length;
        
        const percent = total > 0 ? Math.round(((completed + errors) / total) * 100) : 0;
        
        progressBarFill.style.width = `${percent}%`;
        progressPercent.textContent = `${percent}%`;
        progressText.textContent = `${completed} de ${total} archivos completados`;
        
        if (percent >= 100) {
            progressBarFill.classList.add("completed");
            progressPercent.classList.add("completed");
        }
    }

    // Manejar selección de archivos
    function handleFiles(files) {
        // Filtrar solo imágenes
        const imageFiles = Array.from(files).filter(f => f.type.startsWith("image/"));
        
        if (imageFiles.length === 0) {
            alert("Por favor, selecciona archivos de imagen válidos.");
            return;
        }

        selectedFiles = imageFiles;
        fileStatuses = {};
        batchResults = [];
        
        // Inicializar estados
        imageFiles.forEach(f => {
            fileStatuses[f.name] = { status: 'pending', progress: 0 };
        });

        // Mostrar sección de carga
        showElement(uploadSection);
        showElement(metaForm);
        renderFilesTable();
        updateOverallProgress();

        // Iniciar procesamiento automático
        startProcessing();
    }

    // Procesar archivos uno por uno
   async function startProcessing() {
    if (isProcessing) return;
    
    isProcessing = true;
    abortController = new AbortController();
    viewResultsBtn.disabled = true;

    // Recolectar datos
    const loteId = document.getElementById("lote-id")?.value || `LOTE-${Date.now()}`;
    const location = document.getElementById("location")?.value || "";
    const description = document.getElementById("description")?.value || "";

    const formData = new FormData();
    formData.append("lote_id", loteId);
    formData.append("location", location);
    formData.append("description", description);

    selectedFiles.forEach(f => formData.append("files", f));

    try {
        // ENVIAR Y ESPERAR RESULTADO (SIN POLLING)
        console.log("Enviando lote para procesamiento...");
        
        const response = await fetch("/predict-batch", {
            method: "POST",
            body: formData,
            signal: abortController.signal,
            timeout: 330000  // 5.5 minutos
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();
        console.log("Procesamiento completado:", data);

        // Guardar en localStorage
        localStorage.setItem("sv_current_batch", JSON.stringify(data));

        // Ir a resultados
        setTimeout(() => {
            window.location.href = "/results";
        }, 1000);

    } catch (err) {
        isProcessing = false;
        hideElement(uploadSection);
        showElement(progressContainer);
        
        const errorMsg = err.name === 'AbortError' 
            ? 'Solicitud cancelada'
            : err.message;
        
        alert(`Error: ${errorMsg}`);
        console.error(err);
    }
}

    // Cancelar procesamiento
    function cancelProcessing() {
        if (abortController) {
            abortController.abort();
        }
        isProcessing = false;
        
        // Resetear UI
        selectedFiles = [];
        fileStatuses = {};
        batchResults = [];
        hideElement(uploadSection);
        hideElement(metaForm);
        progressBarFill.style.width = "0%";
        progressPercent.textContent = "0%";
        progressBarFill.classList.remove("completed");
    }

    // Ver resultados
    function viewResults() {
        // Guardar resultados en localStorage para la página de resultados
        const loteId = document.getElementById("lote-id")?.value || `LOTE-${Date.now()}`;
        const location = document.getElementById("location")?.value || "";
        const description = document.getElementById("description")?.value || "";

        const batchData = {
            id: loteId,
            created_at: new Date().toISOString(),
            location,
            description,
            num_files: selectedFiles.length,
            results: batchResults
        };

        localStorage.setItem("sv_current_batch", JSON.stringify(batchData));
        
        // También agregar al historial
        const history = JSON.parse(localStorage.getItem("sv_history") || "[]");
        history.unshift({
            id: `batch-${Date.now()}`,
            type: "batch",
            created_at: batchData.created_at,
            lote_id: loteId,
            num_files: batchData.num_files,
            location,
            description
        });
        if (history.length > 50) history.pop();
        localStorage.setItem("sv_history", JSON.stringify(history));

        // Redirigir a página de resultados
        window.location.href = "/results";
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
            handleFiles(files);
        }
    });

    // Click en dropzone
    dropzone.addEventListener("click", () => {
        if (!isProcessing) {
            fileInput.click();
        }
    });

    selectBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (!isProcessing) {
            fileInput.click();
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files?.length > 0) {
            handleFiles(fileInput.files);
        }
    });

    cancelBtn.addEventListener("click", cancelProcessing);
    viewResultsBtn.addEventListener("click", viewResults);
});
