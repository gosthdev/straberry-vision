const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const selectFilesBtn = document.getElementById('select-files-btn');
const uploadProgressContainer = document.getElementById('upload-progress-container');
const progressBar = document.getElementById('progress-bar');
const progressText = document.getElementById('progress-text');
const uploadStatusText = document.getElementById('upload-status-text');
const analyzeBtn = document.getElementById('analyze-btn');
const resultsBody = document.getElementById('results-body');

// Views
const uploadView = document.getElementById('upload-view');
const resultsView = document.getElementById('results-view');

// Pagination elements
const paginationContainer = document.getElementById('pagination-container');
const prevPageBtn = document.getElementById('prev-page-btn');
const nextPageBtn = document.getElementById('next-page-btn');
const pageInfo = document.getElementById('page-info');
const resetBtn = document.getElementById('reset-btn');

// Modal elements
const imageModal = document.getElementById('image-modal');
const modalImage = document.getElementById('modal-image');

let uploadedFiles = [];
let allResults = [];
let currentPage = 1;
const itemsPerPage = 5;

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('bg-gray-100', 'dark:bg-[#2a1d1e]');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('bg-gray-100', 'dark:bg-[#2a1d1e]');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('bg-gray-100', 'dark:bg-[#2a1d1e]');
    handleFiles(e.dataTransfer.files);
});

selectFilesBtn.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
});

async function handleFiles(files) {
    if (files.length === 0) return;

    uploadProgressContainer.classList.remove('hidden');
    analyzeBtn.disabled = true;
    uploadedFiles = []; 
    resultsBody.innerHTML = ''; 

    const totalFiles = files.length;
    let completedFiles = 0;

    uploadStatusText.textContent = 'Subiendo imágenes...';
    progressBar.style.width = '0%';
    progressText.textContent = `0 de ${totalFiles} completados`;

    try {
        for (let i = 0; i < files.length; i++) {
            const singleFormData = new FormData();
            singleFormData.append('files', files[i]);

            const response = await fetch(`${window.APP_CONFIG.API_URL}/upload`, {
                method: 'POST',
                body: singleFormData
            });

            if (response.ok) {
                const data = await response.json();
                uploadedFiles.push(...data.files);
                completedFiles++;
                const percent = (completedFiles / totalFiles) * 100;
                progressBar.style.width = `${percent}%`;
                progressText.textContent = `${completedFiles} de ${totalFiles} completados`;
            } else {
                console.error('Error uploading file:', files[i].name);
            }
        }

        uploadStatusText.textContent = 'Carga completa. Iniciando análisis...';
        
        await analyzeImages();

    } catch (error) {
        console.error('Error uploading files:', error);
        uploadStatusText.textContent = 'Error en la carga';
    }
}

async function analyzeImages() {
    if (uploadedFiles.length === 0) return;

    uploadStatusText.textContent = 'Analizando imágenes...';
    
    const filenames = uploadedFiles.map(f => f.server_filename);
    const originalFilenames = uploadedFiles.map(f => f.original_filename);

    try {
        const response = await fetch(`${window.APP_CONFIG.API_URL}/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ filenames: filenames, original_filenames: originalFilenames })
        });

        if (response.ok) {
            const data = await response.json();
            allResults = data.results;
            currentPage = 1;
            
            const totalStrawberries = allResults.reduce((acc, r) => acc + r.count, 0);
            document.getElementById('total-strawberries').textContent = totalStrawberries;
            const avgConf = allResults.reduce((acc, r) => acc + r.avg_confidence, 0) / (allResults.length || 1);
            document.getElementById('avg-confidence').textContent = `${avgConf.toFixed(1)}%`;
            document.getElementById('total-images').textContent = allResults.length;

            updateMaturityChart(allResults, allResults.length);

            updatePagination();
            
            uploadView.classList.add('hidden');
            resultsView.classList.remove('hidden');
            
        } else {
            console.error('Error analyzing images');
            alert('Error al analizar las imágenes.');
            uploadStatusText.textContent = 'Error en el análisis';
        }
    } catch (error) {
        console.error('Error analyzing images:', error);
        alert('Error de conexión al analizar las imágenes.');
        uploadStatusText.textContent = 'Error de conexión';
    }
}

resetBtn.addEventListener('click', async () => {
    if (uploadedFiles.length === 0) {
        resultsView.classList.add('hidden');
        uploadView.classList.remove('hidden');
        return;
    }

    resetBtn.disabled = true;
    resetBtn.innerHTML = '<span class="material-symbols-outlined animate-spin">refresh</span> Limpiando...';

    const filenames = uploadedFiles.map(f => f.server_filename);

    try {
        await fetch(`${window.APP_CONFIG.API_URL}/cleanup`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(filenames)
        });
    } catch (error) {
        console.error('Error cleaning up:', error);
    }

    // Reset UI state
    uploadedFiles = [];
    allResults = [];
    resultsBody.innerHTML = '';
    
    // Switch Views
    resultsView.classList.add('hidden');
    uploadView.classList.remove('hidden');
    uploadProgressContainer.classList.add('hidden');
    
    resetBtn.disabled = false;
    resetBtn.innerHTML = '<span class="material-symbols-outlined mr-2">add_photo_alternate</span> Nueva Consulta';
    
    // Reset file input
    fileInput.value = '';
    
    // Reset progress bar
    progressBar.style.width = '0%';
    progressText.textContent = '0 de 0 completados';
    uploadStatusText.textContent = 'Subiendo imágenes...';
});

function updatePagination() {
    const totalPages = Math.ceil(allResults.length / itemsPerPage);
    
    if (totalPages <= 1) {
        paginationContainer.classList.add('hidden');
    } else {
        paginationContainer.classList.remove('hidden');
    }

    pageInfo.textContent = `Página ${currentPage} de ${totalPages}`;
    
    prevPageBtn.disabled = currentPage === 1;
    nextPageBtn.disabled = currentPage === totalPages;

    const start = (currentPage - 1) * itemsPerPage;
    const end = start + itemsPerPage;
    const pageResults = allResults.slice(start, end);
    
    renderResults(pageResults);
}

prevPageBtn.addEventListener('click', () => {
    if (currentPage > 1) {
        currentPage--;
        updatePagination();
    }
});

nextPageBtn.addEventListener('click', () => {
    const totalPages = Math.ceil(allResults.length / itemsPerPage);
    if (currentPage < totalPages) {
        currentPage++;
        updatePagination();
    }
});

function renderResults(results) {
    resultsBody.innerHTML = '';
    results.forEach((result, index) => {
        const rowId = `result-row-${index}`;
        const detailsId = `result-details-${index}`;
        
        const tr = document.createElement('tr');
        tr.className = 'cursor-pointer bg-primary/5 dark:bg-primary/10 hover:bg-primary/10 dark:hover:bg-primary/20';
        tr.onclick = (e) => {
            if (!e.target.closest('button')) {
                toggleDetails(detailsId);
            }
        };
        
        tr.innerHTML = `
            <td class="p-4 align-middle">
                <div class="relative w-20 h-20 rounded-md overflow-hidden ring-2 ring-primary">
                    <img alt="strawberry detection thumbnail" class="w-full h-full object-cover" src="${window.APP_CONFIG.API_URL}${result.thumbnail_url}"/>
                </div>
            </td>
            <td class="p-4 align-middle font-medium text-gray-900 dark:text-white">${result.filename}</td>
            <td class="p-4 align-middle text-center text-gray-800 dark:text-gray-200">${result.count}</td>
            <td class="p-4 align-middle text-center text-gray-800 dark:text-gray-200">${result.avg_confidence.toFixed(1)}%</td>
            <td class="p-4 align-middle text-center">
                <button onclick="openModal('${window.APP_CONFIG.API_URL}${result.annotated_url}')" class="inline-flex items-center justify-center rounded-md bg-primary px-3 py-1.5 text-sm font-semibold text-white shadow-sm hover:bg-primary/90 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary">
                    Ver
                </button>
            </td>
            <td class="p-4 align-middle text-center"><span class="material-symbols-outlined transform transition-transform" id="icon-${detailsId}">expand_more</span></td>
        `;
        
        const trDetails = document.createElement('tr');
        trDetails.id = detailsId;
        trDetails.className = 'hidden bg-primary/5 dark:bg-primary/10';
        
        let detectionsHtml = '';
        result.detections.forEach(det => {
            detectionsHtml += `
                <div class="p-3 rounded-md bg-white dark:bg-gray-800/60 border border-black/5 dark:border-white/5">
                    <p class="font-bold text-primary">${det.class} <span class="text-gray-800 dark:text-gray-200">- ${(det.score * 100).toFixed(1)}%</span></p>
                    <p class="text-xs text-gray-600 dark:text-gray-400">Coordenadas (x, y, w, h): ${det.box[0]}, ${det.box[1]}, ${det.box[2]-det.box[0]}, ${det.box[3]-det.box[1]}</p>
                </div>
            `;
        });
        
        if (result.detections.length === 0) {
            detectionsHtml = '<p class="text-sm text-gray-600 dark:text-gray-400">No se detectaron fresas.</p>';
        }

        trDetails.innerHTML = `
            <td class="p-0" colspan="6">
                <div class="p-4 bg-gray-100 dark:bg-gray-900/50">
                    <h5 class="text-base font-bold mb-3 text-gray-900 dark:text-white">Detecciones Individuales</h5>
                    <div class="space-y-3">
                        ${detectionsHtml}
                    </div>
                    <!-- Image removed from here as requested -->
                </div>
            </td>
        `;
        
        resultsBody.appendChild(tr);
        resultsBody.appendChild(trDetails);
    });
}

function toggleDetails(id) {
    const element = document.getElementById(id);
    const icon = document.getElementById(`icon-${id}`);
    if (element.classList.contains('hidden')) {
        element.classList.remove('hidden');
        icon.classList.add('rotate-180');
    } else {
        element.classList.add('hidden');
        icon.classList.remove('rotate-180');
    }
}

function openModal(imageUrl) {
    modalImage.src = imageUrl;
    imageModal.classList.remove('hidden');
    imageModal.classList.add('flex');
    document.body.style.overflow = 'hidden'; 
}

function closeModal(event, force = false) {
    if (force || event.target === imageModal) {
        imageModal.classList.add('hidden');
        imageModal.classList.remove('flex');
        modalImage.src = '';
        document.body.style.overflow = ''; 
    }
}

let maturityChart = null;

function updateMaturityChart(results, totalImages) {
    const ctx = document.getElementById('maturity-chart').getContext('2d');
    const chartTotalCenter = document.getElementById('chart-total-center');
    const chartLegend = document.getElementById('chart-legend');

    // Calculate distribution based on CLASS_NAMES
    // ['flowering', 'growing_g', 'growing_w', 'nearly_m', 'mature']
    let counts = {
        'mature': 0,
        'nearly_m': 0,
        'growing_w': 0,
        'growing_g': 0,
        'flowering': 0
    };

    results.forEach(result => {
        result.detections.forEach(det => {
            if (counts.hasOwnProperty(det.class)) {
                counts[det.class]++;
            }
        });
    });

    const totalStrawberries = Object.values(counts).reduce((a, b) => a + b, 0);
    
    const formattedTotal = totalStrawberries >= 1000 ? (totalStrawberries / 1000).toFixed(1) + 'k' : totalStrawberries;
    chartTotalCenter.textContent = formattedTotal;

    const labels = ['mature', 'nearly_m', 'growing_w', 'growing_g', 'flowering'];
    
    const colorMap = {
        'mature': '#dc2626',    // Red
        'nearly_m': '#f97316',  // Orange
        'growing_w': '#facc15', // Yellow
        'growing_g': '#84cc16', // Green
        'flowering': '#e879f9'  // Pink
    };

    const dataValues = labels.map(l => counts[l]);
    const bgColors = labels.map(l => colorMap[l]);

    const data = {
        labels: labels,
        datasets: [{
            data: dataValues,
            backgroundColor: bgColors,
            borderWidth: 0,
            hoverOffset: 4
        }]
    };

    if (maturityChart) {
        maturityChart.destroy();
    }

    maturityChart = new Chart(ctx, {
        type: 'doughnut',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '70%',
            plugins: {
                legend: {
                    display: false 
                },
                tooltip: {
                    enabled: true
                }
            }
        }
    });

    chartLegend.innerHTML = '';
    
    labels.forEach((label, index) => {
        const count = counts[label];
        const percentage = totalStrawberries > 0 ? Math.round((count / totalStrawberries) * 100) : 0;
        
        const item = document.createElement('div');
        item.className = 'flex items-center gap-2 w-full';
        item.innerHTML = `
            <span class="w-3 h-3 rounded-full flex-shrink-0" style="background-color: ${bgColors[index]}"></span>
            <span class="font-medium text-xs truncate flex-grow" title="${label}">${label}</span>
            <span class="text-gray-500 text-xs">${percentage}%</span>
        `;
        chartLegend.appendChild(item);
    });
}

// Prevent accidental page refresh if files are uploaded
window.addEventListener('beforeunload', (e) => {
    if (uploadedFiles.length > 0) {
        e.preventDefault();
        e.returnValue = '';
    }
});