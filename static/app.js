document.addEventListener("DOMContentLoaded", () => {
    // -----------------------------------------
    // STATE VARIABLES
    // -----------------------------------------
    let currentSampleImageBlob = null;
    let selectedSampleImageIndex = null;
    let trainingPollInterval = null;
    let clinicalAttentionChartInstance = null;
    let performanceChartInstance = null;
    let classChartInstance = null;
    
    const FINDINGS = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
        'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
        'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia'
    ];

    // Default healthy clinical parameters
    const DEFAULT_VITALS = {
        age: 45,
        gender: "M",
        view_pos: "PA",
        temp: 36.8,
        spo2: 98.0,
        wbc: 7000,
        hr: 75,
        sbp: 120,
        cough: "None",
        pain: "None"
    };

    // -----------------------------------------
    // TAB NAVIGATION ROUTING
    // -----------------------------------------
    const navTabs = document.querySelectorAll(".nav-tab");
    const tabPanes = document.querySelectorAll(".tab-pane");

    navTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            navTabs.forEach(t => t.classList.remove("active"));
            tabPanes.forEach(pane => pane.classList.remove("active"));

            tab.classList.add("active");
            const targetPane = document.getElementById(tab.getAttribute("data-tab"));
            targetPane.classList.add("active");

            // Fetch metrics if opening analytics
            if (tab.getAttribute("data-tab") === "tab-analytics") {
                fetchEvaluationMetrics();
            }
        });
    });

    // -----------------------------------------
    // SLIDERS REALTIME DISPLAY UPDATE
    // -----------------------------------------
    const sliders = [
        { id: "slider-age", valId: "val-age" },
        { id: "slider-temp", valId: "val-temp" },
        { id: "slider-spo2", valId: "val-spo2" },
        { id: "slider-wbc", valId: "val-wbc" },
        { id: "slider-hr", valId: "val-hr" },
        { id: "slider-sbp", valId: "val-sbp" }
    ];

    sliders.forEach(slider => {
        const inputEl = document.getElementById(slider.id);
        const valEl = document.getElementById(slider.valId);
        if (inputEl && valEl) {
            inputEl.addEventListener("input", () => {
                valEl.textContent = inputEl.value;
            });
        }
    });

    // Reset button
    const btnReset = document.getElementById("btn-reset-vitals");
    if (btnReset) {
        btnReset.addEventListener("click", () => {
            resetVitalsForm(DEFAULT_VITALS);
        });
    }

    function resetVitalsForm(vitals) {
        document.getElementById("slider-age").value = vitals.age;
        document.getElementById("val-age").textContent = vitals.age;
        
        document.getElementById("slider-temp").value = vitals.temp;
        document.getElementById("val-temp").textContent = vitals.temp;
        
        document.getElementById("slider-spo2").value = vitals.spo2;
        document.getElementById("val-spo2").textContent = vitals.spo2;
        
        document.getElementById("slider-wbc").value = vitals.wbc;
        document.getElementById("val-wbc").textContent = vitals.wbc;
        
        document.getElementById("slider-hr").value = vitals.hr;
        document.getElementById("val-hr").textContent = vitals.hr;
        
        document.getElementById("slider-sbp").value = vitals.sbp;
        document.getElementById("val-sbp").textContent = vitals.sbp;
        
        document.getElementById("select-gender").value = vitals.gender;
        document.getElementById("select-view").value = vitals.view_pos;
        document.getElementById("select-cough").value = vitals.cough;
        document.getElementById("select-pain").value = vitals.pain;
    }

    // -----------------------------------------
    // FILE UPLOAD AND DRAG-AND-DROP
    // -----------------------------------------
    const dropZone = document.getElementById("drag-drop-zone");
    const fileInput = document.getElementById("image-file");
    const dropPrompt = document.getElementById("drop-prompt");
    const dropPreviewContainer = document.getElementById("drop-preview-container");
    const imagePreview = document.getElementById("image-preview");
    const btnRemoveImage = document.getElementById("btn-remove-image");

    if (dropZone) {
        dropZone.addEventListener("click", (e) => {
            if (e.target.id !== "btn-remove-image" && !btnRemoveImage.contains(e.target)) {
                fileInput.click();
            }
        });
        
        dropZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropZone.style.borderColor = "var(--accent-cyan)";
        });

        dropZone.addEventListener("dragleave", () => {
            dropZone.style.borderColor = "rgba(255, 255, 255, 0.1)";
        });

        dropZone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropZone.style.borderColor = "rgba(255, 255, 255, 0.1)";
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleImageSelect(files[0]);
            }
        });
    }

    if (fileInput) {
        fileInput.addEventListener("change", () => {
            if (fileInput.files.length > 0) {
                handleImageSelect(fileInput.files[0]);
            }
        });
    }

    if (btnRemoveImage) {
        btnRemoveImage.addEventListener("click", (e) => {
            e.stopPropagation(); // prevent opening file browser
            clearImageUpload();
        });
    }

    function handleImageSelect(file) {
        // Validation
        if (!file.type.startsWith("image/")) {
            alert("Please upload an image file.");
            return;
        }
        
        currentSampleImageBlob = file; // Save reference
        selectedSampleImageIndex = null;
        
        // Remove active highlights from sample cards
        document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));
        
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropPrompt.style.display = "none";
            dropPreviewContainer.style.display = "flex";
            
            // Set base image for Grad-CAM overlay
            document.getElementById("gradcam-base-img").src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    function clearImageUpload() {
        fileInput.value = "";
        imagePreview.src = "#";
        dropPreviewContainer.style.display = "none";
        dropPrompt.style.display = "flex";
        currentSampleImageBlob = null;
        selectedSampleImageIndex = null;
        document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));
    }

    // -----------------------------------------
    // CURATED CASE STUDIES (SAMPLES) LOADER
    // -----------------------------------------
    const samplesContainer = document.getElementById("samples-container");

    async function loadSampleCases() {
        try {
            const response = await fetch("/api/samples");
            if (!response.ok) throw new Error("Failed to load sample cases.");
            
            const samples = await response.json();
            samplesContainer.innerHTML = ""; // Clear skeletons
            
            samples.forEach((sample, index) => {
                const label = sample['Finding Labels'] === 'No Finding' ? 'Normal' : sample['Finding Labels'].split('|')[0];
                const hasFinding = sample['Finding Labels'] !== 'No Finding';
                
                const card = document.createElement("div");
                card.className = "sample-card";
                card.innerHTML = `
                    <div class="sample-title">Patient #${sample['Patient ID']}</div>
                    <div class="sample-meta">
                        <span>Age: ${sample['Patient Age']} | ${sample['Patient Gender']}</span>
                        <span class="sample-badge ${hasFinding ? 'has-finding' : 'normal'}">${label}</span>
                    </div>
                `;
                
                card.addEventListener("click", () => {
                    document.querySelectorAll(".sample-card").forEach(c => c.classList.remove("active"));
                    card.classList.add("active");
                    loadSampleIntoForm(sample, index);
                });
                
                samplesContainer.appendChild(card);
            });
        } catch (error) {
            console.error("Error loading sample cases:", error);
            samplesContainer.innerHTML = `<div class="alert-box note" style="grid-column: span 4;">Failed to load sample cases from server.</div>`;
        }
    }

    async function loadSampleIntoForm(sample, index) {
        selectedSampleImageIndex = sample['Image Index'];
        
        // Show loading state in upload box
        dropPrompt.style.display = "none";
        dropPreviewContainer.style.display = "flex";
        imagePreview.src = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='128' height='128'><rect width='128' height='128' fill='%2305060b'/><text x='50%27 y='55%27 fill='white%27 text-anchor='middle%27>Loading Image...</text></svg>";
        
        // Set values in inputs
        resetVitalsForm({
            age: sample['Patient Age'],
            gender: sample['Patient Gender'],
            view_pos: sample['View Position'],
            temp: sample['Body Temperature'],
            spo2: sample['Oxygen Saturation'],
            wbc: sample['WBC Count'],
            hr: sample['Heart Rate'],
            sbp: sample['Systolic BP'],
            cough: sample['Cough Severity'],
            pain: sample['Chest Pain']
        });
        
        // Fetch image file from backend and save as blob
        try {
            const imgUrl = `/api/images/${sample['Image Index']}`;
            const imgResponse = await fetch(imgUrl);
            if (!imgResponse.ok) throw new Error("Sample image not found.");
            
            const blob = await imgResponse.blob();
            currentSampleImageBlob = new File([blob], sample['Image Index'], { type: "image/png" });
            
            // Show preview
            const objectURL = URL.createObjectURL(blob);
            imagePreview.src = objectURL;
            document.getElementById("gradcam-base-img").src = objectURL;
        } catch (error) {
            console.error("Failed to load sample image binary:", error);
            imagePreview.src = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='128' height='128'><rect width='128' height='128' fill='%23200505'/><text x='50%27 y='55%27 fill='red%27 text-anchor='middle%27>Image Load Error</text></svg>";
            currentSampleImageBlob = null;
        }
    }

    // -----------------------------------------
    // EXECUTE DIAGNOSIS & INTERPRETATION
    // -----------------------------------------
    const diagnosticForm = document.getElementById("diagnostic-form");
    const resultsEmptyState = document.getElementById("results-empty-state");
    const resultsDataPanel = document.getElementById("results-data-panel");
    const findingsList = document.getElementById("findings-predictions-list");
    const explainClassSelect = document.getElementById("select-explain-class");
    const explainWrapper = document.getElementById("explain-class-select-wrapper");
    const btnDiagnose = document.getElementById("btn-run-diagnosis");
    
    // Save last diagnostic response to allow switching Grad-CAM class on the fly
    let lastDiagnosticResponse = null;

    if (diagnosticForm) {
        diagnosticForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            
            if (!currentSampleImageBlob) {
                alert("Please upload a chest X-ray image or select a patient case study first.");
                return;
            }
            
            btnDiagnose.disabled = true;
            btnDiagnose.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Analyzing Multimodal Features...`;
            
            const formData = new FormData(diagnosticForm);
            
            // Append file if it was loaded
            formData.set("file", currentSampleImageBlob);
            
            // Set initial class to compute Grad-CAM for (use the dropdown if it was previously filled, otherwise default to Cardiomegaly)
            const targetExplainClass = explainClassSelect.value || "Cardiomegaly";
            formData.set("explain_class", targetExplainClass);
            
            try {
                const response = await fetch("/api/predict", {
                    method: "POST",
                    body: formData
                });
                
                if (!response.ok) {
                    const errData = await response.json();
                    throw new Error(errData.message || "Diagnostic failed.");
                }
                
                const result = await response.json();
                lastDiagnosticResponse = result;
                
                renderDiagnosisResults(result, targetExplainClass);
                
            } catch (error) {
                console.error("Diagnosis error:", error);
                alert(`Diagnostic processing error: ${error.message}. Ensure the model is trained.`);
            } finally {
                btnDiagnose.disabled = false;
                btnDiagnose.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Execute Multimodal Diagnosis`;
            }
        });
    }

    // Handle changing the Grad-CAM focus class
    explainClassSelect.addEventListener("change", async () => {
        if (!lastDiagnosticResponse || !currentSampleImageBlob) return;
        
        const newClass = explainClassSelect.value;
        btnDiagnose.disabled = true;
        
        // Re-call predict but only requesting the new Grad-CAM grid
        const formData = new FormData(diagnosticForm);
        formData.set("file", currentSampleImageBlob);
        formData.set("explain_class", newClass);
        
        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                body: formData
            });
            if (response.ok) {
                const result = await response.json();
                lastDiagnosticResponse = result;
                // Render Grad-CAM grid for the new class
                renderGradcamGrid(result.grad_cam_grid);
            }
        } catch (e) {
            console.error("GradCAM change error:", e);
        } finally {
            btnDiagnose.disabled = false;
        }
    });

    function renderDiagnosisResults(result, selectedClass) {
        // Hide empty state and show results panel
        resultsEmptyState.style.display = "none";
        resultsDataPanel.style.display = "flex";
        
        // Populate findings list
        findingsList.innerHTML = "";
        
        // Sort predictions by confidence
        const preds = Object.entries(result.predictions).sort((a, b) => b[1] - a[1]);
        
        preds.forEach(([name, prob]) => {
            const pct = (prob * 100).toFixed(1);
            let riskClass = "normal";
            if (prob > 0.5) riskClass = "high-risk";
            else if (prob > 0.15) riskClass = "med-risk";
            else if (prob > 0.03) riskClass = "low-risk";
            
            const row = document.createElement("div");
            row.className = "finding-row";
            row.innerHTML = `
                <span class="finding-name" title="${name}">${name}</span>
                <div class="finding-prob-bar-wrapper">
                    <div class="finding-prob-bar ${riskClass}" style="width: ${pct}%;"></div>
                </div>
                <span class="finding-percentage">${pct}%</span>
            `;
            findingsList.appendChild(row);
        });
        
        // Populate explainability dropdown selector
        explainWrapper.style.display = "flex";
        explainClassSelect.innerHTML = "";
        
        FINDINGS.forEach(name => {
            const opt = document.createElement("option");
            opt.value = name;
            opt.textContent = name;
            if (name === selectedClass) opt.selected = true;
            explainClassSelect.appendChild(opt);
        });

        // Draw Gradcam Grid
        renderGradcamGrid(result.grad_cam_grid);
        
        // Draw clinical attention weights chart
        renderClinicalAttentionChart(result.clinical_attention);
    }

    function renderGradcamGrid(grid) {
        const canvas = document.getElementById("gradcam-canvas");
        const ctx = canvas.getContext("2d");
        const baseImg = document.getElementById("gradcam-base-img");
        
        // Ensure image is loaded before drawing
        if (baseImg.complete && baseImg.src !== "#") {
            drawCamOverlay();
        } else {
            baseImg.onload = () => {
                drawCamOverlay();
            };
        }
        
        function drawCamOverlay() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            // Draw clear grayscale chest X-ray
            ctx.drawImage(baseImg, 0, 0, canvas.width, canvas.height);
            
            // Create offscreen canvas for heatmap blur
            const offscreen = document.createElement("canvas");
            offscreen.width = 128;
            offscreen.height = 128;
            const oCtx = offscreen.getContext("2d");
            
            const cellW = offscreen.width / 8;
            const cellH = offscreen.height / 8;
            
            for (let r = 0; r < 8; r++) {
                for (let c = 0; c < 8; c++) {
                    const val = grid[r][c];
                    if (val > 0.05) {
                        // Map 1.0 to red (0), 0.0 to blue (240)
                        const hue = (1.0 - val) * 240;
                        oCtx.fillStyle = `hsla(${hue}, 100%, 50%, ${val * 0.7})`;
                        oCtx.fillRect(c * cellW, r * cellH, cellW, cellH);
                    }
                }
            }
            
            // Layer the blurred heat-map overlay onto main canvas
            ctx.save();
            ctx.globalAlpha = 0.8;
            ctx.filter = 'blur(8px)';
            ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
            ctx.restore();
        }
    }

    function renderClinicalAttentionChart(attnData) {
        const ctx = document.getElementById("clinical-attention-chart").getContext("2d");
        
        const labels = Object.keys(attnData);
        const data = Object.values(attnData);
        
        if (clinicalAttentionChartInstance) {
            clinicalAttentionChartInstance.destroy();
        }
        
        clinicalAttentionChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Attention Weight (%)',
                    data: data,
                    backgroundColor: 'rgba(177, 84, 252, 0.65)',
                    borderColor: 'rgba(177, 84, 252, 1)',
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (context) => `Weight: ${context.parsed.x.toFixed(1)}%`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#9ca3af', font: { family: 'Outfit' } },
                        max: 100
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: '#e5e7eb', font: { family: 'Outfit', size: 10 } }
                    }
                }
            }
        });
    }

    // -----------------------------------------
    // ENGINE TRAINING WORKFLOW MONITORING
    // -----------------------------------------
    const trainingForm = document.getElementById("training-form");
    const btnTrain = document.getElementById("btn-start-training");
    const progressBar = document.getElementById("progress-bar");
    const statusMsg = document.getElementById("status-msg");
    const statusPercent = document.getElementById("status-percent");
    const consoleLogs = document.getElementById("console-logs");

    if (trainingForm) {
        trainingForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            
            btnTrain.disabled = true;
            btnTrain.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Initializing Pipeline...`;
            
            const formData = new FormData(trainingForm);
            
            try {
                const response = await fetch("/api/train", {
                    method: "POST",
                    body: formData
                });
                
                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.message || "Failed to start training.");
                }
                
                // Clear console
                consoleLogs.innerHTML = `<div class="log-line system">[SYSTEM] Training sequence queued in backend. Starting status monitor...</div>`;
                
                // Poll status
                startStatusPolling();
                
            } catch (error) {
                console.error("Training trigger error:", error);
                alert(`Training Error: ${error.message}`);
                btnTrain.disabled = false;
                btnTrain.innerHTML = `<i class="fa-solid fa-play"></i> Initiate Training`;
            }
        });
    }

    function startStatusPolling() {
        if (trainingPollInterval) clearInterval(trainingPollInterval);
        
        trainingPollInterval = setInterval(async () => {
            try {
                const res = await fetch("/api/status");
                const status = await res.json();
                
                // Update text and progress bar
                statusMsg.textContent = status.message;
                
                let pct = 0;
                if (status.is_training) {
                    // Estimate percentage based on logs or epochs
                    // Since epochs is a simple division, we can estimate
                    const lines = consoleLogs.querySelectorAll(".log-line").length;
                    pct = Math.min((lines / 50) * 100, 95); // cap at 95 until done
                    
                    // Add logs to console if message changes
                    addConsoleLog(status.message);
                } else if (status.completed) {
                    pct = 100;
                    addConsoleLog("[SUCCESS] Model trained and updated. Metrics calculated.");
                    clearInterval(trainingPollInterval);
                    
                    // Reset buttons
                    btnTrain.disabled = false;
                    btnTrain.innerHTML = `<i class="fa-solid fa-play"></i> Initiate Training`;
                    
                    // Reload evaluation and case studies
                    fetchEvaluationMetrics();
                    loadSampleCases();
                } else {
                    pct = 0;
                    clearInterval(trainingPollInterval);
                    btnTrain.disabled = false;
                    btnTrain.innerHTML = `<i class="fa-solid fa-play"></i> Initiate Training`;
                }
                
                progressBar.style.width = `${pct}%`;
                statusPercent.textContent = `${Math.round(pct)}%`;
                
            } catch (e) {
                console.error("Polling error:", e);
            }
        }, 1000);
    }

    function addConsoleLog(message) {
        // Only append if last log is different
        const lastLog = consoleLogs.lastElementChild;
        if (!lastLog || lastLog.textContent !== message) {
            const isError = message.toLowerCase().includes("failed") || message.toLowerCase().includes("error");
            const div = document.createElement("div");
            div.className = `log-line ${isError ? 'error' : ''}`;
            div.textContent = `[PROCESS] ${message}`;
            consoleLogs.appendChild(div);
            consoleLogs.scrollTop = consoleLogs.scrollHeight; // Autoscroll
        }
    }

    // -----------------------------------------
    // PERFORMANCE ANALYTICS & COMPARATIVE TABLES
    // -----------------------------------------
    const metricsTableBody = document.getElementById("metrics-table-body");

    async function fetchEvaluationMetrics() {
        try {
            const response = await fetch("/api/evaluate");
            if (!response.ok) {
                // Not trained yet
                metricsTableBody.innerHTML = `
                    <tr>
                        <td colspan="6" class="table-empty">
                            No performance records found. Open the System Overview tab and train the engine to view benchmarking.
                        </td>
                    </tr>
                `;
                return;
            }
            
            const results = await response.json();
            
            // 1. Populate summary table
            metricsTableBody.innerHTML = "";
            const modelNames = ['Multimodal', 'Image-only', 'Clinical-only'];
            
            modelNames.forEach(name => {
                const data = results[name];
                if (!data) return;
                
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td style="font-weight: 600;">${name === 'Multimodal' ? '<i class="fa-solid fa-star" style="color: var(--accent-cyan);"></i> ' : ''}${name}</td>
                    <td>${(data.metrics.accuracy).toFixed(4)}</td>
                    <td>${(data.metrics.precision).toFixed(4)}</td>
                    <td>${(data.metrics.recall).toFixed(4)}</td>
                    <td>${(data.metrics.f1).toFixed(4)}</td>
                    <td style="font-weight: 600; color: var(--accent-cyan);">${(data.metrics.mean_auc).toFixed(4)}</td>
                `;
                metricsTableBody.appendChild(tr);
            });
            
            // Update big stat cards
            document.getElementById("stat-auc-multimodal").textContent = results['Multimodal'] ? results['Multimodal'].metrics.mean_auc.toFixed(3) : "--";
            document.getElementById("stat-auc-image").textContent = results['Image-only'] ? results['Image-only'].metrics.mean_auc.toFixed(3) : "--";
            document.getElementById("stat-auc-clinical").textContent = results['Clinical-only'] ? results['Clinical-only'].metrics.mean_auc.toFixed(3) : "--";
            
            // 2. Draw benchmarking chart (Accuracy, F1, AUC side-by-side comparison)
            drawBenchmarkingChart(results);
            
            // 3. Draw class-wise multi-label AUC bar chart
            drawClassBenchmarkingChart(results['Multimodal'].class_auc);
            
        } catch (e) {
            console.error("Failed to load metrics:", e);
        }
    }

    function drawBenchmarkingChart(results) {
        const ctx = document.getElementById("performance-comparison-chart").getContext("2d");
        
        const models = ['Multimodal', 'Image-only', 'Clinical-only'];
        const metrics = ['accuracy', 'f1', 'mean_auc'];
        const datasets = [];
        
        const colors = {
            'Multimodal': { bg: 'rgba(0, 242, 254, 0.65)', border: 'rgba(0, 242, 254, 1)' },
            'Image-only': { bg: 'rgba(0, 112, 243, 0.5)', border: 'rgba(0, 112, 243, 1)' },
            'Clinical-only': { bg: 'rgba(243, 85, 136, 0.5)', border: 'rgba(243, 85, 136, 1)' }
        };
        
        models.forEach(modelName => {
            if (!results[modelName]) return;
            datasets.push({
                label: modelName,
                data: [
                    results[modelName].metrics.accuracy,
                    results[modelName].metrics.f1,
                    results[modelName].metrics.mean_auc
                ],
                backgroundColor: colors[modelName].bg,
                borderColor: colors[modelName].border,
                borderWidth: 1,
                borderRadius: 4
            });
        });
        
        if (performanceChartInstance) performanceChartInstance.destroy();
        
        performanceChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Accuracy', 'F1-Score', 'Mean ROC-AUC'],
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#e5e7eb', font: { family: 'Outfit' } }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { color: '#9ca3af', font: { family: 'Outfit' } }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#9ca3af', font: { family: 'Outfit' } },
                        min: 0.4,
                        max: 1.0
                    }
                }
            }
        });
    }

    function drawClassBenchmarkingChart(classAuc) {
        const ctx = document.getElementById("class-comparison-chart").getContext("2d");
        
        const labels = Object.keys(classAuc);
        const data = Object.values(classAuc);
        
        if (classChartInstance) classChartInstance.destroy();
        
        classChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'ROC-AUC Score',
                    data: data,
                    backgroundColor: 'rgba(5, 213, 140, 0.65)',
                    borderColor: 'rgba(5, 213, 140, 1)',
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { 
                            color: '#9ca3af', 
                            font: { family: 'Outfit', size: 9 },
                            maxRotation: 45,
                            minRotation: 45
                        }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#9ca3af', font: { family: 'Outfit' } },
                        min: 0.5,
                        max: 1.0
                    }
                }
            }
        });
    }

    // -----------------------------------------
    // INITIALIZATION RUN
    // -----------------------------------------
    loadSampleCases();
    
    // Check if background training is already active
    fetch("/api/status").then(r => r.json()).then(status => {
        if (status.is_training) {
            progressBar.style.width = `15%`;
            statusPercent.textContent = `15%`;
            statusMsg.textContent = status.message;
            btnTrain.disabled = true;
            btnTrain.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Restoring Monitor...`;
            addConsoleLog(status.message);
            startStatusPolling();
        }
    });
});
