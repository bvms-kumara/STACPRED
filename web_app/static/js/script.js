// --- OOD Configuration ---
const MAX_MOL_WEIGHT_LIMIT = 550.0; 
const FINAL_OOD_VALUE = 50.0; // Must match the value returned by prediction_core.py
// -------------------------

// --- Global Variable Declarations (All elements used must be defined) ---
const apiEndpoint = document.getElementById('api-endpoint').textContent;
const uploadForm = document.getElementById('uploadForm');
const fileInput = document.getElementById('mol2_upload');
const submitButton = document.getElementById('submit-button');
const dropZone = document.getElementById('drop-zone');
const dropText = document.getElementById('drop-text');
const statusAlert = document.getElementById('status-alert');
const flowContainer = document.getElementById('flow-container');
const initialMessage = document.getElementById('initial-message');
const inputBox = document.getElementById('input-box');
const processingBox = document.getElementById('processing-box');
const metaModelBox = document.getElementById('meta-model-box');
const baseModelGrid = document.getElementById('base-model-grid');
const modelTemplate = document.getElementById('model-template');

// Static R2 values for visualization
const MODEL_R2_SCORES = {
    'AD4_XGB_PRED': 0.6910, 'DOCK6_XGB_PRED': 0.4505, 'VINA_XGB_PRED': 0.9080, 'VINARDO_XGB_PRED': 0.6001,
    'AD4_MLP_PRED': 0.4815, 'DOCK6_MLP_PRED': 0.3850, 'VINA_MLP_PRED': 0.9057, 'VINARDO_MLP_PRED': 0.6037
};

// Model ordering for consistent display
const MODEL_ORDER = Object.keys(MODEL_R2_SCORES).sort(([keyA], [keyB]) => {
    const order = ['VINA', 'AD4', 'VINARDO', 'DOCK6'];
    const indexA = order.findIndex(prefix => keyA.startsWith(prefix));
    const indexB = order.findIndex(prefix => keyB.startsWith(prefix));
    if (indexA === indexB) { return keyA.includes('XGB') ? -1 : 1; }
    return indexA - indexB;
});

// --- Initialization and Utility Functions ---

function initializeModelGrid() {
    baseModelGrid.innerHTML = '';
    MODEL_ORDER.forEach(modelName => {
        const isXGB = modelName.includes('XGB');
        const clone = modelTemplate.content.cloneNode(true);
        const card = clone.querySelector('.model-card');
        
        card.id = `model-card-${modelName}`;
        card.setAttribute('data-r2', MODEL_R2_SCORES[modelName]);
        card.querySelector('[data-id="name"]').textContent = modelName.replace('_PRED', '');
        card.querySelector('[data-id="bar"]').id = `bar-${modelName}`;
        card.querySelector('[data-id="score"]').id = `score-${modelName}`;

        card.querySelector('[data-id="name"]').classList.add(isXGB ? 'text-cyan-400' : 'text-fuchsia-400');
        
        baseModelGrid.appendChild(card);
    });
}

function showAlert(message, type = 'info') {
    statusAlert.classList.remove('hidden', 'bg-red-700', 'bg-green-700', 'bg-yellow-700', 'bg-blue-900', 'border-red-500', 'border-green-500', 'border-blue-500');
    statusAlert.textContent = message;
    // Use dark theme appropriate colors
    if (type === 'error') {
        statusAlert.classList.add('bg-red-700', 'border-red-500');
    } else if (type === 'success') {
        statusAlert.classList.add('bg-green-700', 'border-green-500');
    } else {
        statusAlert.classList.add('bg-blue-900', 'border-blue-500');
    }
}

function toggleButtonState(enabled, text = 'Initiate Ensemble Prediction') {
    submitButton.disabled = !enabled;
    submitButton.textContent = text;
    if (enabled) {
        submitButton.classList.remove('bg-gray-500', 'hover:bg-gray-600');
        submitButton.classList.add('bg-cyan-600', 'hover:bg-cyan-500', 'text-black');
    } else {
        submitButton.classList.add('bg-gray-500', 'hover:bg-gray-600');
        submitButton.classList.remove('bg-cyan-600', 'hover:bg-cyan-500', 'text-black');
    }
}

function resetFlowDiagram() {
    flowContainer.classList.add('hidden');
    initialMessage.classList.remove('hidden');
    inputBox.classList.remove('active-pulse', 'bg-cyan-500', 'text-black');
    metaModelBox.classList.remove('active-pulse', 'bg-cyan-500');
    metaModelBox.style.backgroundColor = '';
    metaModelBox.querySelector('.text-xl').classList.remove('text-cyan-200');
    metaModelBox.querySelector('.text-xl').classList.add('text-cyan-400');

    document.getElementById('flow-filename').textContent = 'Awaiting File...';
    document.getElementById('flow-result-filename').textContent = '--';
    document.getElementById('final-score').textContent = '--';

    document.querySelectorAll('.model-bar').forEach(bar => {
        bar.style.width = '0%';
        bar.classList.remove('model-active', 'model-mlp-active');
    });
    document.querySelectorAll('.model-card').forEach(card => {
        card.classList.remove('active-pulse', 'border-cyan-400');
        card.style.borderColor = '#00c0ff40';
        card.classList.remove('ood-card');
        card.querySelector('[data-id="name"]').classList.remove('text-red-400');
    });
    document.querySelectorAll('[data-id="score"]').forEach(score => {
        score.textContent = 'Score: N/A';
    });
}

function resetUI() {
    statusAlert.classList.add('hidden');
    dropZone.classList.remove('drop-zone-active');
    dropText.innerHTML = '<span class="font-bold text-cyan-300">Click to upload</span> or drag and drop.';
    fileInput.value = null; 
    toggleButtonState(false);
    resetFlowDiagram();
}

function animatePrediction(result) {
    const baseScores = result.base_scores;
    const finalScore = parseFloat(result.final_score); 
    const FINAL_OOD_VALUE = 50.0;
    
    // 1. Highlight Input Module
    inputBox.classList.add('active-pulse', 'bg-cyan-500', 'text-black');
    document.getElementById('flow-filename').textContent = result.filename;
    initialMessage.classList.add('hidden');
    flowContainer.classList.remove('hidden');
    
    // 2. Feature Extraction Delay
    setTimeout(() => {
        inputBox.classList.remove('active-pulse', 'bg-cyan-500', 'text-black');
        showAlert('Feature Extraction Complete. Initializing Parallel Models...', 'info');
    }, 1000); 

    // 3. Parallel Base Model Animation
    setTimeout(() => {
        let animationDuration = 0;
        
        MODEL_ORDER.forEach((modelName, index) => {
            const r2 = MODEL_R2_SCORES[modelName];
            const confidence = Math.max(30, Math.round(r2 * 100));
            const isXGB = modelName.includes('XGB');
            
            const bar = document.getElementById(`bar-${modelName}`);
            const scoreDiv = document.getElementById(`score-${modelName}`);
            const card = document.getElementById(`model-card-${modelName}`);
            
            const staggerDelay = index * 50;
            const completionTime = 1200;

            let displayScore = result.base_scores[modelName];
            
            if (Math.abs(displayScore - FINAL_OOD_VALUE) < 0.01) {
                // OOD path: No animation, show OOD text
                 bar.style.width = '0%';
                 scoreDiv.textContent = 'OOD';
                 card.classList.add('ood-card');
                 card.querySelector('[data-id="name"]').classList.add('text-red-400');
            } else {
                // Normal Prediction path: Run animation
                bar.style.width = `${confidence}%`;
                bar.classList.add(isXGB ? 'model-active' : 'model-mlp-active');
                card.classList.add('active-pulse', 'border-cyan-400');
                
                setTimeout(() => {
                    scoreDiv.textContent = `Score: ${displayScore.toFixed(4)}`;
                    card.classList.remove('active-pulse');
                    card.style.borderColor = isXGB ? '#00c0ff' : '#ff00c0';
                }, 100 + staggerDelay + completionTime);
            }
            
            animationDuration = Math.max(animationDuration, staggerDelay + completionTime);
        });
        
        // 4. Meta-Model Fusion Trigger
        setTimeout(() => {
            showAlert('Base Models Complete. Fusing results...', 'info');
            metaModelBox.classList.add('active-pulse');
            metaModelBox.style.backgroundColor = '#00c0ff30';
        }, 100 + animationDuration + 500);

        // 5. Final Result Display
        setTimeout(() => {
            
            // --- FINAL OOD CHECK & DISPLAY ---
            if (Math.abs(finalScore - FINAL_OOD_VALUE) < 0.01) {
                document.getElementById('final-score').textContent = "OOD FAILURE";
                metaModelBox.style.backgroundColor = 'rgba(60, 10, 10, 0.8)';
                metaModelBox.querySelector('.text-xl').textContent = "INPUT OUT OF DOMAIN";
                metaModelBox.querySelector('.text-xl').classList.remove('text-cyan-400', 'text-cyan-200');
                metaModelBox.querySelector('.text-xl').classList.add('text-red-400');
                showAlert(`OOD ERROR: Molecule too large. Prediction blocked.`, 'error');

            } else {
                // Normal successful prediction
                document.getElementById('final-score').textContent = finalScore.toFixed(4) + ' kcal/mol';
                metaModelBox.style.backgroundColor = '#00c0ff20'; 
                metaModelBox.querySelector('.text-xl').textContent = "FINAL CONSENSUS SCORE";
                metaModelBox.querySelector('.text-xl').classList.add('text-cyan-200');
                metaModelBox.querySelector('.text-xl').classList.remove('text-red-400');
                showAlert(`SUCCESS! Final Affinity: ${finalScore.toFixed(4)} kcal/mol`, 'success');
            }
            // --- END OOD CHECK ---

            document.getElementById('flow-result-filename').textContent = result.filename;
            metaModelBox.classList.remove('active-pulse');
            toggleButtonState(true);

        }, 100 + animationDuration + 1500);

    }, 1000);
}

// --- Event Handlers (Initialization) ---
dropZone.addEventListener('click', () => {
    fileInput.click();
});

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});
function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('drop-zone-active'), false);
});
['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('drop-zone-active'), false);
});

dropZone.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length === 1 && files[0].name.toLowerCase().endsWith('.mol2')) {
        fileInput.files = files;
        handleFileInputChange();
    } else {
        showAlert('Input Error: Please drop only one .mol2 file.', 'error');
    }
}
fileInput.addEventListener('change', handleFileInputChange);

uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!fileInput.files.length) { showAlert('Input Error: Select a MOL2 file.', 'error'); return; }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('mol2_file', file);
    
    toggleButtonState(false, 'EXECUTING: Awaiting Server Response...');
    resetFlowDiagram();
    
    showAlert('Sending file to server for processing...', 'info');

    try {
        const response = await fetch(apiEndpoint, { method: 'POST', body: formData });
        const result = await response.json();

        if (response.ok) {
            animatePrediction(result);
        } else {
            const errorMsg = result.error || 'Server error, check console.';
            showAlert(`Prediction failed: ${errorMsg}`, 'error');
            toggleButtonState(true);
        }

    } catch (error) {
        console.error("Network or Fetch Error:", error);
        showAlert(`Connection Error. Check Flask Server console. Error: ${error.message}`, 'error');
        toggleButtonState(true);
    }
});

// Initialize UI State
initializeModelGrid();
resetUI();
