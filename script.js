// JavaScript for Drug Side Effect Prediction

// DOM Elements
const form = document.getElementById('prediction-form');
const drugInput = document.getElementById('drug-input');
const predictBtn = document.getElementById('predict-btn');
const loading = document.getElementById('loading');
const errorMessage = document.getElementById('error-message');
const results = document.getElementById('results');
const autocompleteList = document.getElementById('autocomplete-list');

// State
let autocompleteData = [];
let debounceTimer = null;

// Event Listeners
form.addEventListener('submit', handleSubmit);
drugInput.addEventListener('input', handleAutocomplete);
drugInput.addEventListener('focus', () => {
    if (autocompleteData.length > 0) {
        autocompleteList.classList.add('active');
    }
});

// Click outside to close autocomplete
document.addEventListener('click', (e) => {
    if (!e.target.closest('.input-wrapper')) {
        autocompleteList.classList.remove('active');
    }
});

// Handle form submission
async function handleSubmit(e) {
    e.preventDefault();
    
    const drugName = drugInput.value.trim();
    
    if (!drugName) {
        showError('Please enter a drug name');
        return;
    }
    
    // Hide previous results and errors
    hideError();
    hideResults();
    
    // Show loading
    showLoading();
    
    try {
        // Make prediction request
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ drug_name: drugName })
        });
        
        const data = await response.json();
        
        // Hide loading
        hideLoading();
        
        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'An error occurred during prediction');
        }
    } catch (error) {
        hideLoading();
        showError('Network error. Please check your connection and try again.');
        console.error('Error:', error);
    }
}

// Handle autocomplete
function handleAutocomplete(e) {
    const query = e.target.value.trim();
    
    // Clear previous timer
    clearTimeout(debounceTimer);
    
    if (query.length < 2) {
        autocompleteList.classList.remove('active');
        return;
    }
    
    // Debounce API call
    debounceTimer = setTimeout(async () => {
        try {
            const response = await fetch(`/api/drugs?q=${encodeURIComponent(query)}`);
            const data = await response.json();
            
            if (data.success && data.drugs.length > 0) {
                autocompleteData = data.drugs;
                displayAutocomplete(data.drugs, query);
            } else {
                autocompleteList.classList.remove('active');
            }
        } catch (error) {
            console.error('Autocomplete error:', error);
        }
    }, 300);
}

// Display autocomplete suggestions
function displayAutocomplete(drugs, query) {
    autocompleteList.innerHTML = '';
    
    drugs.forEach(drug => {
        const item = document.createElement('div');
        item.className = 'autocomplete-item';
        
        // Highlight matching text
        const name = drug.name;
        const regex = new RegExp(`(${query})`, 'gi');
        const highlightedName = name.replace(regex, '<strong>$1</strong>');
        
        item.innerHTML = highlightedName;
        
        item.addEventListener('click', () => {
            drugInput.value = drug.name;
            autocompleteList.classList.remove('active');
        });
        
        autocompleteList.appendChild(item);
    });
    
    autocompleteList.classList.add('active');
}

// Display prediction results
function displayResults(data) {
    // Set drug name
    document.getElementById('result-drug-name').textContent = data.drug_name;
    
    // Display side effects
    const sideEffectsList = document.getElementById('side-effects-list');
    sideEffectsList.innerHTML = '';
    
    data.side_effects.forEach((effect, index) => {
        const item = document.createElement('div');
        item.className = `side-effect-item severity-${effect.severity}`;
        item.style.animationDelay = `${index * 0.05}s`;
        
        item.innerHTML = `
            <div class="side-effect-header">
                <span class="side-effect-name">${effect.name}</span>
                <span class="side-effect-probability">${effect.percentage.toFixed(1)}%</span>
            </div>
            <div class="probability-bar">
                <div class="probability-fill" style="width: ${effect.percentage}%"></div>
            </div>
        `;
        
        sideEffectsList.appendChild(item);
    });
    
    // Display precautions
    const precautionsList = document.getElementById('precautions-list');
    precautionsList.innerHTML = '';
    
    data.precautions.forEach((precaution, index) => {
        const item = document.createElement('li');
        item.textContent = precaution;
        item.style.animationDelay = `${index * 0.05}s`;
        precautionsList.appendChild(item);
    });
    
    // Show results
    showResults();
}

// Show/Hide functions
function showLoading() {
    loading.classList.remove('hidden');
    predictBtn.disabled = true;
}

function hideLoading() {
    loading.classList.add('hidden');
    predictBtn.disabled = false;
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove('hidden');
}

function hideError() {
    errorMessage.classList.add('hidden');
}

function showResults() {
    results.classList.remove('hidden');
    
    // Scroll to results
    setTimeout(() => {
        results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}

function hideResults() {
    results.classList.add('hidden');
}
