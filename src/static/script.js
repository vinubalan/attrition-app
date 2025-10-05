const form = document.getElementById('predictionForm');
const predictBtn = document.getElementById('predictBtn');
const resultSection = document.getElementById('resultSection');
const resultCard = document.getElementById('resultCard');
const resultIcon = document.getElementById('resultIcon');
const predictionText = document.getElementById('predictionText');
const probabilityFill = document.getElementById('probabilityFill');
const probabilityValue = document.getElementById('probabilityValue');
const riskBadge = document.getElementById('riskBadge');

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Disable button and show loading
    predictBtn.disabled = true;
    predictBtn.classList.add('loading');
    predictBtn.textContent = 'Predicting...';
    
    // Get form data
    const formData = new FormData(form);
    const data = {};
    formData.forEach((value, key) => {
        // Convert to number if it's a numeric field
        data[key] = isNaN(value) ? value : Number(value);
    });
    
    try {
        // Make API call
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const result = await response.json();
        
        // Display results
        displayResults(result);
        
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to get prediction. Please check if all fields are filled correctly.');
    } finally {
        // Re-enable button
        predictBtn.disabled = false;
        predictBtn.classList.remove('loading');
        predictBtn.textContent = '🔮 Predict Attrition';
    }
});

function displayResults(result) {
    // Show result section
    resultSection.style.display = 'block';
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    
    // Update prediction text and icon
    if (result.prediction === 'Will Leave') {
        predictionText.textContent = '⚠️ High Risk of Attrition';
        predictionText.style.color = '#dc2626';
        resultIcon.textContent = '⚠️';
    } else {
        predictionText.textContent = '✅ Low Risk of Attrition';
        predictionText.style.color = '#16a34a';
        resultIcon.textContent = '✅';
    }
    
    // Update probability bar
    const probability = result.probability * 100;
    probabilityFill.style.width = `${probability}%`;
    probabilityValue.textContent = `${probability.toFixed(1)}%`;
    
    // Update risk badge
    riskBadge.textContent = `${result.risk_level} Risk`;
    riskBadge.className = 'risk-badge';
    
    if (result.risk_level === 'Low') {
        riskBadge.classList.add('risk-low');
    } else if (result.risk_level === 'Medium') {
        riskBadge.classList.add('risk-medium');
    } else {
        riskBadge.classList.add('risk-high');
    }
}