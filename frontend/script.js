const imageInput = document.getElementById('imageInput');
const uploadBtn = document.getElementById('uploadBtn');
const fileName = document.getElementById('fileName');
const imagePreview = document.getElementById('imagePreview');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');

const API_URL = 'http://localhost:5000/api';

let selectedFile = null;

uploadBtn.addEventListener('click', () => {
    imageInput.click();
});

imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        selectedFile = file;
        fileName.textContent = file.name;

        // Preview image
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
            analyzeBtn.style.display = 'block';
            results.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }
});

analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        alert('Please select an image first!');
        return;
    }

    const formData = new FormData();
    formData.append('image', selectedFile);

    // Show loading
    loading.style.display = 'block';
    results.style.display = 'none';
    analyzeBtn.style.display = 'none';

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Failed to analyze image');
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        loading.style.display = 'none';
        analyzeBtn.style.display = 'block';
    }
});

function displayResults(data) {
    // Display caption
    const captionElement = document.getElementById('caption');
    if (data.caption_status === 'success') {
        captionElement.textContent = data.caption;
    } else {
        captionElement.textContent = '⚠️ ' + data.caption;
        captionElement.style.color = '#e74c3c';
    }

    // Clear previous predictions
    document.getElementById('captionPredictions').innerHTML = '<p>Generated caption shown above</p>';

    // Display action
    const actionResult = document.getElementById('actionResult');
    if (data.action_status === 'success') {
        actionResult.textContent = `Action: ${data.action.action.toUpperCase()} (${(data.action.confidence * 100).toFixed(2)}%)`;
        actionResult.style.background = '#667eea';

        // Display top predictions
        const actionPredictions = document.getElementById('actionPredictions');
        actionPredictions.innerHTML = data.action.all_predictions.map(pred => `
            <div class="prediction-item">
                <span class="prediction-class">${pred.action}</span>
                <span class="prediction-confidence">${(pred.confidence * 100).toFixed(2)}%</span>
            </div>
        `).join('');
    } else {
        actionResult.textContent = '⚠️ ' + data.action.action;
        actionResult.style.background = '#e74c3c';
        document.getElementById('actionPredictions').innerHTML = '';
    }

    results.style.display = 'block';
}