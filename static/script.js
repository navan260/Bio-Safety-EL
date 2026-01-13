document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('analysisForm');
    const generateBtn = document.getElementById('generateBtn');
    const resultSection = document.getElementById('resultSection');
    const resultContent = document.getElementById('resultContent');
    const downloadBtn = document.getElementById('downloadBtn');
    const btnContent = generateBtn.querySelector('.btn-content');
    const spinner = generateBtn.querySelector('.loading-spinner');
    const offlineToggle = document.getElementById('offlineToggle');

    let currentData = null;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Get values
        const foodName = document.getElementById('foodName').value;
        const weather = document.getElementById('weather').value;
        const duration = document.getElementById('duration').value;
        const context = document.getElementById('context').value;
        const isOffline = offlineToggle.checked;

        // UI Loading State
        setLoading(true);
        resultSection.classList.add('hidden');
        resultContent.textContent = ''; // Clear previous

        try {
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    food_name: foodName,
                    weather: weather,
                    storage_duration: duration,
                    additional_context: context,
                    offline_mode: isOffline
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Failed to generate analysis');
            }

            // Store data for report generation
            currentData = {
                food_name: foodName,
                weather: weather,
                storage_duration: duration,
                additional_context: context,
                answer: data.answer
            };

            // Display Result
            resultContent.textContent = data.answer;
            resultSection.classList.remove('hidden');

            // Smooth scroll to result
            resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

        } catch (error) {
            console.error(error);
            alert('Error: ' + error.message);
        } finally {
            setLoading(false);
        }
    });

    downloadBtn.addEventListener('click', async () => {
        if (!currentData) return;

        try {
            const originalText = downloadBtn.innerHTML;
            downloadBtn.innerHTML = '<span>Generating...</span>';
            downloadBtn.disabled = true;

            const response = await fetch('/api/report', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(currentData)
            });

            if (!response.ok) {
                throw new Error('Failed to generate report');
            }

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `BioSafety_Report_${new Date().getTime()}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            a.remove();

        } catch (error) {
            alert('Error generating report: ' + error.message);
        } finally {
            downloadBtn.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
                <span>Export PDF</span>
            `;
            downloadBtn.disabled = false;
        }
    });

    function setLoading(isLoading) {
        generateBtn.disabled = isLoading;
        if (isLoading) {
            btnContent.textContent = 'Analyzing...';
            spinner.classList.remove('hidden');
        } else {
            btnContent.textContent = 'Run Analysis';
            spinner.classList.add('hidden');
        }
    }
});
