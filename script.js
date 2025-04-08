function checkURL() {
    let url = document.getElementById("urlInput").value;
    let resultText = document.getElementById("result");

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            resultText.innerText = "Error: " + data.error;
            resultText.style.color = "red";
        } else {
            let msg = data.is_phishing ? "⚠️ Phishing URL Detected!" : "✅ Safe URL";
            resultText.innerText = msg;
            resultText.style.color = data.is_phishing ? "red" : "green";
        }
    })
    .catch(error => {
        resultText.innerText = "Error: Could not connect to server.";
        resultText.style.color = "red";
    });
}
