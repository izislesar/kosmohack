function renderAnalytics(data, fireCount) {

document.getElementById("analytics-fires").textContent =
    fireCount;

document.getElementById("fire-count").textContent =
    fireCount;

document.getElementById("analytics-area").textContent =
    data.totalAreaHa.toFixed(1);

document.getElementById("analytics-weak").textContent =
    data.severity.weak.toFixed(1);

document.getElementById("analytics-medium").textContent =
    data.severity.medium.toFixed(1);

document.getElementById("analytics-strong").textContent =
    data.severity.strong.toFixed(1);

}