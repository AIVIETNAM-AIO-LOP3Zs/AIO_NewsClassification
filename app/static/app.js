/**
 * AI BBC News Classifier — Frontend Controller
 *
 * Toggle logic:
 *   - TF-IDF button  →  POST /api/v1/news/classify/tfidf
 *   - BERT   button  →  POST /api/v1/news/classify/bert
 *
 * The active model determines which API route receives the request.
 */

// ── State ──────────────────────────────────────────────────────────────────
const ROUTES = {
    tfidf: "/api/v1/news/classify/tfidf",
    bert: "/api/v1/news/classify/bert",
};

let activeModel = "tfidf";

// ── DOM References ─────────────────────────────────────────────────────────
const btnTfidf = document.getElementById("btn-tfidf");
const btnBert = document.getElementById("btn-bert");
const activeRouteUrl = document.getElementById("active-route-url");
const textArea = document.getElementById("news-content");
const classifyForm = document.getElementById("classify-form");
const btnSubmit = document.getElementById("btn-submit");
const btnSubmitText = document.getElementById("btn-submit-text");
const resultsSection = document.getElementById("results-section");

// ── Toggle Logic ───────────────────────────────────────────────────────────

function switchModel(model) {
    activeModel = model;

    // Update button states
    btnTfidf.classList.toggle("is-active", model === "tfidf");
    btnBert.classList.toggle("is-active", model === "bert");

    // Update displayed route
    activeRouteUrl.textContent = ROUTES[model];
}

btnTfidf.addEventListener("click", () => switchModel("tfidf"));
btnBert.addEventListener("click", () => switchModel("bert"));

// ── Classification Request ─────────────────────────────────────────────────

classifyForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const content = textArea.value.trim();
    if (!content) return;

    // Loading state
    btnSubmit.disabled = true;
    btnSubmitText.innerHTML = '<span class="spinner"></span>Đang phân tích...';
    resultsSection.innerHTML = "";

    const endpoint = ROUTES[activeModel];

    try {
        const res = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ content }),
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => null);
            throw new Error(errData?.detail || `API responded with status ${res.status}`);
        }

        const data = await res.json();
        renderResults(data);
    } catch (err) {
        renderError(err.message);
    } finally {
        btnSubmit.disabled = false;
        btnSubmitText.textContent = "Phân Loại Ngay";
    }
});

// ── Render Results ─────────────────────────────────────────────────────────

function renderResults(data) {
    const { result, model_version } = data;
    const { label, confidence, probabilities, model_used } = result;
    const confidencePct = (confidence * 100).toFixed(1);

    // Sort probabilities descending
    const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

    const probBarsHTML = sorted
        .map(
            ([cat, prob]) => `
        <div class="prob-item">
            <div class="prob-item__labels">
                <span class="prob-item__name">${cat}</span>
                <span class="prob-item__percent">${(prob * 100).toFixed(1)}%</span>
            </div>
            <div class="prob-bar">
                <div class="prob-bar__fill" data-category="${cat}" data-width="${prob * 100}"></div>
            </div>
        </div>`
        )
        .join("");

    resultsSection.innerHTML = `
        <div class="results">
            <div class="results__header">
                <div class="results__category">
                    <span class="category-badge" data-category="${label}">${label}</span>
                </div>
                <div class="results__confidence">
                    Độ tin cậy: <span class="results__confidence-value">${confidencePct}%</span>
                </div>
            </div>

            <div class="results__meta">
                <span>Model: <span class="results__model-tag">${model_used.toUpperCase()}</span></span>
                <span>Version: ${model_version}</span>
            </div>

            <div class="prob-list">
                ${probBarsHTML}
            </div>
        </div>
    `;

    // Animate bars after DOM paint
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            document.querySelectorAll(".prob-bar__fill[data-width]").forEach((bar) => {
                bar.style.width = bar.dataset.width + "%";
            });
        });
    });
}

function renderError(message) {
    resultsSection.innerHTML = `
        <div class="error-toast">
            ⚠️ Lỗi: ${message}
        </div>
    `;
}
