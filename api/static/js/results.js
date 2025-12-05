// api/static/js/results.js
document.addEventListener("DOMContentLoaded", () => {
  const container = document.getElementById("history-container");
  const clearBtn = document.getElementById("clear-history-btn");

  function loadHistory() {
    const raw = localStorage.getItem("sv_history");
    let history = [];
    try {
      if (raw) {
        history = JSON.parse(raw);
        if (!Array.isArray(history)) history = [];
      }
    } catch {
      history = [];
    }

    if (!history.length) {
      container.innerHTML =
        '<p class="sv-empty-state">Todavía no hay resultados guardados en este navegador.</p>';
      return;
    }

    // más reciente primero
    history.sort(
      (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    );

    const singles = history.filter((h) => h.type === "single");
    const batches = history.filter((h) => h.type === "batch");

    let html = "";

    if (singles.length) {
      html += `
        <section class="sv-history-section">
          <h2>Historial: Clasificación de Imagen Única</h2>
          <div class="table-wrapper">
            <table class="sv-table">
              <thead>
                <tr>
                  <th>Fecha</th>
                  <th>Archivo</th>
                  <th># detecciones</th>
                </tr>
              </thead>
              <tbody>
      `;
      singles.forEach((e) => {
        const dateStr = new Date(e.created_at).toLocaleString();
        const meta = e.meta || {};
        html += `
          <tr>
            <td>${dateStr}</td>
            <td>${meta.filename || "-"}</td>
            <td>${meta.num_detections ?? "-"}</td>
          </tr>
        `;
      });
      html += `
              </tbody>
            </table>
          </div>
        </section>
      `;
    }

    if (batches.length) {
      html += `
        <section class="sv-history-section">
          <h2>Historial: Detección por Lotes</h2>
          <div class="table-wrapper">
            <table class="sv-table">
              <thead>
                <tr>
                  <th>Fecha</th>
                  <th>ID lote</th>
                  <th>Ubicación</th>
                  <th># archivos</th>
                </tr>
              </thead>
              <tbody>
      `;
      batches.forEach((e) => {
        const dateStr = new Date(e.created_at).toLocaleString();
        const meta = e.meta || {};
        html += `
          <tr>
            <td>${dateStr}</td>
            <td>${meta.lote_id || "-"}</td>
            <td>${meta.location || "-"}</td>
            <td>${meta.num_files ?? "-"}</td>
          </tr>
        `;
      });
      html += `
              </tbody>
            </table>
          </div>
        </section>
      `;
    }

    container.innerHTML = html;
  }

  clearBtn.addEventListener("click", () => {
    const ok = confirm(
      "¿Seguro que deseas borrar todo el historial local de resultados?"
    );
    if (!ok) return;
    localStorage.removeItem("sv_history");
    loadHistory();
  });

  loadHistory();
});
