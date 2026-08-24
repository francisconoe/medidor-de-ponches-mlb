function showSection(id) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  event.currentTarget.classList.add('active');
}

function formatearFecha(fecha) {
  const d = new Date(fecha + 'T00:00:00');
  return d.toLocaleDateString('es-MX', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
}

function americanToProb(odds) {
  if (odds > 0) return 100 / (odds + 100);
  else return -odds / (-odds + 100);
}

function expectedValue(prob, odds) {
  if (odds > 0) return prob * (odds / 100) - (1 - prob);
  else return prob * (100 / -odds) - (1 - prob);
}

function breakEvenProb(odds) {
  return americanToProb(odds);
}

function cuotaMinima(prob) {
  if (prob >= 0.5) {
    return -Math.round(prob / (1 - prob) * 100);
  } else {
    return Math.round((1 - prob) / prob * 100);
  }
}

let datos = {};

fetch('data.json')
  .then(response => response.json())
  .then(data => {
    datos = data;
    document.getElementById('fecha').textContent = formatearFecha(data.fecha);
    renderGames(data.juegos);
    renderHistory(data.historial);
  })
  .catch(error => console.error('Error:', error));

function renderGames(juegos) {
  const filtrados = juegos.filter(j => {
    return Math.max(
      j.prob_over_3_5, j.prob_over_4_5, j.prob_over_5_5, j.prob_over_6_5,
      1 - j.prob_over_3_5, 1 - j.prob_over_4_5, 1 - j.prob_over_5_5, 1 - j.prob_over_6_5
    ) >= 0.55;
  });
  const container = document.getElementById('tabla-juegos');
  if (!filtrados.length) {
    container.innerHTML = '<p class="sin-juegos">No hay juegos con probabilidad ≥ 55%.</p>';
    return;
  }

  let html = '<table><thead><tr><th>Pitcher</th><th>Equipo</th><th>Línea casa</th><th>Cuota</th><th>Evaluar</th><th>Resultado</th></tr></thead><tbody>';
  filtrados.forEach((juego, idx) => {
    html += `<tr>
      <td>${juego.pitcher_name}</td>
      <td>${juego.team || ''}</td>
      <td><input type="number" step="0.5" id="line-${idx}" value="${juego.line || 6.5}"></td>
      <td><input type="text" id="odds-${idx}" value="-110"></td>
      <td><button onclick="evaluar(${idx})">Evaluar</button></td>
      <td id="result-${idx}"></td>
    </tr>`;
  });
  html += '</tbody></table>';
  container.innerHTML = html;
}

function evaluar(idx) {
  const juego = datos.juegos[idx];
  const linea = parseFloat(document.getElementById(`line-${idx}`).value);
  const odds = parseInt(document.getElementById(`odds-${idx}`).value, 10);
  if (isNaN(linea) || isNaN(odds)) {
    alert('Ingresa línea y cuota válidas');
    return;
  }

  const lineasConocidas = [3.5, 4.5, 5.5, 6.5];
  const probsOver = lineasConocidas.map(l => juego[`prob_over_${l}`] || 0.5);

  let pOver;
  if (linea <= 3.5) {
    pOver = probsOver[0];
  } else if (linea >= 6.5) {
    pOver = probsOver[3];
  } else {
    const i = Math.min(2, Math.floor((linea - 3.5) / 1));
    const x0 = lineasConocidas[i], x1 = lineasConocidas[i+1];
    const y0 = probsOver[i], y1 = probsOver[i+1];
    pOver = y0 + (linea - x0) * (y1 - y0) / (x1 - x0);
  }
  pOver = Math.max(0.05, Math.min(0.95, pOver));
  const pUnder = 1 - pOver;
  const evOver = expectedValue(pOver, odds);
  const evUnder = expectedValue(pUnder, -odds);

  let resultadoHTML = '';
  const mejorEV = Math.max(evOver, evUnder);
  if (mejorEV > 0) {
    const lado = evOver > evUnder ? 'Over' : 'Under';
    const prob = evOver > evUnder ? pOver : pUnder;
    const cuotaRec = cuotaMinima(prob);
    resultadoHTML = `<span class="valor">Valor en ${lado} ${linea} · Prob ${(prob*100).toFixed(1)}% · EV +${mejorEV.toFixed(3)} · Cuota mínima: ${cuotaRec}</span>`;
  } else {
    resultadoHTML = `<span class="sin-valor">Sin valor · EV máximo ${mejorEV.toFixed(3)}</span>`;
  }
  document.getElementById(`result-${idx}`).innerHTML = resultadoHTML;
}

function renderHistory(historial) {
  const container = document.getElementById('tabla-historial');
  const filtrado = historial.filter(h => {
    const fecha = h.fecha || h.game_date || '';
    return fecha >= '2026-08-24';
  });
  if (!filtrado.length) {
    container.innerHTML = '<p class="sin-juegos">No hay historial desde 24/08/2026.</p>';
    return;
  }
  let html = '<table><thead><tr><th>Fecha</th><th>Pitcher</th><th>Apuesta</th><th>Resultado</th></tr></thead><tbody>';
  filtrado.slice(0, 50).forEach(row => {
    html += `<tr>
      <td>${row.fecha || row.game_date || ''}</td>
      <td>${row.pitcher || row.pitcher_name || ''}</td>
      <td>${row.apuesta || row.call || ''}</td>
      <td>${row.resultado || row.result || ''}</td>
    </tr>`;
  });
  html += '</tbody></table>';
  container.innerHTML = html;
}