function showSection(id, event) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  if (event && event.currentTarget) {
    event.currentTarget.classList.add('active');
  }
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
  if (odds > 0) {
    return prob * (odds / 100) - (1 - prob);
  } else {
    return prob * (100 / -odds) - (1 - prob);
  }
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

function elegirMejorLado(juego) {
  // Busca la línea y lado con mayor probabilidad ≥ 0.55, o la más alta disponible
  const lineas = [3.5, 4.5, 5.5, 6.5];
  let mejor = null;
  for (const l of lineas) {
    const pOver = juego[`prob_over_${l}`] || 0;
    const pUnder = 1 - pOver;
    const candidatos = [
      { lado: 'Over', linea: l, prob: pOver },
      { lado: 'Under', linea: l, prob: pUnder }
    ];
    for (const c of candidatos) {
      if (c.prob >= 0.55) {
        if (!mejor || c.prob > mejor.prob) {
          mejor = c;
        }
      }
    }
  }
  // Si ningún lado llega a 55%, tomar el más alto
  if (!mejor) {
    for (const l of lineas) {
      const pOver = juego[`prob_over_${l}`] || 0;
      const pUnder = 1 - pOver;
      const candidatos = [
        { lado: 'Over', linea: l, prob: pOver },
        { lado: 'Under', linea: l, prob: pUnder }
      ];
      for (const c of candidatos) {
        if (!mejor || c.prob > mejor.prob) {
          mejor = c;
        }
      }
    }
  }
  return mejor;
}

function renderGames(juegos) {
  const container = document.getElementById('tabla-juegos');
  if (!juegos.length) {
    container.innerHTML = '<p class="sin-juegos">No hay juegos disponibles.</p>';
    return;
  }

  let html = '<table><thead><tr><th>Pitcher</th><th>Equipo</th><th>Lado sugerido</th><th>Probabilidad</th><th>Línea casa</th><th>Cuota</th><th>Evaluar</th><th>Resultado</th></tr></thead><tbody>';

  juegos.forEach((juego, idx) => {
    const mejor = elegirMejorLado(juego);
    const ladoSugerido = mejor ? `${mejor.lado} ${mejor.linea}` : '';
    const probSugerida = mejor ? (mejor.prob * 100).toFixed(1) : '';
    const lineaDefault = juego.line || (mejor ? mejor.linea : 6.5);

    html += `<tr>
      <td>${juego.pitcher_name}</td>
      <td>${juego.team || ''}</td>
      <td>${ladoSugerido}</td>
      <td>${probSugerida}%</td>
      <td><input type="number" step="0.5" id="line-${idx}" value="${lineaDefault}"></td>
      <td><input type="text" id="odds-${idx}" value="-110"></td>
      <td><button onclick="evaluar(${idx})">Evaluar</button></td>
      <td id="result-${idx}"></td>
    </tr>`;
  });
  html += '</tbody></table>';
  container.innerHTML = html;

  datos.juegosFiltrados = juegos;
}

function evaluar(idx) {
  const juego = datos.juegosFiltrados ? datos.juegosFiltrados[idx] : datos.juegos[idx];
  if (!juego) return;
  const linea = parseFloat(document.getElementById(`line-${idx}`).value);
  const odds = parseInt(document.getElementById(`odds-${idx}`).value, 10);
  if (isNaN(linea) || isNaN(odds)) {
    alert('Ingresa línea y cuota válidas');
    return;
  }

  const mejor = elegirMejorLado(juego);
  if (!mejor) {
    document.getElementById(`result-${idx}`).innerHTML = '<span class="sin-valor">Sin probabilidades</span>';
    return;
  }

  const lado = mejor.lado;
  const prob = mejor.prob;

  // Calcular EV para el lado sugerido con la cuota ingresada
  const ev = expectedValue(prob, odds);
  const cuotaRec = cuotaMinima(prob);

  let resultadoHTML = '';
  if (ev > 0) {
    resultadoHTML = `<span class="valor">Valor en ${lado} ${linea} · Prob ${(prob * 100).toFixed(1)}% · EV +${ev.toFixed(3)} · Cuota mínima: ${cuotaRec}</span>`;
  } else {
    resultadoHTML = `<span class="sin-valor">Sin valor en ${lado} ${linea} · Prob ${(prob * 100).toFixed(1)}% · EV ${ev.toFixed(3)} · Cuota mínima: ${cuotaRec}</span>`;
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