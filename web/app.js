// Constantes de utilidad
function americanToProb(odds) {
  if (odds > 0) {
    return 100 / (odds + 100);
  } else {
    return (-odds) / (-odds + 100);
  }
}

function expectedValue(prob, odds) {
  if (odds > 0) {
    return prob * (odds / 100) - (1 - prob);
  } else {
    return prob * (100 / -odds) - (1 - prob);
  }
}

function formatearFecha(fecha) {
  const d = new Date(fecha + 'T00:00:00');
  return d.toLocaleDateString('es-MX', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
}

// Cargar datos
fetch('data.json')
  .then(response => response.json())
  .then(data => {
    const fechaEl = document.getElementById('fecha');
    fechaEl.textContent = formatearFecha(data.fecha);

    renderGames(data.juegos);
    renderHistory(data.historial);
  })
  .catch(error => {
    console.error('Error al cargar data.json:', error);
    document.getElementById('tabla-juegos').innerHTML = '<p class="sin-juegos">No se pudieron cargar las predicciones.</p>';
  });

function renderGames(juegos) {
  const container = document.getElementById('tabla-juegos');
  if (!juegos || juegos.length === 0) {
    container.innerHTML = '<p class="sin-juegos">No hay juegos para hoy.</p>';
    return;
  }

  let html = '<table><thead><tr><th>Pitcher</th><th>Oponente</th><th>Línea</th><th>Call</th><th>Prob. modelo</th><th>Cuota actual</th><th>Evaluar</th><th>Resultado</th></tr></thead><tbody>';

  juegos.forEach((juego, idx) => {
    html += `<tr>
      <td>${juego.pitcher_name}</td>
      <td>${juego.opponent}</td>
      <td>${juego.line}</td>
      <td>${juego.call}</td>
      <td>${(juego.model_prob * 100).toFixed(1)}%</td>
      <td><input type="text" id="odds-${idx}" value="-110" placeholder="-110"></td>
      <td><button onclick="evaluar(${idx})">Evaluar</button></td>
      <td id="result-${idx}"></td>
    </tr>`;
  });

  html += '</tbody></table>';
  container.innerHTML = html;
}

function renderHistory(historial) {
  const container = document.getElementById('tabla-historial');
  if (!historial || historial.length === 0) {
    container.innerHTML = '<p class="sin-juegos">Aún no hay historial registrado.</p>';
    return;
  }

  // Muestra solo las últimas 20 filas para no saturar
  const ultimas = historial.slice(0, 20);
  let html = '<table><thead><tr><th>Fecha</th><th>Pitcher</th><th>Apuesta</th><th>Resultado</th></tr></thead><tbody>';
  ultimas.forEach(row => {
    html += `<tr>
      <td>${row.fecha || row.game_date || ''}</td>
      <td>${row.pitcher || row.pitcher_name || ''}</td>
      <td>${row.apuesta || row.PLAY || row.call || ''}</td>
      <td>${row.resultado || row.result || ''}</td>
    </tr>`;
  });
  html += '</tbody></table>';
  container.innerHTML = html;
}

function evaluar(idx) {
  const input = document.getElementById(`odds-${idx}`);
  const odds = parseInt(input.value, 10);
  if (isNaN(odds)) {
    alert('Ingresa una cuota válida, ej. -110 o +120');
    return;
  }

  // Obtener el juego correspondiente del arreglo global
  fetch('data.json')
    .then(response => response.json())
    .then(data => {
      const juego = data.juegos[idx];
      const prob = juego.model_prob; // Probabilidad del lado seleccionado
      const ev = expectedValue(prob, odds);
      const resultadoEl = document.getElementById(`result-${idx}`);

      if (ev > 0) {
        resultadoEl.innerHTML = `<span class="apostar">Apostar ${juego.call} · EV = +$${ev.toFixed(3)}</span>`;
      } else {
        resultadoEl.innerHTML = `<span class="pasar">Dejar pasar · EV = $${ev.toFixed(3)}</span>`;
      }
    });
}