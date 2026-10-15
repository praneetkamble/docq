(function () {
  'use strict';
  var $ = function (s) { return document.querySelector(s); };
  var esc = function (s) { var d = document.createElement('div'); d.textContent = s; return d.innerHTML; };

  function api(path, opts) {
    return fetch(path, opts).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (j) {
        if (!r.ok) throw new Error(j.detail || ('HTTP ' + r.status));
        return j;
      });
    });
  }

  function refresh() {
    api('/api/health').then(function (h) {
      $('#status').textContent = h.documents + ' document(s) loaded';
      $('#retr').textContent = h.retriever + (h.embedding_model ? ' · ' + h.embedding_model : '') + (h.vector_backend ? ' · ' + h.vector_backend : '');
      $('#gen').textContent = h.generator;
    }).catch(function () { $('#status').textContent = 'server not reachable'; });
    api('/api/documents').then(function (docs) {
      var ul = $('#docs'); ul.innerHTML = '';
      docs.forEach(function (d) {
        var li = document.createElement('li');
        li.innerHTML = '<span>' + esc(d.name) + '</span><small>' + d.pages + ' p · ' + d.chunks + ' chunks</small>';
        var x = document.createElement('button'); x.className = 'x'; x.type = 'button'; x.textContent = '×';
        x.setAttribute('aria-label', 'Remove ' + d.name);
        x.onclick = function () { api('/api/documents/' + encodeURIComponent(d.name), { method: 'DELETE' }).then(refresh); };
        li.appendChild(x); ul.appendChild(li);
      });
    });
  }

  function upload(files) {
    $('#docMsg').textContent = 'Indexing…';
    var chain = Promise.resolve();
    Array.prototype.forEach.call(files, function (f) {
      chain = chain.then(function () {
        var fd = new FormData(); fd.append('file', f);
        return api('/api/documents', { method: 'POST', body: fd });
      });
    });
    chain.then(function () { $('#docMsg').textContent = ''; refresh(); })
         .catch(function (e) { $('#docMsg').textContent = e.message; refresh(); });
  }

  $('#file').addEventListener('change', function (e) { upload(e.target.files); e.target.value = ''; });
  var drop = $('#drop');
  ['dragenter', 'dragover'].forEach(function (ev) { drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.add('over'); }); });
  ['dragleave', 'drop'].forEach(function (ev) { drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.remove('over'); }); });
  drop.addEventListener('drop', function (e) { upload(e.dataTransfer.files); });
  $('#samples').addEventListener('click', function () {
    $('#docMsg').textContent = 'Indexing samples…';
    api('/api/load-samples', { method: 'POST' }).then(function () { $('#docMsg').textContent = ''; refresh(); })
      .catch(function (e) { $('#docMsg').textContent = e.message; });
  });

  function ask(q) {
    var out = $('#out'); out.innerHTML = '<p class="hint">Searching…</p>';
    api('/api/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: q }) })
      .then(function (r) {
        var cited = {}; (r.citations || []).forEach(function (c) { cited[c.doc + '|' + c.page] = 1; });
        var h = '<div class="answer' + (r.abstained ? ' abstain' : '') + '">' + esc(r.answer) + '</div>';
        h += '<p class="meta">mode: ' + esc(r.mode) + ' · retriever: ' + esc(String(r.retriever)) + ' · confidence ' + (r.confidence != null ? r.confidence : '–') + ' · ' + r.latency_ms + ' ms</p>';
        if (r.llm_error) h += '<p class="msg">LLM call failed, showing extractive answer instead.</p>';
        if ((r.sources || []).length) h += '<h2>Retrieved passages</h2>';
        (r.sources || []).forEach(function (s) {
          var on = cited[s.doc + '|' + s.page];
          h += '<div class="src' + (on ? ' cited' : '') + '"><b>' + esc(s.doc) + ' · page ' + s.page + '</b>' + (on ? '<span class="tag">CITED</span>' : '') +
               ' <span class="meta">sim ' + s.score + '</span><p>' + esc(s.text) + '</p></div>';
        });
        out.innerHTML = h;
      }).catch(function (e) { out.innerHTML = '<p class="msg">' + esc(e.message) + '</p>'; });
  }
  $('#askForm').addEventListener('submit', function (e) { e.preventDefault(); ask($('#q').value.trim()); });
  document.addEventListener('click', function (e) {
    var a = e.target.closest && e.target.closest('[data-q]');
    if (a) { e.preventDefault(); $('#q').value = a.getAttribute('data-q'); ask(a.getAttribute('data-q')); }
  });
  refresh();
})();
