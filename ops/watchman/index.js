'use strict';

// LNQ Watchman: activity-driven shelve/unshelve manager.
//
// Sits in front of Caddy's upstream so it sees every request to the core.
// When the core has been idle for longer than IDLE_TIMEOUT_SEC, the watchman
// shelves it via the OpenStack CLI. When traffic arrives while shelved, the
// watchman serves a placeholder page (with stats), triggers an unshelve, and
// flips back to proxying once /_up returns 200 on the core.
//
// Shelve/unshelve actions are performed by shelling out to the `openstack`
// CLI installed on the doorman. Auth env vars come from a systemd
// EnvironmentFile (see /etc/lnq/watchman.env, populated by Terraform).

const fastify = require('fastify')({
  logger: { level: process.env.LOG_LEVEL || 'info' },
  disableRequestLogging: true,
});
const http = require('http');
const fs = require('fs/promises');
const path = require('path');
const { execFile } = require('child_process');
const { promisify } = require('util');
const execFileAsync = promisify(execFile);

const CONFIG = {
  port: parseInt(process.env.WATCHMAN_PORT || '5990', 10),
  coreHost: process.env.CORE_HOST || '127.0.0.1',
  coreCouchPort: parseInt(process.env.CORE_COUCHDB_PORT || '5984', 10),
  coreDicomwebPort: parseInt(process.env.CORE_DICOMWEB_PORT || '5985', 10),
  idleTimeoutMs: parseInt(process.env.IDLE_TIMEOUT_SEC || '1200', 10) * 1000,
  pollIntervalMs: 2000,
  statsFile: process.env.STATS_FILE || '/var/lib/watchman/stats.json',
  coreInstanceName: process.env.CORE_INSTANCE_NAME || 'lnq-core',
  // OS_* env vars (used by the openstack CLI) come from systemd EnvironmentFile.
};

let state = 'awake';                  // awake | shelved | waking
let lastActivityAt = Date.now();
let lastShelveAt = null;              // ms epoch when we entered 'shelved'
let wakeStartedAt = null;             // ms epoch when current wake began
let inFlightTransition = null;        // promise; serializes shelve and wake

const stats = {
  shelveCount: 0,
  unshelveCount: 0,
  lastShelveAt: null,                 // ISO string
  lastUnshelveAt: null,               // ISO string
  lastShelveDurationSec: null,        // how long the most recent shelve lasted
  lastWakeDurationSec: null,          // how long the most recent wake took
  totalShelveDurationSec: 0,          // cumulative time shelved
  totalWakeDurationSec: 0,            // cumulative time spent waking
  watchmanStartedAt: new Date().toISOString(),
};

// ---------- stats persistence ----------

async function loadStats() {
  try {
    const raw = await fs.readFile(CONFIG.statsFile, 'utf-8');
    Object.assign(stats, JSON.parse(raw));
    fastify.log.info('Loaded prior stats from disk');
  } catch (e) {
    if (e.code !== 'ENOENT') fastify.log.warn(`Stats load failed: ${e.message}`);
  }
}

async function saveStats() {
  try {
    await fs.mkdir(path.dirname(CONFIG.statsFile), { recursive: true });
    const tmp = CONFIG.statsFile + '.tmp';
    await fs.writeFile(tmp, JSON.stringify(stats, null, 2));
    await fs.rename(tmp, CONFIG.statsFile);
  } catch (e) {
    fastify.log.error(`Stats save failed: ${e.message}`);
  }
}

// ---------- OpenStack actions (shell out to `openstack`) ----------

async function runOpenstack(args) {
  // Returns { stdout, stderr }; throws on non-zero exit.
  return execFileAsync('openstack', args, { timeout: 30_000 });
}

async function shelveCore() {
  fastify.log.info(`Shelving ${CONFIG.coreInstanceName}`);
  await runOpenstack(['server', 'shelve', CONFIG.coreInstanceName]);
}

async function unshelveCore() {
  fastify.log.info(`Unshelving ${CONFIG.coreInstanceName}`);
  await runOpenstack(['server', 'unshelve', CONFIG.coreInstanceName]);
}

async function isCoreUp() {
  try {
    const res = await fetch(`http://${CONFIG.coreHost}:${CONFIG.coreCouchPort}/_up`, {
      signal: AbortSignal.timeout(2000),
    });
    return res.ok;
  } catch (e) {
    return false;
  }
}

// ---------- state transitions ----------

async function transitionToShelved() {
  if (inFlightTransition) return inFlightTransition;
  inFlightTransition = (async () => {
    try {
      await shelveCore();
      lastShelveAt = Date.now();
      state = 'shelved';
      stats.shelveCount += 1;
      stats.lastShelveAt = new Date(lastShelveAt).toISOString();
      await saveStats();
      fastify.log.info(`State -> shelved (count=${stats.shelveCount})`);
    } catch (e) {
      fastify.log.error(`Shelve failed: ${e.message}; staying awake`);
    } finally {
      inFlightTransition = null;
    }
  })();
  return inFlightTransition;
}

async function transitionToWaking() {
  if (inFlightTransition) return inFlightTransition;
  inFlightTransition = (async () => {
    state = 'waking';
    wakeStartedAt = Date.now();
    fastify.log.info(`State -> waking; calling unshelve`);
    try {
      await unshelveCore();
    } catch (e) {
      fastify.log.error(`Unshelve API failed: ${e.message}; will retry on next request`);
      state = 'shelved';
      wakeStartedAt = null;
      return;
    }
    // Poll /_up until ready.
    while (true) {
      await new Promise((r) => setTimeout(r, CONFIG.pollIntervalMs));
      if (await isCoreUp()) break;
      // Bail after 10 minutes of polling (Js2 unshelves take ~60-120s typically).
      if (Date.now() - wakeStartedAt > 10 * 60 * 1000) {
        fastify.log.error('Wake timeout exceeded 10 min; giving up');
        state = 'shelved';
        wakeStartedAt = null;
        return;
      }
    }
    const wakeDurationSec = Math.round((Date.now() - wakeStartedAt) / 1000);
    const shelveDurationSec = lastShelveAt
      ? Math.round((Date.now() - lastShelveAt) / 1000)
      : 0;
    state = 'awake';
    lastActivityAt = Date.now();
    stats.unshelveCount += 1;
    stats.lastUnshelveAt = new Date().toISOString();
    stats.lastWakeDurationSec = wakeDurationSec;
    stats.lastShelveDurationSec = shelveDurationSec;
    stats.totalWakeDurationSec += wakeDurationSec;
    stats.totalShelveDurationSec += shelveDurationSec;
    await saveStats();
    fastify.log.info(`State -> awake (wake=${wakeDurationSec}s, shelved=${shelveDurationSec}s)`);
  })().finally(() => {
    inFlightTransition = null;
  });
  return inFlightTransition;
}

function checkIdle() {
  if (state !== 'awake') return;
  if (Date.now() - lastActivityAt > CONFIG.idleTimeoutMs) {
    transitionToShelved();
  }
}

// ---------- request classification ----------

function isInternalPath(url) {
  return (
    url.startsWith('/.well-known/') ||
    url === '/health/doorman' ||
    url.startsWith('/_watchman/')
  );
}

// ---------- placeholder rendering ----------

function formatDuration(sec) {
  if (sec == null) return '—';
  if (sec < 60) return `${sec}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ${sec % 60}s`;
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  return `${h}h ${m}m`;
}

function placeholderHTML() {
  const heading = state === 'waking' ? 'Waking up the LNQ server…' : 'Starting the LNQ server…';
  const elapsed = wakeStartedAt ? Math.round((Date.now() - wakeStartedAt) / 1000) : 0;
  const avgWakeSec = stats.unshelveCount > 0
    ? Math.round(stats.totalWakeDurationSec / stats.unshelveCount)
    : 90;
  const savedSu = (stats.totalShelveDurationSec / 3600) * 8; // m3.medium = 8 SU/hr
  const savedSuStr = savedSu < 1 ? savedSu.toFixed(1) : Math.round(savedSu).toString();

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="5">
<title>LNQ Chronicle — waking up</title>
<style>
  body { font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
         max-width: 560px; margin: 4em auto; padding: 0 1em; color: #222; }
  h1 { font-size: 1.4em; margin: 0 0 0.5em; font-weight: 500; }
  .spinner { display: inline-block; width: 0.9em; height: 0.9em;
             border: 2px solid #ccc; border-top-color: #444;
             border-radius: 50%; animation: spin 0.8s linear infinite;
             vertical-align: -0.1em; margin-right: 0.5em; }
  @keyframes spin { to { transform: rotate(360deg); } }
  p.lede { color: #555; }
  p.small { font-size: 0.9em; color: #888; margin-top: 0; }
  .stats { margin-top: 2.5em; padding-top: 1em; border-top: 1px solid #eee;
           font-size: 0.9em; color: #444; }
  .stats h2 { font-size: 0.95em; font-weight: 600; color: #666; margin: 0 0 0.5em;
              text-transform: uppercase; letter-spacing: 0.04em; }
  .stats table { border-collapse: collapse; }
  .stats td { padding: 0.18em 1.5em 0.18em 0; vertical-align: top; }
  .stats td:first-child { color: #888; }
  .stats td:nth-child(2) { font-variant-numeric: tabular-nums; }
  .footer { margin-top: 2em; font-size: 0.8em; color: #aaa; }
</style>
</head>
<body>
<h1><span class="spinner"></span>${heading}</h1>
<p class="lede">This page will refresh automatically every 5 seconds.</p>
<p class="small">Elapsed: ${elapsed}s &middot; typical wake-up takes about ${avgWakeSec}s.</p>

<div class="stats">
  <h2>SU conservation</h2>
  <table>
    <tr><td>Shelves</td><td>${stats.shelveCount}</td></tr>
    <tr><td>Unshelves</td><td>${stats.unshelveCount}</td></tr>
    <tr><td>Last shelve duration</td><td>${formatDuration(stats.lastShelveDurationSec)}</td></tr>
    <tr><td>Last wake-up time</td><td>${formatDuration(stats.lastWakeDurationSec)}</td></tr>
    <tr><td>Average wake-up time</td><td>${formatDuration(avgWakeSec)}</td></tr>
    <tr><td>Total time shelved</td><td>${formatDuration(stats.totalShelveDurationSec)}</td></tr>
    <tr><td>Estimated SUs saved</td><td>~${savedSuStr}</td></tr>
  </table>
</div>

<p class="footer">SlicerLNQ-Chronicle &middot; watchman</p>
</body>
</html>`;
}

// ---------- proxying ----------

// Don't parse bodies — we stream them straight to the upstream.
fastify.addContentTypeParser('*', (req, payload, done) => done(null));

function proxyToCore(request, reply) {
  const url = request.raw.url;
  const isDicomweb = url.startsWith('/dicomweb/') || url === '/dicomweb';
  const targetPort = isDicomweb ? CONFIG.coreDicomwebPort : CONFIG.coreCouchPort;
  const targetPath = isDicomweb ? (url.slice('/dicomweb'.length) || '/') : url;

  return new Promise((resolve) => {
    const proxyReq = http.request(
      {
        host: CONFIG.coreHost,
        port: targetPort,
        method: request.method,
        path: targetPath,
        headers: request.headers,
      },
      (proxyRes) => {
        reply.code(proxyRes.statusCode);
        for (const [k, v] of Object.entries(proxyRes.headers)) reply.header(k, v);
        reply.send(proxyRes); // pipes the response stream
        proxyRes.on('end', resolve);
        proxyRes.on('close', resolve);
      },
    );
    proxyReq.on('error', (err) => {
      fastify.log.error(`Proxy error to ${targetPort}: ${err.message}`);
      if (!reply.sent) reply.code(502).send({ error: 'core unavailable', message: err.message });
      resolve();
    });
    request.raw.pipe(proxyReq);
  });
}

// ---------- routes ----------

fastify.get('/_watchman/health', async () => ({ ok: true, state }));

fastify.get('/_watchman/stats', async () => ({
  state,
  config: {
    idleTimeoutSec: CONFIG.idleTimeoutMs / 1000,
    coreInstance: CONFIG.coreInstanceName,
  },
  stats,
  secondsSinceLastActivity: Math.round((Date.now() - lastActivityAt) / 1000),
  inFlightTransition: !!inFlightTransition,
}));

fastify.all('/*', async (request, reply) => {
  const url = request.raw.url;
  const internal = isInternalPath(url);
  if (!internal) lastActivityAt = Date.now();

  if (state === 'awake') return proxyToCore(request, reply);

  if (state === 'shelved' && !internal) {
    // Fire-and-forget; the wake transition runs in the background.
    transitionToWaking();
  }

  reply.code(200).type('text/html; charset=utf-8').send(placeholderHTML());
});

// ---------- startup ----------

async function detectInitialState() {
  // If the core responds, we're awake. Otherwise assume shelved; the next
  // request will trigger a wake. This handles watchman restarts cleanly.
  if (await isCoreUp()) {
    state = 'awake';
    fastify.log.info('Initial state: awake (core /_up responded)');
  } else {
    state = 'shelved';
    fastify.log.info('Initial state: shelved (core /_up did not respond)');
  }
}

async function main() {
  await loadStats();
  await detectInitialState();

  setInterval(checkIdle, 30 * 1000);

  await fastify.listen({ host: '127.0.0.1', port: CONFIG.port });
  fastify.log.info(
    `Watchman listening on 127.0.0.1:${CONFIG.port} ` +
      `(idle=${CONFIG.idleTimeoutMs / 1000}s, core=${CONFIG.coreHost}:${CONFIG.coreCouchPort}/${CONFIG.coreDicomwebPort})`,
  );
}

process.on('SIGTERM', async () => {
  fastify.log.info('SIGTERM received, saving stats and exiting');
  await saveStats();
  process.exit(0);
});

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
