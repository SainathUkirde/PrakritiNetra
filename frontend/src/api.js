// Central API base URL — set VITE_API_URL in Vercel environment variables
// to point at your Render service, e.g. https://prakriti-netra-api.onrender.com
const BASE = import.meta.env.VITE_API_URL || ''

// Render free-tier cold starts can take up to 60–90 s.
// Use a generous timeout (90 s) so the browser doesn't give up first.
async function apiFetch(url, timeoutMs = 90_000) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch(url, { signal: controller.signal })
    clearTimeout(timer)
    if (!res.ok) {
      const text = await res.text().catch(() => res.statusText)
      throw new Error(`HTTP ${res.status}: ${text}`)
    }
    return res.json()
  } catch (err) {
    clearTimeout(timer)
    if (err.name === 'AbortError') {
      throw new Error(
        `Request timed out after ${timeoutMs / 1000}s. ` +
        'The Render backend may still be waking up — please wait a moment and refresh.'
      )
    }
    throw err
  }
}

export async function fetchConfig() {
  return apiFetch(`${BASE}/api/config`)
}

export async function fetchPipeline(method = 'kmeans', nRegimes = 3) {
  // Pipeline endpoint is the slowest — allow 120 s (Render cold-start + data processing)
  return apiFetch(
    `${BASE}/api/pipeline?method=${method}&n_regimes=${nRegimes}`,
    120_000,
  )
}

export async function fetchForecast({ variable, leadTime, timeIndex, method, nRegimes }) {
  const url =
    `${BASE}/api/forecast?variable=${variable}&lead_time=${leadTime}` +
    `&time_index=${timeIndex}&method=${method}&n_regimes=${nRegimes}`
  return apiFetch(url)
}

export async function fetchSkill({ variable, method, nRegimes }) {
  const url =
    `${BASE}/api/skill?variable=${variable}&method=${method}&n_regimes=${nRegimes}`
  return apiFetch(url)
}

export async function fetchExtreme({ leadTime, timeIndex, method, nRegimes }) {
  const url =
    `${BASE}/api/extreme?lead_time=${leadTime}&time_index=${timeIndex}` +
    `&method=${method}&n_regimes=${nRegimes}`
  return apiFetch(url)
}

export async function fetchWeightMap({ variable, leadTime, season, regime, method, nRegimes }) {
  const url =
    `${BASE}/api/weights?variable=${variable}&lead_time=${leadTime}` +
    `&season=${season}&regime=${regime}&method=${method}&n_regimes=${nRegimes}`
  return apiFetch(url)
}
