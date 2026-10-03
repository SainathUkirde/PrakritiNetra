// Central API base URL — set VITE_API_URL in Vercel environment variables
// to point at your Render service, e.g. https://prakriti-netra-api.onrender.com
const BASE = import.meta.env.VITE_API_URL || ''

export async function fetchConfig() {
  const res = await fetch(`${BASE}/api/config`)
  if (!res.ok) throw new Error('Failed to fetch config')
  return res.json()
}

export async function fetchPipeline(method = 'kmeans', nRegimes = 3) {
  const res = await fetch(`${BASE}/api/pipeline?method=${method}&n_regimes=${nRegimes}`)
  if (!res.ok) throw new Error('Failed to fetch pipeline')
  return res.json()
}

export async function fetchForecast({ variable, leadTime, timeIndex, method, nRegimes }) {
  const url = `${BASE}/api/forecast?variable=${variable}&lead_time=${leadTime}&time_index=${timeIndex}&method=${method}&n_regimes=${nRegimes}`
  const res = await fetch(url)
  if (!res.ok) throw new Error('Failed to fetch forecast')
  return res.json()
}

export async function fetchSkill({ variable, method, nRegimes }) {
  const url = `${BASE}/api/skill?variable=${variable}&method=${method}&n_regimes=${nRegimes}`
  const res = await fetch(url)
  if (!res.ok) throw new Error('Failed to fetch skill')
  return res.json()
}

export async function fetchExtreme({ leadTime, timeIndex, method, nRegimes }) {
  const url = `${BASE}/api/extreme?lead_time=${leadTime}&time_index=${timeIndex}&method=${method}&n_regimes=${nRegimes}`
  const res = await fetch(url)
  if (!res.ok) throw new Error('Failed to fetch extremes')
  return res.json()
}
