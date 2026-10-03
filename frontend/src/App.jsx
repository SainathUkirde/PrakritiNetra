import React, { useState, useEffect, useCallback } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
  LineChart, Line, CartesianGrid,
} from 'recharts'
import { fetchConfig, fetchPipeline, fetchForecast, fetchSkill, fetchExtreme, fetchWeightMap } from './api'

// ── colour tokens ────────────────────────────────────────────────────────────
const BG       = '#0f1117'
const SURFACE  = '#1a1f2e'
const BORDER   = 'rgba(255,255,255,0.10)'
const TEXT     = '#e2e8f0'
const MUTED    = '#94a3b8'
const LABEL    = '#64748b'
const HL       = '#60a5fa'

const SOURCE_COLORS = {
  nwp:      '#3b82f6',
  ensemble: '#10b981',
  ai:       '#f59e0b',
  blended:  '#6366f1',
  naive:    '#6b7280',
}

const VAR_LABELS = { rainfall: 'Rainfall (mm)', temperature: 'Temperature (°C)', wind: 'Wind Speed (m/s)' }
const LEAD_LABELS = { 6: '6 h (nowcast)', 24: '24 h (day-1)', 72: '72 h (day-3)', 168: '168 h (day-7)' }
const EW_META = {
  rainfall:    { label: 'Heavy Rainfall', icon: '🌧️', color: '#3b82f6', unit: 'mm' },
  temperature: { label: 'Heatwave',       icon: '🌡️', color: '#ef4444', unit: '°C' },
  wind:        { label: 'Gale-force Wind',icon: '💨', color: '#10b981', unit: 'm/s' },
}

// ── reusable sub-components ───────────────────────────────────────────────────
function Card({ children, style }) {
  return (
    <div style={{
      background: 'rgba(255,255,255,0.04)',
      border: `1px solid ${BORDER}`,
      borderRadius: '0.75rem',
      padding: '1.25rem 1.5rem',
      marginBottom: '0.75rem',
      ...style,
    }}>
      {children}
    </div>
  )
}

function SectionHead({ children }) {
  return (
    <div style={{ fontSize: '0.72rem', fontWeight: 700, color: LABEL,
                  textTransform: 'uppercase', letterSpacing: '0.08em',
                  margin: '1.25rem 0 0.5rem' }}>
      {children}
    </div>
  )
}

function Badge({ label, regime }) {
  const colorMap = {
    convective: { bg: 'rgba(251,191,36,0.2)',  color: '#d97706' },
    stratiform: { bg: 'rgba(96,165,250,0.2)',  color: '#2563eb' },
    clear:      { bg: 'rgba(52,211,153,0.2)',  color: '#059669' },
    monsoon:    { bg: 'rgba(167,139,250,0.2)', color: '#7c3aed' },
    winter:     { bg: 'rgba(56,189,248,0.2)',  color: '#0ea5e9' },
  }
  const style = colorMap[regime?.toLowerCase()] || { bg: SURFACE, color: TEXT }
  return (
    <span style={{ display: 'inline-block', padding: '0.2rem 0.65rem',
                   borderRadius: '9999px', fontSize: '0.78rem', fontWeight: 600,
                   background: style.bg, color: style.color }}>
      {label || regime}
    </span>
  )
}

// Simple grid heatmap using SVG
// Parse a 6-digit hex colour string (#rrggbb) into [r, g, b] components.
// Falls back to a safe neutral if the string is not a valid hex colour.
function _hexToRgb(hex) {
  const m = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex)
  return m ? [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)] : [96, 165, 250]
}

function HeatmapSVG({ grid, lats, lons, title, colorHigh = '#3b82f6', unit = '' }) {
  if (!grid || !lats || !lons) return <div style={{ color: MUTED, fontSize: '0.8rem' }}>Loading…</div>
  const rows = grid.length
  const cols = grid[0]?.length || 0
  if (!rows || !cols) return null

  const flat = grid.flat().filter(v => v !== null && isFinite(v))

  // If all values are null / NaN — show a "no data" placeholder instead of crashing
  if (flat.length === 0) {
    return (
      <div>
        <div style={{ fontSize: '0.78rem', fontWeight: 600, color: TEXT, marginBottom: 4 }}>{title}</div>
        <div style={{ width: '100%', height: 160, borderRadius: 6, background: '#1a1f2e',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontSize: '0.75rem', color: MUTED }}>
          No data for this selection
        </div>
      </div>
    )
  }

  const vmin  = Math.min(...flat)
  const vmax  = Math.max(...flat)
  const range = vmax - vmin || 1
  const [hr, hg, hb] = _hexToRgb(colorHigh)

  // Dark base colour: #0f1117 = rgb(15,17,23)
  const lerp = v => {
    const t = (v - vmin) / range
    const r = Math.round(15 + t * (hr - 15))
    const g = Math.round(17 + t * (hg - 17))
    const b = Math.round(23 + t * (hb - 23))
    return `rgb(${r},${g},${b})`
  }

  // Use viewBox so the SVG fills its container responsively
  const VW = cols * 10
  const VH = rows * 10
  const cellW = VW / cols
  const cellH = VH / rows

  return (
    <div>
      <div style={{ fontSize: '0.78rem', fontWeight: 600, color: TEXT, marginBottom: 4 }}>{title}</div>
      <svg viewBox={`0 0 ${VW} ${VH}`} width="100%" height={160}
           preserveAspectRatio="none"
           style={{ borderRadius: 6, overflow: 'hidden', display: 'block' }}>
        {grid.map((row, ri) =>
          row.map((v, ci) => (
            <rect key={`${ri}-${ci}`}
              x={ci * cellW} y={ri * cellH}
              width={cellW} height={cellH}
              fill={v === null || !isFinite(v) ? '#1a1f2e' : lerp(v)}
            />
          ))
        )}
      </svg>
      <div style={{ fontSize: '0.68rem', color: MUTED, marginTop: 2 }}>
        min {vmin.toFixed(1)} · max {vmax.toFixed(1)} {unit}
      </div>
    </div>
  )
}

// ── main App ──────────────────────────────────────────────────────────────────
export default function App() {
  // Config
  const [config, setConfig]       = useState(null)
  const [pipeline, setPipeline]   = useState(null)
  const [forecast, setForecast]   = useState(null)
  const [skill, setSkill]         = useState(null)
  const [extreme, setExtreme]     = useState(null)
  const [weightMaps, setWeightMaps] = useState(null)
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState(null)

  // Controls
  const [variable, setVariable]   = useState('rainfall')
  const [leadTime, setLeadTime]   = useState(24)
  const [timeIdx, setTimeIdx]     = useState(0)
  const [method, setMethod]       = useState('kmeans')
  const [nRegimes, setNRegimes]   = useState(3)

  // Initial config load
  useEffect(() => {
    fetchConfig().then(setConfig).catch(e => setError(e.message))
  }, [])

  // Re-fetch pipeline when method/nRegimes change
  // Render free tier cold-starts can take up to 60s — retry up to 4 times
  useEffect(() => {
    let cancelled = false
    setLoading(true)
    const attempt = (retries) => {
      fetchPipeline(method, nRegimes)
        .then(p => {
          if (!cancelled) { setPipeline(p); setTimeIdx(Math.floor(p.n_times / 2)) }
        })
        .catch(e => {
          if (!cancelled && retries > 0) {
            setTimeout(() => attempt(retries - 1), 15000)
          } else if (!cancelled) {
            setError(e.message)
          }
        })
        .finally(() => { if (!cancelled) setLoading(false) })
    }
    attempt(4)
    return () => { cancelled = true }
  }, [method, nRegimes])

  // Re-fetch forecast + skill + extreme + weight maps when any selector changes
  const refresh = useCallback(() => {
    if (!pipeline) return
    setLoading(true)
    const regimes  = pipeline?.regimes  || []
    const seasons  = pipeline?.seasons  || []
    const curRegime = regimes[timeIdx]  || 'clear'
    const curSeason = seasons[timeIdx]  || 'monsoon'
    const params = { variable, leadTime, timeIndex: timeIdx, method, nRegimes }
    Promise.all([
      fetchForecast(params),
      fetchSkill({ variable, method, nRegimes }),
      fetchExtreme({ leadTime, timeIndex: timeIdx, method, nRegimes }),
      fetchWeightMap({ variable, leadTime, season: curSeason, regime: curRegime, method, nRegimes }),
    ])
      .then(([f, s, e, w]) => { setForecast(f); setSkill(s); setExtreme(e); setWeightMaps(w) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [pipeline, variable, leadTime, timeIdx, method, nRegimes])

  useEffect(() => { refresh() }, [refresh])

  if (error) {
    return (
      <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center',
                    justifyContent: 'center', background: BG, padding: '2rem' }}>
        <Card style={{ maxWidth: 480 }}>
          <div style={{ color: '#ef4444', fontWeight: 700, marginBottom: 8 }}>⚠️ API Error</div>
          <div style={{ color: MUTED, fontSize: '0.88rem' }}>{error}</div>
          <div style={{ fontSize: '0.78rem', color: LABEL, marginTop: 12 }}>
            Make sure the Render backend is running and <code>VITE_API_URL</code> is set
            to your Render service URL in Vercel environment variables.
          </div>
        </Card>
      </div>
    )
  }

  const times = pipeline?.time_steps || []
  const regimes = pipeline?.regimes || []
  const seasons = pipeline?.seasons || []
  const curRegime = regimes[timeIdx] || '—'
  const curSeason = seasons[timeIdx] || '—'

  return (
    <div style={{ minHeight: '100vh', background: BG, color: TEXT,
                  fontFamily: "-apple-system,'Segoe UI',system-ui,sans-serif" }}>

      {/* ── Header ── */}
      <div style={{ borderBottom: '2px solid rgba(255,255,255,0.10)',
                    padding: '1rem 2rem 0.75rem', maxWidth: 1280, margin: '0 auto' }}>
        <h1 style={{ margin: 0, fontSize: '2rem', fontWeight: 900, letterSpacing: '-0.03em' }}>
          <span style={{ color: '#22c55e' }}>Prakriti</span>
          <span style={{ color: '#3b82f6' }}>Netra</span>
        </h1>
        <p style={{ margin: '4px 0 0', fontSize: '0.88rem', color: MUTED }}>
          Smarter Forecasts&nbsp;·&nbsp;Healthier Tomorrow&nbsp;·&nbsp;
          <span style={{ color: HL }}>Rainfall, Temperature &amp; Wind</span>
          &nbsp;·&nbsp;India domain&nbsp;·&nbsp;Hybrid AI‑NWP Adaptive Blending
        </p>
        {loading && (
          <div style={{ marginTop: 4, fontSize: '0.75rem', color: '#f59e0b' }}>
            ⏳ Loading pipeline… (first load may take ~60s while Render wakes up)
          </div>
        )}
      </div>

      <div style={{ maxWidth: 1280, margin: '0 auto', display: 'flex', gap: '1.5rem', padding: '1rem 2rem' }}>

        {/* ── Sidebar ── */}
        <aside style={{ width: 220, flexShrink: 0 }}>
          <Card>
            <div style={{ fontSize: '0.72rem', fontWeight: 700, color: LABEL,
                          textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>
              Controls
            </div>
            <label style={{ display: 'block', fontSize: '0.8rem', color: MUTED, marginBottom: 2 }}>Variable</label>
            <select value={variable} onChange={e => setVariable(e.target.value)}
              style={{ width: '100%', background: SURFACE, color: TEXT, border: `1px solid ${BORDER}`,
                       borderRadius: 6, padding: '0.35rem 0.5rem', fontSize: '0.85rem', marginBottom: 10 }}>
              {(config?.variables || ['rainfall','temperature','wind']).map(v =>
                <option key={v} value={v}>{VAR_LABELS[v]}</option>
              )}
            </select>

            <label style={{ display: 'block', fontSize: '0.8rem', color: MUTED, marginBottom: 2 }}>Lead time</label>
            <select value={leadTime} onChange={e => setLeadTime(Number(e.target.value))}
              style={{ width: '100%', background: SURFACE, color: TEXT, border: `1px solid ${BORDER}`,
                       borderRadius: 6, padding: '0.35rem 0.5rem', fontSize: '0.85rem', marginBottom: 10 }}>
              {(config?.lead_times || [6,24,72,168]).map(l =>
                <option key={l} value={l}>{LEAD_LABELS[l]}</option>
              )}
            </select>

            <label style={{ display: 'block', fontSize: '0.8rem', color: MUTED, marginBottom: 2 }}>
              Time step
            </label>
            <input type="range" min={0} max={Math.max(0, times.length - 1)} value={timeIdx}
              onChange={e => setTimeIdx(Number(e.target.value))}
              style={{ width: '100%', accentColor: HL, marginBottom: 4 }} />
            <div style={{ fontSize: '0.72rem', color: MUTED }}>
              {times[timeIdx] || '—'}
            </div>

            <div style={{ borderTop: `1px solid ${BORDER}`, margin: '10px 0' }} />
            <div style={{ fontSize: '0.72rem', fontWeight: 700, color: LABEL,
                          textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6 }}>
              Model settings
            </div>
            <label style={{ display: 'block', fontSize: '0.8rem', color: MUTED, marginBottom: 2 }}>Method</label>
            <select value={method} onChange={e => setMethod(e.target.value)}
              style={{ width: '100%', background: SURFACE, color: TEXT, border: `1px solid ${BORDER}`,
                       borderRadius: 6, padding: '0.35rem 0.5rem', fontSize: '0.85rem', marginBottom: 10 }}>
              <option value="kmeans">k-Means</option>
              <option value="gmm">GMM</option>
            </select>

            <label style={{ display: 'block', fontSize: '0.8rem', color: MUTED, marginBottom: 2 }}>
              Regimes: {nRegimes}
            </label>
            <input type="range" min={2} max={5} value={nRegimes}
              onChange={e => setNRegimes(Number(e.target.value))}
              style={{ width: '100%', accentColor: HL }} />
          </Card>
          {config && (
            <Card style={{ fontSize: '0.72rem', color: MUTED }}>
              {config.real_data
                ? <span>📡 Real GFS data loaded</span>
                : <span style={{ color: '#d97706' }}>⚠️ Synthetic demo data</span>
              }
            </Card>
          )}
        </aside>

        {/* ── Main content ── */}
        <main style={{ flex: 1, minWidth: 0 }}>

          {/* Context badges */}
          {forecast && (
            <Card style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap', alignItems: 'flex-start' }}>
              <div>
                <div style={{ fontSize: '0.68rem', color: LABEL, textTransform: 'uppercase',
                              letterSpacing: '0.07em', marginBottom: 2 }}>Selected date</div>
                <div style={{ fontWeight: 600 }}>{forecast.time}</div>
              </div>
              <div>
                <div style={{ fontSize: '0.68rem', color: LABEL, textTransform: 'uppercase',
                              letterSpacing: '0.07em', marginBottom: 2 }}>Regime</div>
                <Badge regime={curRegime} />
              </div>
              <div>
                <div style={{ fontSize: '0.68rem', color: LABEL, textTransform: 'uppercase',
                              letterSpacing: '0.07em', marginBottom: 2 }}>Season</div>
                <Badge regime={curSeason} />
              </div>
              <div>
                <div style={{ fontSize: '0.68rem', color: LABEL, textTransform: 'uppercase',
                              letterSpacing: '0.07em', marginBottom: 2 }}>Variable / Lead</div>
                <div style={{ fontWeight: 600 }}>{VAR_LABELS[variable]} · {LEAD_LABELS[leadTime]}</div>
              </div>
            </Card>
          )}

          {/* Forecast maps */}
          <SectionHead>Forecast Maps</SectionHead>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '0.75rem' }}>
            {forecast ? (
              <>
                <Card>
                  <HeatmapSVG grid={forecast.truth}   lats={forecast.lats} lons={forecast.lons}
                    title="📍 Observed (Truth)"
                    colorHigh={variable === 'rainfall' ? '#3b82f6' : variable === 'temperature' ? '#ef4444' : '#10b981'}
                    unit={variable === 'rainfall' ? 'mm' : variable === 'temperature' ? '°C' : 'm/s'} />
                </Card>
                <Card>
                  <HeatmapSVG grid={forecast.blended} lats={forecast.lats} lons={forecast.lons}
                    title="🔀 Blended Forecast"
                    colorHigh={variable === 'rainfall' ? '#3b82f6' : variable === 'temperature' ? '#ef4444' : '#10b981'}
                    unit={variable === 'rainfall' ? 'mm' : variable === 'temperature' ? '°C' : 'm/s'} />
                </Card>
                <Card>
                  <HeatmapSVG grid={forecast.naive}   lats={forecast.lats} lons={forecast.lons}
                    title="⚖️ Naive Blend"
                    colorHigh="#6b7280"
                    unit={variable === 'rainfall' ? 'mm' : variable === 'temperature' ? '°C' : 'm/s'} />
                </Card>
              </>
            ) : (
              [0,1,2].map(i => <Card key={i} style={{ height: 200, display: 'flex',
                alignItems: 'center', justifyContent: 'center', color: MUTED }}>
                Loading…
              </Card>)
            )}
          </div>

          {/* Per-source maps */}
          <SectionHead>Individual Model Forecasts</SectionHead>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '0.75rem' }}>
            {(['nwp','ensemble','ai']).map(src => {
              const icons = { nwp: '🏗️', ensemble: '🧩', ai: '🤖' }
              const srcDesc = {
                nwp:      'Physical NWP — accurate short-term, degrades fast.',
                ensemble: 'Ensemble — smooth, underestimates extremes.',
                ai:       'AI/ML model — stable at long lead times.',
              }
              const arr = forecast?.[src]
              return (
                <Card key={src}>
                  {/* forecast not yet loaded → spinner */}
                  {!forecast ? (
                    <div style={{ height: 160, display: 'flex', alignItems: 'center',
                                  justifyContent: 'center', flexDirection: 'column', color: MUTED }}>
                      <div style={{ fontSize: '1.5rem' }}>{icons[src]}</div>
                      <div style={{ fontWeight: 600, marginTop: 4 }}>{src.toUpperCase()}</div>
                      <div style={{ fontSize: '0.72rem', marginTop: 2 }}>Loading…</div>
                    </div>
                  ) : arr ? (
                    /* forecast loaded AND this source has data */
                    <>
                      <HeatmapSVG grid={arr} lats={forecast.lats} lons={forecast.lons}
                        title={`${icons[src]} ${src.toUpperCase()}`}
                        colorHigh={SOURCE_COLORS[src]}
                        unit={variable === 'rainfall' ? 'mm' : variable === 'temperature' ? '°C' : 'm/s'} />
                      <div style={{ fontSize: '0.68rem', color: MUTED, marginTop: 4 }}>
                        {srcDesc[src]}
                      </div>
                    </>
                  ) : (
                    /* forecast loaded but this source missing (real-data GFS-only mode) */
                    <div style={{ height: 160, display: 'flex', alignItems: 'center',
                                  justifyContent: 'center', flexDirection: 'column', color: MUTED }}>
                      <div style={{ fontSize: '1.5rem' }}>{icons[src]}</div>
                      <div style={{ fontWeight: 600, marginTop: 4 }}>{src.toUpperCase()}</div>
                      <div style={{ fontSize: '0.72rem', marginTop: 2, textAlign: 'center' }}>
                        No data available<br />
                        <span style={{ color: LABEL }}>Add {src.toUpperCase()} files to enable</span>
                      </div>
                    </div>
                  )}
                </Card>
              )
            })}
          </div>

          {/* RMSE chart */}
          <SectionHead>Forecast Skill — RMSE by Lead Time</SectionHead>
          {skill?.bar_data ? (
            <Card>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={skill.bar_data} margin={{ top: 10, right: 20, bottom: 30, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="lead_time" tickFormatter={l => `${l}h`}
                    tick={{ fill: MUTED, fontSize: 11 }} />
                  <YAxis tick={{ fill: MUTED, fontSize: 11 }} label={{ value: 'RMSE', angle: -90,
                    position: 'insideLeft', fill: MUTED, fontSize: 11 }} />
                  <Tooltip contentStyle={{ background: SURFACE, border: `1px solid ${BORDER}`,
                    borderRadius: 6, color: TEXT }} />
                  <Legend wrapperStyle={{ color: MUTED, fontSize: 11 }} />
                  {Object.entries(SOURCE_COLORS).map(([src, col]) => (
                    skill.bar_data[0]?.[src] !== undefined &&
                    <Bar key={src} dataKey={src} fill={col} name={src} radius={[3,3,0,0]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </Card>
          ) : (
            <Card style={{ height: 120, display:'flex', alignItems:'center',
                           justifyContent:'center', color: MUTED }}>Loading…</Card>
          )}

          {/* Region RMSE lines */}
          <SectionHead>RMSE by Region @ {LEAD_LABELS[leadTime]}</SectionHead>
          {skill?.region_data?.length > 0 ? (
            <Card>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={skill.region_data} margin={{ top: 10, right: 20, bottom: 40, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="region" tick={{ fill: MUTED, fontSize: 10 }} />
                  <YAxis tick={{ fill: MUTED, fontSize: 11 }} />
                  <Tooltip contentStyle={{ background: SURFACE, border: `1px solid ${BORDER}`,
                    borderRadius: 6, color: TEXT }} />
                  <Legend wrapperStyle={{ color: MUTED, fontSize: 11 }} />
                  {Object.entries(SOURCE_COLORS).map(([src, col]) => (
                    skill.region_data[0]?.[src] !== undefined &&
                    <Line key={src} type="monotone" dataKey={src} stroke={col}
                      strokeWidth={src === 'blended' ? 3 : 1.5} dot={{ r: 3 }} name={src} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </Card>
          ) : null}

          {/* Weight maps */}
          <SectionHead>🗺️ Model Weight Maps</SectionHead>
          {weightMaps ? (
            <>
              <div style={{ fontSize: '0.75rem', color: MUTED, marginBottom: '0.5rem' }}>
                How much each model is trusted in each region for the current variable,
                lead time, season and weather regime. Brighter cells = higher trust.
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '0.75rem', marginBottom: '0.75rem' }}>
                {(weightMaps.sources || []).map(src => {
                  const icons = { nwp: '🏗️', ensemble: '🧩', ai: '🤖' }
                  const grid = weightMaps.source_maps?.[src]
                  return (
                    <Card key={src}>
                      <HeatmapSVG
                        grid={grid}
                        lats={weightMaps.lats}
                        lons={weightMaps.lons}
                        title={`${icons[src] || ''} ${src.toUpperCase()} Weight`}
                        colorHigh={SOURCE_COLORS[src] || '#60a5fa'}
                        unit="weight"
                      />
                    </Card>
                  )
                })}
              </div>
              <Card>
                <HeatmapSVG
                  grid={weightMaps.dominant_map}
                  lats={weightMaps.lats}
                  lons={weightMaps.lons}
                  title="Dominant Source Map — most-trusted model per region"
                  colorHigh="#6366f1"
                  unit="index"
                />
                <div style={{ fontSize: '0.72rem', color: MUTED, marginTop: 4 }}>
                  Each cell is coloured by the model index with the highest weight for the
                  current variable / lead / season / regime.
                </div>
              </Card>
            </>
          ) : (
            <Card style={{ height: 100, display: 'flex', alignItems: 'center',
                           justifyContent: 'center', color: MUTED }}>Loading…</Card>
          )}

          {/* Extreme weather */}
          <SectionHead>⚡ Extreme Weather Indicators</SectionHead>
          {extreme ? (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '0.75rem' }}>
              {Object.entries(extreme.hazards).map(([vname, info]) => {
                const meta = EW_META[vname]
                const status = info.frac_flagged > 0
                  ? { icon: '🔴', text: `${info.frac_flagged.toFixed(1)}% cells flagged`, color: '#ef4444' }
                  : info.domain_max >= 0.7 * (info.threshold || 1)
                    ? { icon: '🟡', text: 'Approaching threshold', color: '#f59e0b' }
                    : { icon: '🟢', text: 'No event', color: '#22c55e' }
                return (
                  <Card key={vname} style={{ borderColor: meta.color + '66',
                    background: meta.color + '14' }}>
                    <div style={{ fontSize: '1.4rem' }}>{meta.icon}</div>
                    <div style={{ fontWeight: 700, marginTop: 4 }}>{meta.label}</div>
                    <div style={{ fontSize: '0.72rem', color: MUTED, marginTop: 2 }}>
                      Threshold: ≥{info.threshold} {meta.unit}
                    </div>
                    <div style={{ fontWeight: 600, color: status.color, marginTop: 6, fontSize: '0.82rem' }}>
                      {status.icon} {status.text}
                    </div>
                    <div style={{ fontSize: '0.72rem', color: MUTED, marginTop: 3 }}>
                      Max: <strong style={{ color: TEXT }}>{info.domain_max.toFixed(1)} {meta.unit}</strong>
                      &nbsp;·&nbsp;
                      Mean: <strong style={{ color: TEXT }}>{info.domain_mean.toFixed(1)} {meta.unit}</strong>
                    </div>
                  </Card>
                )
              })}
            </div>
          ) : (
            <Card style={{ height: 100, display:'flex', alignItems:'center',
                           justifyContent:'center', color: MUTED }}>Loading…</Card>
          )}

          {/* Footer */}
          <div style={{ marginTop: '2rem', paddingTop: '1rem', textAlign: 'center',
                        borderTop: `1px solid rgba(255,255,255,0.08)`, fontSize: '0.72rem',
                        color: LABEL }}>
            PrakritiNetra — AI-NWP Adaptive Blending · India Forecast Domain ·
            Backend on <a href="https://render.com" style={{ color: HL }}>Render</a> ·
            Frontend on <a href="https://vercel.com" style={{ color: HL }}>Vercel</a>
          </div>
        </main>
      </div>
    </div>
  )
}
