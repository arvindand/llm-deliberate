import React, { useState, useEffect, useMemo, useRef } from 'react'
import PropTypes from 'prop-types'
import { Zap, Loader, AlertCircle, CheckCircle, ChevronDown, ChevronUp, ChevronRight } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const API_BASE = '/api'

const USD_COST_FORMATTER = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  minimumFractionDigits: 2,
  maximumFractionDigits: 6,
})

const MARKDOWN_COMPONENTS = {
  h1: ({ children, ...props }) => (
    <h1 {...props} className="text-xl font-bold text-ink mt-4 mb-2">
      {children}
    </h1>
  ),
  h2: ({ children, ...props }) => (
    <h2 {...props} className="text-lg font-semibold text-ink mt-3 mb-2">
      {children}
    </h2>
  ),
  h3: ({ children, ...props }) => (
    <h3 {...props} className="text-base font-semibold text-ink mt-2 mb-1">
      {children}
    </h3>
  ),
  p: ({ children, ...props }) => (
    <p {...props} className="text-sm text-ink leading-relaxed mb-1">
      {children}
    </p>
  ),
  ul: ({ children, ...props }) => (
    <ul {...props} className="text-sm text-ink leading-relaxed list-disc ml-6 my-1">
      {children}
    </ul>
  ),
  ol: ({ children, ...props }) => (
    <ol {...props} className="text-sm text-ink leading-relaxed list-decimal ml-6 my-1">
      {children}
    </ol>
  ),
  li: ({ children, ...props }) => (
    <li {...props} className="text-sm text-ink leading-relaxed">
      {children}
    </li>
  ),
  a: ({ children, ...props }) => (
    <a {...props} className="text-sepia underline hover:text-rust" target="_blank" rel="noreferrer noopener">
      {children}
    </a>
  ),
  table: ({ children, ...props }) => (
    <div className="overflow-x-auto my-2">
      <table {...props} className="border-collapse w-full">
        {children}
      </table>
    </div>
  ),
  th: ({ children, ...props }) => (
    <th {...props} className="text-left text-xs font-semibold text-slate border border-sepia/20 bg-parchment/30 px-2 py-1 align-top">
      {children}
    </th>
  ),
  td: ({ children, ...props }) => (
    <td {...props} className="text-xs text-ink border border-sepia/20 px-2 py-1 align-top">
      {children}
    </td>
  ),
  code: ({ inline, children, ...props }) => {
    if (inline) {
      return (
        <code {...props} className="bg-slate/5 px-1 rounded text-xs font-mono">
          {children}
        </code>
      )
    }
    return (
      <code {...props} className="text-xs font-mono">
        {children}
      </code>
    )
  },
  pre: ({ children, ...props }) => (
    <pre {...props} className="bg-slate/5 p-3 rounded text-xs font-mono overflow-x-auto my-2">
      {children}
    </pre>
  ),
  blockquote: ({ children, ...props }) => (
    <blockquote {...props} className="border-l-2 border-sepia/30 pl-3 text-sm text-slate italic my-2">
      {children}
    </blockquote>
  ),
  hr: (props) => <hr {...props} className="border-sepia/20 my-3" />,
}

function sortModelsWithTopPinned(models) {
  const hasOpenRouterOrder = (models || []).some(m => Number.isFinite(m?.openrouter_order))

  const fallbackTopIds = [
    'openai/gpt-4o',
    'anthropic/claude-3.5-sonnet',
    'anthropic/claude-3-opus',
    'openai/gpt-4-turbo',
  ]
  const fallbackTopIndex = new Map(fallbackTopIds.map((id, idx) => [id, idx]))

  const getRank = hasOpenRouterOrder
    ? (model) => (Number.isFinite(model?.openrouter_order) ? model.openrouter_order : Number.POSITIVE_INFINITY)
    : (model) => (fallbackTopIndex.get(model?.id) ?? 1000)

  const getProvider = (model) => (model?.id || '').split('/')[0] || ''
  const getName = (model) => model?.name || model?.id || ''

  return [...(models || [])].sort((a, b) => {
    const aRank = getRank(a)
    const bRank = getRank(b)
    if (aRank !== bRank) return aRank - bRank

    const aProvider = getProvider(a)
    const bProvider = getProvider(b)
    if (aProvider !== bProvider) return aProvider.localeCompare(bProvider)

    return getName(a).localeCompare(getName(b))
  })
}

async function fetchAvailableModels(setAvailable, setError, setLoading) {
  try {
    const response = await fetchAPI('/config/models')
    setAvailable(response.models || [])
  } catch (err) {
    setError(`Failed to load models: ${err.message}`)
  } finally {
    setLoading(false)
  }
}

function getProviderIds(models) {
  return models?.length ? [...new Set(models.map(m => m.id.split('/')[0]))] : []
}

function getJobStatusDisplay(status, labels) {
  if (status === 'completed') {
    return { icon: <CheckCircle className="w-5 h-5 text-green-600" />, text: labels.completed }
  }
  if (status === 'failed') {
    return { icon: <AlertCircle className="w-5 h-5 text-rust" />, text: labels.failed }
  }
  return { icon: <Loader className="w-5 h-5 animate-spin text-sepia" />, text: labels.inProgress }
}

function getDeliberationStatus(jobStatus, maxRounds) {
  if (jobStatus?.status === 'completed') {
    return { icon: <CheckCircle className="w-5 h-5 text-green-600" />, text: 'Deliberation complete!' }
  }
  if (jobStatus?.status === 'failed') {
    return { icon: <AlertCircle className="w-5 h-5 text-rust" />, text: 'Deliberation failed' }
  }
  if (jobStatus?.progress?.converged) {
    return { icon: <Loader className="w-5 h-5 animate-spin text-sepia" />, text: 'Converged early!' }
  }
  if (jobStatus?.progress?.round) {
    return {
      icon: <Loader className="w-5 h-5 animate-spin text-sepia" />,
      text: `Deliberating Round ${jobStatus.progress.round} of ${maxRounds}...`
    }
  }
  return { icon: <Loader className="w-5 h-5 animate-spin text-sepia" />, text: 'Starting deliberation...' }
}

function toggleSelection(id, setState) {
  setState(prev => (prev.includes(id) ? prev.filter(item => item !== id) : [...prev, id]))
}

function mergeUniqueById(existingItems, newItems) {
  if (!Array.isArray(newItems) || newItems.length === 0) return existingItems

  const seen = new Set((existingItems || []).map(item => item?.id))
  const merged = [...(existingItems || [])]
  for (const item of newItems) {
    if (item?.id && !seen.has(item.id)) {
      merged.push(item)
      seen.add(item.id)
    }
  }
  return merged
}

async function fetchAPI(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const error = await res.json().catch(() => ({}))
    throw new Error(error.detail || 'API error')
  }
  return res.json()
}

function subscribeToJobStatus({ experimentId, jobId, onStatus, onPartialResults, onError }) {
  const url = `${API_BASE}/experiments/${experimentId}/automation/stream/${jobId}`
  const es = new EventSource(url)

  const handler = (event) => {
    try {
      const data = JSON.parse(event.data)
      onStatus(data)
    } catch (err) {
      onError(err)
    }
  }

  es.addEventListener('status', handler)

  if (onPartialResults) {
    es.addEventListener('partial_results', (event) => {
      try {
        const data = JSON.parse(event.data)
        onPartialResults(data)
      } catch (err) {
        onError(err)
      }
    })
  }

  es.onerror = (e) => {
    es.close()
    onError(e)
  }

  return () => es.close()
}

function stopJobStream(closeStreamRef) {
  if (closeStreamRef?.current) closeStreamRef.current()
  if (closeStreamRef) closeStreamRef.current = null
}

function startJobStream({ experimentId, jobId, closeStreamRef, onStatus, onPartialResults, onError }) {
  stopJobStream(closeStreamRef)
  closeStreamRef.current = subscribeToJobStatus({
    experimentId,
    jobId,
    onStatus,
    onPartialResults,
    onError,
  })
}

function createAutomationStatusHandler({
  closeStreamRef,
  setJobStatus,
  setJobInProgress,
  startedJobIdRef,
  onJobStarted,
  resultType,
}) {
  return (status) => {
    setJobStatus(status)

    if (status.status === 'completed') {
      stopJobStream(closeStreamRef)
      setJobInProgress(false)
      const jobId = startedJobIdRef.current
      if (jobId) setTimeout(() => onJobStarted(jobId, resultType, true), 2000)
      return
    }

    if (status.status === 'failed') {
      // Stop streaming updates, but keep the in-progress view visible so the
      // user can see jobStatus.errors instead of auto-refreshing to "nothing".
      stopJobStream(closeStreamRef)
    }
  }
}

function createAutomationPartialHandler(setPartialResults) {
  return (payload) => {
    const items = payload?.items || []
    if (!items.length) return
    setPartialResults(prev => mergeUniqueById(prev, items))
  }
}

// === AutomatedResponseForm ===

export function AutomatedResponseForm({ experimentId, questionId, question, onJobStarted, onClose }) {
  const [selectedModels, setSelectedModels] = useState([])
  const [availableModels, setAvailableModels] = useState([])
  const [selectedProvider, setSelectedProvider] = useState(null)
  const [loading, setLoading] = useState(true)
  const [jobInProgress, setJobInProgress] = useState(false)
  const [jobStatus, setJobStatus] = useState(null)
  const [error, setError] = useState(null)
  const [partialResults, setPartialResults] = useState([])
  const closeStreamRef = useRef(null)
  const startedJobIdRef = useRef(null)
  const pollRetryCount = useRef(0)

  useEffect(() => {
    fetchAvailableModels(setAvailableModels, setError, setLoading)
  }, [])

  useEffect(() => () => {
    if (closeStreamRef.current) closeStreamRef.current()
  }, [])

  const handleStreamStatus = createAutomationStatusHandler({
    closeStreamRef,
    setJobStatus,
    setJobInProgress,
    startedJobIdRef,
    onJobStarted,
    resultType: 'responses',
  })
  const handleStreamPartialResults = createAutomationPartialHandler(setPartialResults)

  const estimatedCost = useMemo(() => {
    if (selectedModels.length === 0) return 0
    const questionTokens = Math.ceil((question?.text?.length || 0) / 4)
    const avgResponseTokens = 500
    return selectedModels.reduce((sum, modelId) => {
      const model = availableModels.find(m => m.id === modelId)
      if (!model?.pricing) return sum
      const inputCost = (questionTokens / 1000000) * model.pricing.prompt
      const outputCost = (avgResponseTokens / 1000000) * model.pricing.completion
      return sum + inputCost + outputCost
    }, 0)
  }, [selectedModels, question?.text, availableModels])

  function handleModelToggle(modelId) {
    toggleSelection(modelId, setSelectedModels)
  }

  function handleProviderFilter(provider) {
    setSelectedProvider(prev => (prev === provider ? null : provider))
  }

  async function handleStartJob() {
    if (selectedModels.length === 0) return

    setJobInProgress(true)
    setError(null)
    pollRetryCount.current = 0  // Reset backoff counter
    try {
      const response = await fetchAPI(
        `/experiments/${experimentId}/automate/responses`,
        {
          method: 'POST',
          body: JSON.stringify({
            question_id: questionId,
            models: selectedModels
          })
        }
      )

      setJobStatus(response)
      setPartialResults([])
      startedJobIdRef.current = response.job_id

      // Stream job status (fallback to polling if SSE fails)
      startJobStream({
        experimentId,
        jobId: response.job_id,
        closeStreamRef,
        onStatus: handleStreamStatus,
        onPartialResults: handleStreamPartialResults,
        onError: () => {
          pollJobStatus(response.job_id)
        },
      })
    } catch (err) {
      setError(`Failed to start automation: ${err.message}`)
      setJobInProgress(false)
    }
  }

  async function pollJobStatus(jobId) {
    try {
      const status = await fetchAPI(`/experiments/${experimentId}/automation/status/${jobId}`)
      setJobStatus(status)

      if (status.status === 'completed') {
        setJobInProgress(false)
        pollRetryCount.current = 0  // Reset on completion
        // Wait 2 seconds before auto-closing to show completion state
        setTimeout(() => {
          onJobStarted(jobId, 'responses', true)
        }, 2000)
      } else if (status.status === 'failed') {
        pollRetryCount.current = 0  // Reset on failure
        // Keep the error state visible; don't auto-refresh.
        stopJobStream(closeStreamRef)
      } else {
        // Exponential backoff: 500ms → 750ms → 1125ms → max 3s
        pollRetryCount.current += 1
        const delay = Math.min(500 * Math.pow(1.5, pollRetryCount.current), 3000)
        setTimeout(() => pollJobStatus(jobId), delay)
      }
    } catch (err) {
      setError(`Failed to check status: ${err.message}`)
      setJobInProgress(false)
      pollRetryCount.current = 0  // Reset on error
    }
  }

  const providers = useMemo(() => getProviderIds(availableModels), [availableModels])
  const filteredModels = useMemo(() => {
    const models = selectedProvider
      ? availableModels.filter(m => m.id.startsWith(`${selectedProvider}/`))
      : availableModels
    return sortModelsWithTopPinned(models)
  }, [availableModels, selectedProvider])

  return (
    <div className="mb-4 p-4 bg-gradient-to-br from-white to-parchment/30 rounded-lg border border-sepia/20 shadow-sm animate-slide-up">
      {jobInProgress ? (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            {getJobStatusDisplay(jobStatus?.status, {
              completed: 'Collection complete!',
              failed: 'Collection failed',
              inProgress: 'Collecting responses...'
            }).icon}
            <div className="flex-1">
              <div className="text-sm font-semibold text-ink">
                {getJobStatusDisplay(jobStatus?.status, {
                  completed: 'Collection complete!',
                  failed: 'Collection failed',
                  inProgress: 'Collecting responses...'
                }).text}
              </div>
              {jobStatus?.message && (
                <div className="text-xs text-slate">{jobStatus.message}</div>
              )}
            </div>
          </div>

          {jobStatus?.progress && (
            <div className="bg-white rounded-lg p-3">
              <div className="flex items-center justify-between text-sm mb-2">
                <span className="text-slate">Progress</span>
                <span className="font-mono text-ink">
                  {jobStatus.progress.completed}/{jobStatus.progress.total}
                </span>
              </div>
              <div className="w-full bg-sepia/10 rounded-full h-2">
                <div
                  className="bg-sepia rounded-full h-2 transition-all duration-300"
                  style={{
                    width: `${Math.min((jobStatus.progress.completed / jobStatus.progress.total) * 100, 100)}%`
                  }}
                />
              </div>
            </div>
          )}

          {jobStatus?.errors && jobStatus.errors.length > 0 && (
            <div className="bg-rust/5 rounded-lg p-3 border border-rust/20">
              <div className="text-sm font-semibold text-rust mb-1">Errors:</div>
              <ul className="text-xs text-rust space-y-1">
                {jobStatus.errors.map(err => {
                  const errorKey = `${err.model || 'unknown'}-${err.message || 'error'}`
                  return (
                    <li key={errorKey}>{err.model}: {err.message}</li>
                  )
                })}
              </ul>
            </div>
          )}

          {partialResults.length > 0 && (
            <div className="bg-sepia/5 rounded-lg p-3 border border-sepia/10">
              <div className="text-xs font-semibold text-slate mb-2">Received so far</div>
              <div className="flex flex-wrap gap-2">
                {partialResults.slice(0, 12).map(item => (
                  <span key={item.id} className="text-xs px-2 py-1 rounded bg-white border border-sepia/20 text-ink">
                    {item.model || item.id}
                  </span>
                ))}
                {partialResults.length > 12 && (
                  <span className="text-xs text-slate">+{partialResults.length - 12} more</span>
                )}
              </div>
            </div>
          )}
        </div>
      ) : (
        <>
          <div className="mb-4">
            <h4 className="text-sm font-semibold text-ink mb-2">Select Models</h4>
            {loading ? (
              <div className="flex items-center gap-2 text-slate">
                <Loader className="w-4 h-4 animate-spin" />
                <span className="text-sm">Loading models...</span>
              </div>
            ) : (
              <>
                <div className="flex flex-wrap gap-2 mb-3">
                  {providers.map(provider => (
                    <button
                      key={provider}
                      type="button"
                      onClick={() => handleProviderFilter(provider)}
                      className={`text-xs px-2 py-1 rounded transition-colors ${selectedProvider === provider
                        ? 'bg-sepia text-white'
                        : 'bg-sepia/10 hover:bg-sepia/20 text-sepia'
                        }`}
                    >
                      {provider}
                    </button>
                  ))}
                </div>

                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {filteredModels.map(model => (
                    <label
                      key={model.id}
                      className="flex items-center gap-2 p-2 hover:bg-sepia/5 rounded cursor-pointer transition-colors"
                    >
                      <input
                        type="checkbox"
                        checked={selectedModels.includes(model.id)}
                        onChange={() => handleModelToggle(model.id)}
                        className="rounded border-sepia/30"
                      />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium text-ink truncate">{model.name}</span>
                          <span className="text-xs text-slate font-mono">{model.id}</span>
                        </div>
                        {model.pricing && (
                          <div className="text-xs text-slate">
                            ${(model.pricing.prompt * 1000).toFixed(3)}/1K in ·
                            ${(model.pricing.completion * 1000).toFixed(3)}/1K out
                          </div>
                        )}
                      </div>
                    </label>
                  ))}
                </div>
              </>
            )}
          </div>

          {selectedModels.length > 0 && (
            <div className="bg-sepia/5 rounded p-3 mb-3 border border-sepia/10">
              <div className="text-xs text-slate mb-1">
                Selected: {selectedModels.length} model{selectedModels.length === 1 ? '' : 's'} = <strong>{selectedModels.length} API call{selectedModels.length === 1 ? '' : 's'}</strong>
              </div>
              <div className="text-sm font-semibold text-ink">
                Estimated cost: ${estimatedCost.toFixed(4)}
              </div>
              {estimatedCost > 0.5 && (
                <div className="text-xs text-rust mt-1">
                  ⚠️ High cost warning: Ensure your OpenRouter balance is sufficient.
                </div>
              )}
              {selectedModels.length > 10 && (
                <div className="text-xs text-rust mt-1 font-semibold">
                  🚨 Consider starting with fewer models to test first.
                </div>
              )}
            </div>
          )}

          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="btn btn-secondary flex-1 text-sm py-1.5"
            >
              Cancel
            </button>
            <button
              onClick={handleStartJob}
              disabled={selectedModels.length === 0 || loading}
              className="btn btn-primary flex-1 text-sm py-1.5"
            >
              Collect Responses
            </button>
          </div>

          {error && (
            <div className="bg-rust/5 rounded-lg p-3 border border-rust/20 mt-3">
              <p className="text-sm text-rust">{error}</p>
            </div>
          )}
        </>
      )}
    </div>
  )
}

// === AutomatedRankingForm ===

export function AutomatedRankingForm({ experimentId, questionId, question, onJobStarted, onClose }) {
  const [selectedJudges, setSelectedJudges] = useState([])
  const [availableModels, setAvailableModels] = useState([])
  const [selectedProvider, setSelectedProvider] = useState(null)
  const [loading, setLoading] = useState(true)
  const [jobInProgress, setJobInProgress] = useState(false)
  const [jobStatus, setJobStatus] = useState(null)
  const [error, setError] = useState(null)
  const [partialResults, setPartialResults] = useState([])
  const closeStreamRef = useRef(null)
  const startedJobIdRef = useRef(null)
  const pollRetryCount = useRef(0)

  useEffect(() => {
    fetchAvailableModels(setAvailableModels, setError, setLoading)
  }, [])

  useEffect(() => () => {
    if (closeStreamRef.current) closeStreamRef.current()
  }, [])

  const handleStreamStatus = createAutomationStatusHandler({
    closeStreamRef,
    setJobStatus,
    setJobInProgress,
    startedJobIdRef,
    onJobStarted,
    resultType: 'rankings',
  })
  const handleStreamPartialResults = createAutomationPartialHandler(setPartialResults)

  function handleJudgeToggle(modelId) {
    toggleSelection(modelId, setSelectedJudges)
  }

  function handleUseResponseModels() {
    const responseModels = (question?.responses || []).map(r => {
      // Find matching model config by matching the model ID from the response
      // Response model format: "anthropic/claude-opus-4.5"
      // Available model ID format: "anthropic/claude-opus-4.5"
      const model = availableModels.find(m =>
        m.id === r.model ||
        m.id.toLowerCase() === r.model.toLowerCase() ||
        r.model.toLowerCase().includes(m.id.toLowerCase()) ||
        m.id.toLowerCase().includes(r.model.toLowerCase())
      )
      return model ? model.id : null
    }).filter(Boolean)

    console.log('Response models:', question?.responses?.map(r => r.model))
    console.log('Matched judge models:', responseModels)
    setSelectedJudges([...new Set(responseModels)])
  }

  function handleProviderFilter(provider) {
    setSelectedProvider(prev => (prev === provider ? null : provider))
  }

  async function handleStartJob() {
    if (selectedJudges.length === 0) return

    setJobInProgress(true)
    setError(null)
    pollRetryCount.current = 0  // Reset backoff counter
    try {
      console.log('Starting ranking job with judges:', selectedJudges)
      const response = await fetchAPI(
        `/experiments/${experimentId}/automate/rankings`,
        {
          method: 'POST',
          body: JSON.stringify({
            question_id: questionId,
            judges: selectedJudges
          })
        }
      )

      console.log('Ranking job started:', response)
      setJobStatus(response)
      setPartialResults([])
      startedJobIdRef.current = response.job_id
      // Don't call onJobStarted here - only call it when completed in pollJobStatus
      // onJobStarted(response.job_id, 'rankings')

      // Stream job status (fallback to polling if SSE fails)
      startJobStream({
        experimentId,
        jobId: response.job_id,
        closeStreamRef,
        onStatus: handleStreamStatus,
        onPartialResults: handleStreamPartialResults,
        onError: () => {
          pollJobStatus(response.job_id)
        },
      })
    } catch (err) {
      console.error('Failed to start ranking automation:', err)
      setError(`Failed to start ranking automation: ${err.message}`)
      setJobInProgress(false)
    }
  }

  async function pollJobStatus(jobId) {
    try {
      const status = await fetchAPI(`/experiments/${experimentId}/automation/status/${jobId}`)
      setJobStatus(status)

      if (status.status === 'completed') {
        setJobInProgress(false)
        pollRetryCount.current = 0  // Reset on completion
        // Wait 2 seconds before auto-closing to show completion state
        setTimeout(() => {
          onJobStarted(jobId, 'rankings', true)
        }, 2000)
      } else if (status.status === 'failed') {
        pollRetryCount.current = 0  // Reset on failure
        // Keep the error state visible; don't auto-refresh.
        stopJobStream(closeStreamRef)
      } else {
        // Exponential backoff: 500ms → 750ms → 1125ms → max 3s
        pollRetryCount.current += 1
        const delay = Math.min(500 * Math.pow(1.5, pollRetryCount.current), 3000)
        setTimeout(() => pollJobStatus(jobId), delay)
      }
    } catch (err) {
      setError(`Failed to check status: ${err.message}`)
      setJobInProgress(false)
      pollRetryCount.current = 0  // Reset on error
    }
  }

  const providers = useMemo(() => getProviderIds(availableModels), [availableModels])
  const filteredModels = useMemo(() => {
    const models = selectedProvider
      ? availableModels.filter(m => m.id.startsWith(`${selectedProvider}/`))
      : availableModels
    return sortModelsWithTopPinned(models)
  }, [availableModels, selectedProvider])

  return (
    <div className="mb-4 p-4 bg-gradient-to-br from-white to-parchment/30 rounded-lg border border-sepia/20 shadow-sm animate-slide-up">
      {jobInProgress ? (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            {getJobStatusDisplay(jobStatus?.status, {
              completed: 'Rankings collected!',
              failed: 'Ranking failed',
              inProgress: 'Collecting rankings...'
            }).icon}
            <div className="flex-1">
              <div className="text-sm font-semibold text-ink">
                {getJobStatusDisplay(jobStatus?.status, {
                  completed: 'Rankings collected!',
                  failed: 'Ranking failed',
                  inProgress: 'Collecting rankings...'
                }).text}
              </div>
              {jobStatus?.message && (
                <div className="text-xs text-slate">{jobStatus.message}</div>
              )}
            </div>
          </div>

          {jobStatus?.progress && (
            <div className="bg-white rounded-lg p-3">
              <div className="flex items-center justify-between text-sm mb-2">
                <span className="text-slate">Progress</span>
                <span className="font-mono text-ink">
                  {jobStatus.progress.completed}/{jobStatus.progress.total}
                </span>
              </div>
              <div className="w-full bg-sepia/10 rounded-full h-2">
                <div
                  className="bg-sepia rounded-full h-2 transition-all duration-300"
                  style={{
                    width: `${Math.min((jobStatus.progress.completed / jobStatus.progress.total) * 100, 100)}%`
                  }}
                />
              </div>
            </div>
          )}

          {jobStatus?.errors && jobStatus.errors.length > 0 && (
            <div className="bg-rust/5 rounded-lg p-3 border border-rust/20">
              <div className="text-sm font-semibold text-rust mb-1">Errors:</div>
              <ul className="text-xs text-rust space-y-1">
                {jobStatus.errors.map(err => {
                  const errorKey = `${err.model || 'unknown'}-${err.message || 'error'}`
                  return (
                    <li key={errorKey}>{err.model}: {err.message}</li>
                  )
                })}
              </ul>
            </div>
          )}

          {partialResults.length > 0 && (
            <div className="bg-sepia/5 rounded-lg p-3 border border-sepia/10">
              <div className="text-xs font-semibold text-slate mb-2">Collected so far</div>
              <div className="flex flex-wrap gap-2">
                {partialResults.slice(0, 12).map(item => (
                  <span key={item.id} className="text-xs px-2 py-1 rounded bg-white border border-sepia/20 text-ink">
                    {item.judge || item.id}
                  </span>
                ))}
                {partialResults.length > 12 && (
                  <span className="text-xs text-slate">+{partialResults.length - 12} more</span>
                )}
              </div>
            </div>
          )}
        </div>
      ) : (
        <>
          <div className="mb-4">
            <h4 className="text-sm font-semibold text-ink mb-2">Select Judge Models</h4>
            {loading ? (
              <div className="flex items-center gap-2 text-slate">
                <Loader className="w-4 h-4 animate-spin" />
                <span className="text-sm">Loading models...</span>
              </div>
            ) : (
              <>
                {(question.responses?.length || 0) > 0 && (
                  <button
                    onClick={handleUseResponseModels}
                    className="w-full mb-3 px-4 py-2.5 bg-gradient-to-r from-sepia/10 to-rust/10 hover:from-sepia/20 hover:to-rust/20 border border-sepia/30 rounded-lg text-sm font-semibold text-sepia hover:text-rust transition-all flex items-center justify-center gap-2 shadow-sm hover:shadow-md"
                  >
                    <Zap className="w-4 h-4" />
                    Use Response Models as Judges
                  </button>
                )}

                <div className="flex flex-wrap gap-2 mb-3">
                  {providers.map(provider => (
                    <button
                      key={provider}
                      type="button"
                      onClick={() => handleProviderFilter(provider)}
                      className={`text-xs px-2 py-1 rounded transition-colors ${selectedProvider === provider
                        ? 'bg-sepia text-white'
                        : 'bg-sepia/10 hover:bg-sepia/20 text-sepia'
                        }`}
                    >
                      {provider}
                    </button>
                  ))}
                </div>

                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {filteredModels.map(model => (
                    <label
                      key={model.id}
                      className="flex items-center gap-2 p-2 hover:bg-sepia/5 rounded cursor-pointer transition-colors"
                    >
                      <input
                        type="checkbox"
                        checked={selectedJudges.includes(model.id)}
                        onChange={() => handleJudgeToggle(model.id)}
                        aria-label={`Select judge model ${model.name || model.id}`}
                        className="rounded border-sepia/30"
                      />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium text-ink truncate">{model.name}</span>
                          <span className="text-xs text-slate font-mono">{model.id}</span>
                        </div>
                      </div>
                    </label>
                  ))}
                </div>
              </>
            )}
          </div>

          {selectedJudges.length > 0 && (
            <div className="bg-sepia/5 rounded p-3 mb-3 border border-sepia/10">
              <div className="text-xs text-slate">
                Selected: {selectedJudges.length} judge{selectedJudges.length === 1 ? '' : 's'}
              </div>
            </div>
          )}

          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="btn btn-secondary flex-1 text-sm py-1.5"
            >
              Cancel
            </button>
            <button
              onClick={handleStartJob}
              disabled={selectedJudges.length === 0 || loading}
              className="btn btn-primary flex-1 text-sm py-1.5"
            >
              Collect Rankings
            </button>
          </div>

          {error && (
            <div className="bg-rust/5 rounded-lg p-3 border border-rust/20">
              <p className="text-sm text-rust">{error}</p>
            </div>
          )}
        </>
      )}
    </div>
  )
}

// === MarkdownRenderer ===

export function MarkdownRenderer({ content }) {
  if (!content) return null

  return (
    <div className="space-y-1">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={MARKDOWN_COMPONENTS}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}

MarkdownRenderer.propTypes = {
  content: PropTypes.string
}

// === ResponseCard ===

export function ResponseCard({ response, index }) {
  const [isExpanded, setIsExpanded] = useState(false)

  const contentPreview = (response.content || '').slice(0, 300)
  const isTruncated = (response.content || '').length > 300

  return (
    <div className="bg-white rounded-lg border border-sepia/20 shadow-sm hover:shadow-md transition-shadow">
      <div className="p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            {response.round > 1 && (
              <span className="text-xs font-semibold text-rust bg-rust/10 px-2 py-1 rounded">
                Round {response.round}
              </span>
            )}
            <span className="text-xs font-semibold text-sepia bg-sepia/10 px-2 py-1 rounded">
              Response {String.fromCodePoint(65 + index)}
            </span>
            <span className={`model-badge ${response.model.split('/')[0]}`}>
              {response.model}
            </span>
            <span className="text-xs text-slate font-mono">{response.id}</span>
          </div>

          {response.metadata && (
            <div className="flex items-center gap-3 text-xs text-slate">
              {response.metadata.tokens_output && (
                <span title="Output tokens">{response.metadata.tokens_output} tokens</span>
              )}
              {response.metadata.latency_ms && (
                <span title="Latency">{(response.metadata.latency_ms / 1000).toFixed(1)}s</span>
              )}
              {response.metadata.cost_usd && (
                <span title="Cost">{USD_COST_FORMATTER.format(response.metadata.cost_usd)}</span>
              )}
            </div>
          )}
        </div>

        <div className="prose prose-sm max-w-none">
          {isExpanded ? (
            <MarkdownRenderer content={response.content} />
          ) : (
            <div className="text-sm text-ink leading-relaxed">
              {isTruncated ? contentPreview + '...' : response.content}
            </div>
          )}
        </div>

        {isTruncated && (
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="mt-3 text-xs text-sepia hover:text-rust font-medium flex items-center gap-1 transition-colors"
          >
            {isExpanded ? (
              <>
                <ChevronUp className="w-3 h-3" />
                Show less
              </>
            ) : (
              <>
                <ChevronDown className="w-3 h-3" />
                Show more
              </>
            )}
          </button>
        )}
      </div>
    </div>
  )
}

// === AutomatedDeliberationForm ===

export function AutomatedDeliberationForm({ experimentId, questionId, question, onJobStarted, onClose }) {
  const [selectedModels, setSelectedModels] = useState([])
  const [availableModels, setAvailableModels] = useState([])
  const [selectedProvider, setSelectedProvider] = useState(null)
  const [loading, setLoading] = useState(true)
  const [maxRounds, setMaxRounds] = useState(3)
  const [jobInProgress, setJobInProgress] = useState(false)
  const [jobStatus, setJobStatus] = useState(null)
  const [error, setError] = useState(null)
  const closeStreamRef = useRef(null)
  const pollRetryCount = useRef(0)

  useEffect(() => {
    fetchAvailableModels(setAvailableModels, setError, setLoading)
  }, [])

  useEffect(() => () => {
    if (closeStreamRef.current) closeStreamRef.current()
  }, [])

  const estimatedCost = useMemo(() => {
    if (selectedModels.length === 0) return 0
    const questionTokens = Math.ceil((question?.text?.length || 0) / 4)
    const avgResponseTokens = 500
    return selectedModels.reduce((sum, modelId) => {
      const model = availableModels.find(m => m.id === modelId)
      if (!model?.pricing) return sum
      const inputCost = (questionTokens / 1000000) * model.pricing.prompt
      const outputCost = (avgResponseTokens / 1000000) * model.pricing.completion
      return sum + (inputCost + outputCost) * maxRounds
    }, 0)
  }, [availableModels, maxRounds, question?.text, selectedModels])

  function handleModelToggle(modelId) {
    toggleSelection(modelId, setSelectedModels)
  }

  function handleProviderFilter(provider) {
    // Toggle filter: if clicking the same provider, clear filter
    setSelectedProvider(prev => prev === provider ? null : provider)
  }

  async function handleStartJob() {
    if (selectedModels.length === 0) return

    setJobInProgress(true)
    setError(null)
    pollRetryCount.current = 0  // Reset backoff counter
    try {
      const response = await fetchAPI(
        `/experiments/${experimentId}/automate/deliberate`,
        {
          method: 'POST',
          body: JSON.stringify({
            question_id: questionId,
            models: selectedModels,
            max_rounds: maxRounds
          })
        }
      )

      setJobStatus(response)

      // Stream job status (fallback to polling if SSE fails)
      if (closeStreamRef.current) closeStreamRef.current()
      closeStreamRef.current = subscribeToJobStatus({
        experimentId,
        jobId: response.job_id,
        onStatus: (status) => {
          setJobStatus(status)
          if (status.status === 'completed') {
            if (closeStreamRef.current) closeStreamRef.current()
            closeStreamRef.current = null
            setJobInProgress(false)
            pollRetryCount.current = 0  // Reset on completion
            setTimeout(() => onJobStarted(response.job_id, 'deliberation', true), 2000)
            return
          }

          if (status.status === 'failed') {
            if (closeStreamRef.current) closeStreamRef.current()
            closeStreamRef.current = null
            pollRetryCount.current = 0  // Reset on failure
          }
        },
        onPartialResults: undefined,
        onError: () => {
          pollJobStatus(response.job_id)
        },
      })
    } catch (err) {
      setError(`Failed to start deliberation: ${err.message}`)
      setJobInProgress(false)
    }
  }

  async function pollJobStatus(jobId) {
    try {
      const status = await fetchAPI(`/experiments/${experimentId}/automation/status/${jobId}`)
      setJobStatus(status)

      if (status.status === 'completed') {
        setJobInProgress(false)
        pollRetryCount.current = 0  // Reset on completion
        // Wait 2 seconds before auto-closing to show completion state
        setTimeout(() => {
          onJobStarted(jobId, 'deliberation', true)
        }, 2000)
      } else if (status.status === 'failed') {
        pollRetryCount.current = 0  // Reset on failure
        // Keep the error state visible; don't auto-refresh.
        stopJobStream(closeStreamRef)
      } else {
        // Exponential backoff: 500ms → 750ms → 1125ms → max 3s
        pollRetryCount.current += 1
        const delay = Math.min(500 * Math.pow(1.5, pollRetryCount.current), 3000)
        setTimeout(() => pollJobStatus(jobId), delay)
      }
    } catch (err) {
      setError(`Failed to check status: ${err.message}`)
      setJobInProgress(false)
      pollRetryCount.current = 0  // Reset on error
    }
  }

  const providers = useMemo(() => getProviderIds(availableModels), [availableModels])

  const filteredModels = useMemo(() => {
    const models = selectedProvider
      ? availableModels.filter(m => m.id.startsWith(`${selectedProvider}/`))
      : availableModels
    return sortModelsWithTopPinned(models)
  }, [availableModels, selectedProvider])

  const selectedPlural = selectedModels.length === 1 ? '' : 's'
  const roundsPlural = maxRounds === 1 ? '' : 's'
  const deliberationStatus = getDeliberationStatus(jobStatus, maxRounds)

  return (
    <div className="mb-4 p-4 bg-gradient-to-br from-white to-parchment/30 rounded-lg border border-sepia/20 shadow-sm animate-slide-up">
      {jobInProgress ? (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            {deliberationStatus.icon}
            <div className="flex-1">
              <div className="text-sm font-semibold text-ink">
                {deliberationStatus.text}
              </div>
              {jobStatus?.message && (
                <div className="text-xs text-slate">{jobStatus.message}</div>
              )}
              {jobStatus?.progress?.converged && (
                <div className="text-xs text-sage mt-1">
                  Models reached consensus after Round {jobStatus.progress.round - 1}
                </div>
              )}
            </div>
          </div>

          {jobStatus?.progress && !jobStatus.progress.converged && (
            <div className="bg-white rounded-lg p-3">
              <div className="flex items-center justify-between text-sm mb-2">
                <span className="text-slate">
                  Round {jobStatus.progress.round || 1} of {maxRounds}
                </span>
                <span className="font-mono text-ink">
                  {jobStatus.progress.completed ?? 0}/{jobStatus.progress.total ?? 0}
                </span>
              </div>
              <div className="w-full bg-sepia/10 rounded-full h-2">
                <div
                  className="bg-sepia rounded-full h-2 transition-all duration-300"
                  style={{
                    width: `${Math.min(((jobStatus.progress.completed ?? 0) / Math.max(jobStatus.progress.total ?? 1, 1)) * 100, 100)}%`
                  }}
                />
              </div>
            </div>
          )}

          {jobStatus?.errors && jobStatus.errors.length > 0 && (
            <div className="bg-rust/5 rounded-lg p-3 border border-rust/20">
              <div className="text-sm font-semibold text-rust mb-1">Errors:</div>
              <ul className="text-xs text-rust space-y-1">
                {jobStatus.errors.map(err => {
                  const errorKey = `${err.model || 'unknown'}-${err.message || 'error'}-${err.round || 'n/a'}`
                  return (
                    <li key={errorKey}>
                      {err.model}: {err.message}
                      {err.round && ` (Round ${err.round})`}
                    </li>
                  )
                })}
              </ul>
            </div>
          )}
        </div>
      ) : (
        <>
          <div className="mb-4">
            <h4 className="text-sm font-semibold text-ink mb-2">Select Models for Deliberation</h4>
            {loading ? (
              <div className="flex items-center gap-2 text-slate">
                <Loader className="w-4 h-4 animate-spin" />
                <span className="text-sm">Loading models...</span>
              </div>
            ) : (
              <>
                <div className="flex flex-wrap gap-2 mb-3">
                  {providers.map(provider => (
                    <button
                      key={provider}
                      onClick={() => handleProviderFilter(provider)}
                      className={`text-xs px-2 py-1 rounded transition-colors ${selectedProvider === provider
                        ? 'bg-sepia text-white'
                        : 'bg-sepia/10 hover:bg-sepia/20 text-sepia'
                        }`}
                    >
                      {provider}
                    </button>
                  ))}
                </div>

                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {filteredModels.map(model => (
                    <label
                      key={model.id}
                      className="flex items-center gap-2 p-2 hover:bg-sepia/5 rounded cursor-pointer transition-colors"
                    >
                      <input
                        type="checkbox"
                        checked={selectedModels.includes(model.id)}
                        onChange={() => handleModelToggle(model.id)}
                        className="rounded border-sepia/30"
                      />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium text-ink truncate">{model.name}</span>
                          <span className="text-xs text-slate font-mono">{model.id}</span>
                        </div>
                        {model.pricing && (
                          <div className="text-xs text-slate">
                            ${(model.pricing.prompt * 1000).toFixed(3)}/1K in ·
                            ${(model.pricing.completion * 1000).toFixed(3)}/1K out
                          </div>
                        )}
                      </div>
                    </label>
                  ))}
                </div>
              </>
            )}
          </div>

          <div className="mb-4">
            <label htmlFor="max-rounds" className="block text-sm font-semibold text-ink mb-2">
              Maximum Rounds
            </label>
            <select
              id="max-rounds"
              value={maxRounds}
              onChange={(e) => setMaxRounds(Number.parseInt(e.target.value, 10))}
              className="w-full px-3 py-2 text-sm border border-sepia/20 rounded-lg focus:outline-none focus:ring-2 focus:ring-sepia/30"
            >
              <option value={1}>1 Round (Initial responses only)</option>
              <option value={2}>2 Rounds</option>
              <option value={3}>3 Rounds (Recommended)</option>
              <option value={4}>4 Rounds</option>
              <option value={5}>5 Rounds (Advanced - Higher cost)</option>
            </select>
            <p className="text-xs text-slate mt-1">
              Models will refine their responses after seeing others' answers each round.
            </p>
            {maxRounds > 3 && (
              <div className="mt-2 p-2 bg-amber-50 border border-amber-200 rounded text-xs text-amber-900">
                <strong>⚠️ Cost Warning:</strong> {maxRounds} rounds will significantly increase API costs. Ensure your OpenRouter balance is sufficient.
              </div>
            )}
          </div>

          {selectedModels.length > 0 && (
            <div className="bg-sepia/5 rounded p-3 mb-3 border border-sepia/10">
              <div className="text-xs text-slate mb-1">
                Selected: {selectedModels.length} model{selectedPlural} × {maxRounds} round{roundsPlural} = <strong>{selectedModels.length * maxRounds} API calls</strong>
              </div>
              <div className="text-sm font-semibold text-ink">
                Estimated cost: ${estimatedCost.toFixed(4)}
              </div>
              {estimatedCost > 0.5 && (
                <div className="text-xs text-rust mt-1">
                  ⚠️ High cost warning: This operation will make {selectedModels.length * maxRounds} API calls and may consume significant OpenRouter credits.
                </div>
              )}
              {selectedModels.length * maxRounds > 15 && (
                <div className="text-xs text-rust mt-1 font-semibold">
                  🚨 Consider starting with fewer models or rounds to test first.
                </div>
              )}
            </div>
          )}

          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="btn btn-secondary flex-1 text-sm py-1.5"
            >
              Cancel
            </button>
            <button
              onClick={handleStartJob}
              disabled={selectedModels.length === 0 || loading}
              className="btn btn-primary flex-1 text-sm py-1.5"
            >
              Start Deliberation
            </button>
          </div>

          {error && (
            <div className="bg-rust/5 rounded-lg p-3 border border-rust/20 mt-3">
              <p className="text-sm text-rust">{error}</p>
            </div>
          )}
        </>
      )}
    </div>
  )
}
// === TabbedRoundView Component ===
export function TabbedRoundView({ roundGroups }) {
  // Default to latest round
  const latestRound = roundGroups.length > 0 ? roundGroups[roundGroups.length - 1].round : 1
  const [activeRound, setActiveRound] = useState(latestRound)

  const roundColors = {
    1: { bg: 'bg-blue-50', border: 'border-blue-400', text: 'text-blue-700', tab: 'bg-blue-100' },
    2: { bg: 'bg-purple-50', border: 'border-purple-400', text: 'text-purple-700', tab: 'bg-purple-100' },
    3: { bg: 'bg-amber-50', border: 'border-amber-400', text: 'text-amber-700', tab: 'bg-amber-100' },
    4: { bg: 'bg-emerald-50', border: 'border-emerald-400', text: 'text-emerald-700', tab: 'bg-emerald-100' },
    5: { bg: 'bg-rose-50', border: 'border-rose-400', text: 'text-rose-700', tab: 'bg-rose-100' }
  }

  const getRoundColor = (roundNum) => {
    if (roundColors[roundNum]) return roundColors[roundNum]
    // Cycle through colors for rounds > 5
    const colorArray = Object.values(roundColors)
    return colorArray[(roundNum - 1) % colorArray.length]
  }

  const getRoundLabel = (roundNum) => {
    const labels = {
      1: "Initial Responses",
      2: "After Seeing Others",
      3: "Further Refinement",
      4: "Continued Refinement",
      5: "Final Refinement"
    }
    return labels[roundNum] || "Refinement"
  }

  const activeGroup = roundGroups.find(g => g.round === activeRound)

  return (
    <div className="border border-sepia/10 rounded-lg overflow-hidden bg-white shadow-sm">
      {/* Horizontal Tab Bar */}
      <div className="flex border-b border-sepia/10 bg-parchment/20 overflow-x-auto scrollbar-thin">
        {roundGroups.map(group => {
          const colors = getRoundColor(group.round)
          const isActive = group.round === activeRound
          const isLatest = group.round === latestRound

          return (
            <button
              key={group.round}
              onClick={() => setActiveRound(group.round)}
              className={`flex-shrink-0 px-6 py-3 border-b-3 transition-all duration-200 ${isActive
                ? `${colors.border} ${colors.text} font-semibold ${colors.tab}`
                : 'border-transparent text-slate hover:text-sepia hover:bg-sepia/5'
                }`}
              style={{ borderBottomWidth: isActive ? '3px' : '0' }}
            >
              <div className="flex flex-col items-start gap-1 min-w-[140px]">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-semibold">Round {group.round}</span>
                  {isLatest && (
                    <span className="px-2 py-0.5 bg-sage text-white text-xs rounded-full font-medium">
                      Latest
                    </span>
                  )}
                </div>
                <div className={`text-xs ${isActive ? 'opacity-90' : 'opacity-60'}`}>
                  {getRoundLabel(group.round)}
                </div>
                <div className={`text-xs ${isActive ? 'opacity-70' : 'opacity-50'}`}>
                  {group.responses.length} response{group.responses.length === 1 ? '' : 's'}
                </div>
              </div>
            </button>
          )
        })}
      </div>

      {/* Active Round Content */}
      {activeGroup && (
        <div className={`p-4 ${getRoundColor(activeRound).bg} animate-fade-in`}>
          <div className="space-y-3">
            {activeGroup.responses.map((resp, i) => (
              <ResponseCard key={resp.id} response={resp} index={i} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// === RoundSection ===

export function RoundSection({ round, responses, isLatest }) {
  const [isExpanded, setIsExpanded] = useState(true) // Always expanded by default

  const roundLabels = {
    1: "Round 1: Initial Responses",
    2: "Round 2: After Seeing Others",
    3: "Round 3: Further Refinement",
    4: "Round 4: Continued Refinement",
    5: "Round 5: Final Refinement"
  }

  const roundColors = {
    1: "border-blue-300",
    2: "border-purple-300",
    3: "border-amber-300",
    4: "border-emerald-300",
    5: "border-rose-300"
  }

  // Generate labels dynamically for rounds > 5
  const getRoundLabel = (roundNum) => {
    if (roundLabels[roundNum]) {
      return roundLabels[roundNum]
    }
    return roundNum === 1 ? "Round 1: Initial Responses" : `Round ${roundNum}: Refinement`
  }

  // Generate colors dynamically for rounds > 5, cycling through existing colors
  const getRoundColor = (roundNum) => {
    if (roundColors[roundNum]) {
      return roundColors[roundNum]
    }
    // Cycle through colors for rounds > 5
    const colorArray = Object.values(roundColors)
    const index = (roundNum - 1) % colorArray.length
    return colorArray[index]
  }

  return (
    <div className={`border-l-4 ${getRoundColor(round)} pl-4 pb-4`}>
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-2 mb-3 text-sm font-semibold text-ink hover:text-sepia transition-colors w-full"
      >
        {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
        <span className="flex items-center gap-2">
          {getRoundLabel(round)}
          <span className="text-xs font-normal text-slate">({responses.length} {responses.length === 1 ? 'response' : 'responses'})</span>
        </span>
        {isLatest && (
          <span className="ml-auto px-2.5 py-0.5 bg-gradient-to-r from-sage to-sage/80 text-white text-xs rounded-full font-medium">
            Latest Round
          </span>
        )}
      </button>

      {isExpanded && (
        <div className="space-y-3 animate-fade-in">
          {responses.map((resp, i) => (
            <ResponseCard key={resp.id} response={resp} index={i} />
          ))}
        </div>
      )}
    </div>
  )
}

// === AgreementMatrixHeatmap Component ===
export function AgreementMatrixHeatmap({ experimentId, questionId, onClose }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function loadMatrix() {
      try {
        const result = await fetchAPI(`/experiments/${experimentId}/questions/${questionId}/agreement`)
        setData(result)
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    loadMatrix()
  }, [experimentId, questionId])

  if (loading) {
    return (
      <div className="fixed inset-0 bg-ink/30 backdrop-blur-sm flex items-center justify-center z-50">
        <Loader className="w-8 h-8 animate-spin text-sepia" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="fixed inset-0 bg-ink/30 backdrop-blur-sm flex items-center justify-center z-50">
        <div className="card p-6 rounded-xl max-w-md">
          <div className="flex items-center gap-2 text-rust mb-3">
            <AlertCircle className="w-5 h-5" />
            <span className="font-semibold">Error</span>
          </div>
          <p className="text-sm text-slate mb-4">{error}</p>
          <button onClick={onClose} className="btn btn-primary w-full">Close</button>
        </div>
      </div>
    )
  }

  const getHeatmapColor = (value) => {
    // Gradient from red (0) to yellow (0.5) to green (1)
    if (value >= 0.5) {
      const t = (value - 0.5) * 2  // 0 to 1
      const r = Math.round(255 * (1 - t))
      const g = 200
      return `rgb(${r}, ${g}, 100)`
    } else {
      const t = value * 2  // 0 to 1
      const r = 255
      const g = Math.round(200 * t)
      return `rgb(${r}, ${g}, 100)`
    }
  }

  return (
    <div className="fixed inset-0 bg-ink/30 backdrop-blur-sm flex items-center justify-center z-50 animate-fade-in p-4">
      <div className="card rounded-2xl w-full max-w-5xl animate-slide-up max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-sepia/10 flex-shrink-0">
          <h2 className="font-display text-2xl font-semibold text-ink mb-3">
            Ranking Agreement (Judges)
          </h2>
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <span className="text-sm text-slate">Diversity Score:</span>
              <span className="text-lg font-mono font-semibold text-sepia">
                {(data.diversity_score * 100).toFixed(1)}%
              </span>
            </div>
            <span className="text-xs text-slate">
              (Higher diversity = more varied rankings)
            </span>
          </div>
        </div>

        {/* Matrix Content */}
        <div className="flex-1 overflow-auto p-6">
          <div className="overflow-x-auto">
            <table className="border-collapse">
              <thead>
                <tr>
                  <th className="p-2"></th>
                  {data.judges.map((judge) => (
                    <th key={judge} className="p-2">
                      <div
                        className="text-xs font-semibold text-slate transform -rotate-45 origin-bottom-left whitespace-nowrap"
                        style={{ height: '120px', width: '120px' }}
                      >
                        <div title={judge}>{judge}</div>
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.judges.map((judge1, i) => (
                  <tr key={judge1}>
                    <td className="p-2 text-xs font-semibold text-slate max-w-[240px] truncate" title={judge1}>
                      {judge1}
                    </td>
                    {data.judges.map((judge2, j) => (
                      <td key={`${judge1}-${judge2}`} className="p-0">
                        <div
                          className="w-20 h-20 flex items-center justify-center text-xs font-mono font-semibold border border-white/50 cursor-help transition-transform hover:scale-105"
                          style={{ backgroundColor: getHeatmapColor(data.matrix[i][j]) }}
                          title={`${judge1} vs ${judge2}: ${(data.matrix[i][j] * 100).toFixed(1)}% agreement`}
                        >
                          {(data.matrix[i][j] * 100).toFixed(0)}%
                        </div>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Legend and Explanation */}
          <div className="mt-6 grid grid-cols-2 gap-4">
            <div className="p-4 bg-parchment/30 rounded-lg">
              <h4 className="text-sm font-semibold text-ink mb-2">Color Legend</h4>
              <div className="space-y-1 text-xs">
                <div className="flex items-center gap-2">
                  <div className="w-6 h-6 rounded" style={{ backgroundColor: getHeatmapColor(1) }}></div>
                  <span className="text-slate">100% agreement (judges ranked identically)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-6 h-6 rounded" style={{ backgroundColor: getHeatmapColor(0.5) }}></div>
                  <span className="text-slate">50% agreement</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-6 h-6 rounded" style={{ backgroundColor: getHeatmapColor(0) }}></div>
                  <span className="text-slate">0% agreement (completely opposite rankings)</span>
                </div>
              </div>
            </div>

            <div className="p-4 bg-parchment/30 rounded-lg">
              <h4 className="text-sm font-semibold text-ink mb-2">How to Interpret</h4>
              <ul className="text-xs text-slate space-y-1">
                <li>• <strong>Diagonal cells</strong>: Always 100% (judge agrees with themselves)</li>
                <li>• <strong>Green cells</strong>: High agreement between judges</li>
                <li>• <strong>Red cells</strong>: Judges ranked responses very differently</li>
                <li>• <strong>Symmetric</strong>: Agreement between A→B equals B→A</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-sepia/10 flex-shrink-0">
          <button onClick={onClose} className="btn btn-primary w-full">
            Close
          </button>
        </div>
      </div>
    </div>
  )
}

const pricingShape = PropTypes.shape({
  prompt: PropTypes.number,
  completion: PropTypes.number
})

const modelShape = PropTypes.shape({
  id: PropTypes.string.isRequired,
  name: PropTypes.string,
  pricing: pricingShape
})

const responseShape = PropTypes.shape({
  id: PropTypes.string.isRequired,
  content: PropTypes.string.isRequired,
  model: PropTypes.string.isRequired,
  round: PropTypes.number,
  metadata: PropTypes.shape({
    tokens_output: PropTypes.number,
    latency_ms: PropTypes.number,
    cost_usd: PropTypes.number
  })
})

const questionShape = PropTypes.shape({
  text: PropTypes.string.isRequired,
  responses: PropTypes.arrayOf(responseShape)
})

AutomatedResponseForm.propTypes = {
  experimentId: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
  questionId: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
  question: questionShape.isRequired,
  onJobStarted: PropTypes.func.isRequired,
  onClose: PropTypes.func.isRequired
}

AutomatedRankingForm.propTypes = {
  experimentId: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
  questionId: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
  question: questionShape.isRequired,
  onJobStarted: PropTypes.func.isRequired,
  onClose: PropTypes.func.isRequired
}

AutomatedDeliberationForm.propTypes = {
  experimentId: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
  questionId: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
  question: questionShape.isRequired,
  onJobStarted: PropTypes.func.isRequired,
  onClose: PropTypes.func.isRequired
}

ResponseCard.propTypes = {
  response: responseShape.isRequired,
  index: PropTypes.number.isRequired
}

RoundSection.propTypes = {
  round: PropTypes.number.isRequired,
  responses: PropTypes.arrayOf(responseShape).isRequired,
  isLatest: PropTypes.bool
}

RoundSection.defaultProps = {
  isLatest: false
}

AgreementMatrixHeatmap.propTypes = {
  experimentId: PropTypes.string.isRequired,
  questionId: PropTypes.string.isRequired,
  onClose: PropTypes.func.isRequired
}

TabbedRoundView.propTypes = {
  roundGroups: PropTypes.arrayOf(
    PropTypes.shape({
      round: PropTypes.number.isRequired,
      responses: PropTypes.arrayOf(responseShape).isRequired
    })
  ).isRequired
}
