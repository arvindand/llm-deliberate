import { useState, useEffect, useMemo } from 'react'
import PropTypes from 'prop-types'
import {
  Scale,
  Plus,
  BarChart3,
  Download,
  Users,
  FileText,
  ChevronRight,
  Trash2,
  Award,
  MessageSquare,
  Sparkles,
  BookOpen,
  Zap,
} from 'lucide-react'
import { AutomatedResponseForm, AutomatedRankingForm, ResponseCard, AutomatedDeliberationForm, MarkdownRenderer, TabbedRoundView, AgreementMatrixHeatmap } from './components'

const API_BASE = '/api'

// === API Helpers ===
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

// === Download Utility ===
function downloadFile(data, filename, type) {
  const blob = new Blob([data], { type })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

async function handleExport(experimentId, questionId, format, type) {
  try {
    let endpoint = `/experiments/${experimentId}/export?format=${format}`
    if (questionId) {
      endpoint = `/experiments/${experimentId}/questions/${questionId}/export?format=${format}`
    }

    const response = await fetch(`${API_BASE}${endpoint}`)
    if (!response.ok) throw new Error('Export failed')

    const data = format === 'csv'
      ? await response.text()
      : JSON.stringify(await response.json(), null, 2)

    const filename = `${type}_${experimentId}${questionId ? '_' + questionId : ''}.${format}`
    const mimeType = format === 'csv' ? 'text/csv' : 'application/json'

    downloadFile(data, filename, mimeType)
  } catch (err) {
    console.error('Export failed:', err)
    alert('Export failed. Please try again.')
  }
}

// === Cost Calculation Utilities ===
function calculateCosts(experiment) {
  const costs = {
    total: 0,
    byQuestion: {},
    byModel: {},
    byRound: {},
    byProvider: {},
    totalResponses: 0
  }

  if (!experiment?.questions) return costs

  for (const question of experiment.questions) {
    costs.byQuestion[question.id] = {
      questionText: question.text,
      totalCost: 0,
      responseCount: 0
    }

    for (const response of question.responses || []) {
      const cost = response.metadata?.cost_usd || 0
      const model = response.model
      const round = response.round || 1
      const provider = model.split('/')[0] || 'unknown'

      costs.total += cost
      costs.totalResponses += 1
      costs.byQuestion[question.id].totalCost += cost
      costs.byQuestion[question.id].responseCount += 1

      costs.byModel[model] = (costs.byModel[model] || 0) + cost
      costs.byRound[round] = (costs.byRound[round] || 0) + cost
      costs.byProvider[provider] = (costs.byProvider[provider] || 0) + cost
    }
  }

  return costs
}

const USD_COST_FORMATTER = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  minimumFractionDigits: 2,
  maximumFractionDigits: 6,
})

function formatCost(usd) {
  const value = Number(usd)
  if (!Number.isFinite(value) || value <= 0) return '$0.00'
  return USD_COST_FORMATTER.format(value)
}

// === Components ===

function Header({ onHome }) {
  return (
    <header className="border-b border-sepia/10 bg-white/50 backdrop-blur-sm sticky top-0 z-50">
      <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
        <button
          type="button"
          onClick={onHome}
          className="flex items-center gap-3 text-left bg-transparent border-none cursor-pointer"
          aria-label="Go to experiments"
        >
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-sepia to-rust flex items-center justify-center">
            <Scale className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="font-display text-xl font-semibold text-ink tracking-tight">
              Deliberate
            </h1>
            <p className="text-xs text-slate -mt-0.5">LLM Council Deliberation Lab</p>
          </div>
        </button>
        <nav className="flex items-center gap-6 text-sm">
          <button
            type="button"
            onClick={onHome}
            className="text-slate hover:text-sepia transition-colors bg-transparent border-none cursor-pointer"
          >
            Experiments
          </button>
          <a
            href="https://github.com/arvindand/llm-deliberate"
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-secondary text-sm py-1.5"
          >
            GitHub
          </a>
        </nav>
      </div>
    </header>
  )
}

Header.propTypes = {
  onHome: PropTypes.func.isRequired,
}

function EmptyState({ onCreateExperiment }) {
  return (
    <div className="max-w-2xl mx-auto text-center py-20 animate-fade-in">
      <div className="w-20 h-20 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-parchment to-cream border border-sepia/10 flex items-center justify-center">
        <BookOpen className="w-10 h-10 text-sepia/60" />
      </div>
      <h2 className="font-display text-2xl font-semibold text-ink mb-3">
        Begin Your Research
      </h2>
      <p className="text-slate mb-8 max-w-md mx-auto">
        Explore how different LLMs deliberate, rank each other's responses,
        and discover which aggregation methods produce the best consensus.
      </p>
      <button onClick={onCreateExperiment} className="btn btn-primary inline-flex items-center gap-2">
        <Plus className="w-4 h-4" />
        Create First Experiment
      </button>

      <div className="mt-16 grid grid-cols-3 gap-6 text-left">
        {[
          { id: 'multi-model', icon: Users, title: 'Multi-Model', desc: 'Collect responses from GPT, Claude, Gemini, and more' },
          { id: 'compare-methods', icon: BarChart3, title: 'Compare Methods', desc: 'Borda, Copeland, Ranked Pairs — see which wins' },
          { id: 'discover-insights', icon: Sparkles, title: 'Discover Insights', desc: 'Find when consensus emerges and when it fails' },
        ].map((item, i) => (
          <div key={item.id} className="card p-5 rounded-xl animate-slide-up" style={{ animationDelay: `${i * 0.1}s` }}>
            <div className="w-10 h-10 rounded-lg bg-sepia/5 flex items-center justify-center mb-3">
              <item.icon className="w-5 h-5 text-sepia" />
            </div>
            <h3 className="font-display font-semibold text-ink mb-1">{item.title}</h3>
            <p className="text-sm text-slate">{item.desc}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

function ExperimentList({ experiments, onSelect, onDelete }) {
  return (
    <div className="space-y-3">
      {experiments.map((exp, i) => (
        <div
          key={exp.id}
          className="card rounded-xl p-4 flex items-center gap-4 hover:border-sepia/30 transition-all animate-slide-up"
          style={{ animationDelay: `${i * 0.05}s` }}
        >
          <button
            type="button"
            className="flex items-center gap-4 flex-1 min-w-0 text-left cursor-pointer"
            onClick={() => onSelect(exp.id)}
          >
            <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-sepia/10 to-rust/10 flex items-center justify-center">
              <FileText className="w-6 h-6 text-sepia" />
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="font-display font-semibold text-ink truncate">{exp.name}</h3>
              <p className="text-sm text-slate">
                {exp.question_count === 1 ? '1 question' : `${exp.question_count} questions`} ·
                Created {new Date(exp.created_at).toLocaleDateString()}
              </p>
            </div>
          </button>
          <button
            type="button"
            onClick={(e) => { e.stopPropagation(); onDelete(exp.id) }}
            className="p-2 text-slate hover:text-rust transition-colors"
          >
            <Trash2 className="w-4 h-4" />
          </button>
          <ChevronRight className="w-5 h-5 text-slate" />
        </div>
      ))}
    </div>
  )
}
function CreateExperimentModal({ onClose, onCreate }) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!name.trim()) return
    onCreate({ name: name.trim(), description: description.trim() || null })
  }

  return (
    <div className="fixed inset-0 bg-ink/30 backdrop-blur-sm flex items-center justify-center z-50 animate-fade-in">
      <div className="card rounded-2xl p-8 w-full max-w-md animate-slide-up">
        <h2 className="font-display text-2xl font-semibold text-ink mb-6">New Experiment</h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="exp-name" className="block text-sm font-medium text-slate mb-1.5">Name</label>
            <input
              id="exp-name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., Math Reasoning Comparison"
              className="w-full"
              autoFocus
            />
          </div>
          <div>
            <label htmlFor="exp-desc" className="block text-sm font-medium text-slate mb-1.5">Description (optional)</label>
            <textarea
              id="exp-desc"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="What are you testing?"
              className="w-full"
              rows={3}
            />
          </div>
          <div className="flex gap-3 pt-2">
            <button type="button" onClick={onClose} className="btn btn-secondary flex-1">
              Cancel
            </button>
            <button type="submit" className="btn btn-primary flex-1" disabled={!name.trim()}>
              Create
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

function groupResponsesByRound(responses) {
  if (!responses || responses.length === 0) return []

  const grouped = responses.reduce((acc, resp) => {
    const round = resp.round || 1
    if (!acc[round]) acc[round] = []
    acc[round].push(resp)
    return acc
  }, {})

  return Object.entries(grouped)
    .map(([round, resps]) => ({ round: Number.parseInt(round, 10), responses: resps }))
    .sort((a, b) => a.round - b.round)
}

function renderResponsesContent(roundGroups, hasMultipleRounds, responses) {
  if (!roundGroups || roundGroups.length === 0) {
    return (
      <p className="text-sm text-slate text-center py-8">
        No responses yet. Add responses manually or use automation.
      </p>
    )
  }

  if (hasMultipleRounds) return <TabbedRoundView roundGroups={roundGroups} />

  return (
    <div className="space-y-3">
      {(responses || []).map((resp, i) => (
        <ResponseCard key={resp.id} response={resp} index={i} />
      ))}
    </div>
  )
}

function QuestionCard({ experimentId, question, onAddResponse, onAddRanking, onCompare, onRefresh, showExperimentActions, onViewCosts, onExportExperiment }) {
  const [showAddResponse, setShowAddResponse] = useState(false)
  const [showAddRanking, setShowAddRanking] = useState(false)
  const [showAutomatedResponse, setShowAutomatedResponse] = useState(false)
  const [showAutomatedRanking, setShowAutomatedRanking] = useState(false)
  const [showDeliberation, setShowDeliberation] = useState(false)
  const [showAgreementMatrix, setShowAgreementMatrix] = useState(false)
  const [newResponse, setNewResponse] = useState({ model: '', content: '' })
  const [newRanking, setNewRanking] = useState({ judge: '', rankings: [], confidence: 1 })

  const models = ['gpt-4o', 'claude-sonnet', 'gemini-pro', 'llama-3', 'mistral', 'deepseek']
  const typeColors = {
    factual: 'bg-blue-50 text-blue-700',
    reasoning: 'bg-purple-50 text-purple-700',
    subjective: 'bg-amber-50 text-amber-700',
    creative: 'bg-emerald-50 text-emerald-700',
  }

  const roundGroups = groupResponsesByRound(question.responses)
  const hasMultipleRounds = roundGroups.length > 1
  const hasResponses = (question.responses?.length || 0) > 0
  const hasRankings = (question.rankings?.length || 0) > 0
  const disableAutoResponses = hasMultipleRounds || hasRankings
  const disableDeliberation = hasResponses || hasRankings

  let autoResponseTitle = 'Automated collection via API'
  if (hasMultipleRounds) {
    autoResponseTitle = 'Disabled for deliberated questions'
  } else if (hasRankings) {
    autoResponseTitle = 'Disabled after rankings exist (adding responses would invalidate rankings)'
  }

  let deliberationTitle = 'Multi-round deliberation'
  if (hasResponses) {
    deliberationTitle = 'Disabled once responses exist (deliberation would generate a new response set)'
  } else if (hasRankings) {
    deliberationTitle = 'Disabled after rankings exist'
  }

  const responsesContent = renderResponsesContent(roundGroups, hasMultipleRounds, question.responses)

  return (
    <div className="card rounded-xl overflow-hidden">
      <div className="p-5 border-b border-sepia/10">
        <div className="flex items-start justify-between gap-3 mb-3">
          <div className="flex items-center gap-2">
            <span className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${typeColors[question.question_type]}`}>
              {question.question_type}
            </span>
            {question.ground_truth && (
              <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-sage/10 text-sage">
                Has ground truth
              </span>
            )}
          </div>
          <div className="flex items-center gap-1">
            {showExperimentActions && (
              <>
                <button
                  type="button"
                  onClick={onViewCosts}
                  className="text-slate hover:text-sepia transition-colors p-1"
                  title="View Costs"
                  aria-label="View experiment costs"
                >
                  <BarChart3 className="w-4 h-4 pointer-events-none" />
                </button>
                <button
                  type="button"
                  onClick={(e) => onExportExperiment(e.shiftKey ? 'csv' : 'json')}
                  className="text-slate hover:text-sepia transition-colors p-1"
                  title="Export Experiment (click: JSON, shift+click: CSV)"
                  aria-label="Export experiment (click JSON, shift click CSV)"
                >
                  <Download className="w-4 h-4 pointer-events-none" />
                </button>
                <span className="mx-1 h-4 w-px bg-sepia/20" aria-hidden="true" />
              </>
            )}

            {question.responses?.length > 0 && (
              <button
                type="button"
                onClick={(e) => handleExport(experimentId, question.id, e.shiftKey ? 'csv' : 'json', 'question')}
                className="text-slate hover:text-sepia transition-colors p-1"
                title="Export Question (click: JSON, shift+click: CSV)"
                aria-label="Export question (click JSON, shift click CSV)"
              >
                <FileText className="w-4 h-4 pointer-events-none" />
              </button>
            )}
          </div>
        </div>
        <p className="font-display text-lg text-ink">{question.text}</p>
      </div>

      {hasMultipleRounds && roundGroups.length > 0 && (
        <div className="px-5 py-4 bg-gradient-to-r from-sage/5 to-transparent border-b border-sepia/10">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-6 h-6 rounded-full bg-sage/10 flex items-center justify-center">
              <Award className="w-3.5 h-3.5 text-sage" />
            </div>
            <h4 className="text-sm font-semibold text-sage uppercase tracking-wide">
              Convergent Answer (Round {roundGroups.at(-1).round})
            </h4>
          </div>
          <div className="bg-white/50 rounded-lg border border-sage/10 p-4">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs font-medium text-slate">Representative Model:</span>
              <span className={`model-badge ${roundGroups.at(-1).responses[0].model.split('/')[0]}`}>
                {roundGroups.at(-1).responses[0].model}
              </span>
            </div>
            <div className="prose prose-sm max-w-none">
              <MarkdownRenderer content={roundGroups.at(-1).responses[0].content} />
            </div>
            <div className="mt-2 text-xs text-sage font-medium">
              Scroll down to see full deliberation history ↓
            </div>
          </div>
        </div>
      )}

      {/* Responses */}
      <div className="p-5 bg-parchment/30">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-sm font-semibold text-slate uppercase tracking-wide">
            Responses ({question.responses?.length || 0})
          </h4>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => {
                if (disableAutoResponses) return
                setShowAutomatedResponse(!showAutomatedResponse)
              }}
              disabled={disableAutoResponses}
              className={`text-sepia hover:text-rust text-sm font-medium flex items-center gap-1 ${disableAutoResponses ? 'opacity-50 cursor-not-allowed hover:text-sepia' : ''}`}
              title={autoResponseTitle}
            >
              <Zap className="w-4 h-4" /> Auto
            </button>
            {!hasMultipleRounds && (
              <button
                type="button"
                onClick={() => {
                  if (disableDeliberation) return
                  setShowDeliberation(!showDeliberation)
                }}
                disabled={disableDeliberation}
                className={`text-sepia hover:text-rust text-sm font-medium flex items-center gap-1 ${disableDeliberation ? 'opacity-50 cursor-not-allowed hover:text-sepia' : ''}`}
                title={deliberationTitle}
              >
                <Zap className="w-4 h-4" /> Deliberate
              </button>
            )}
            <button
              type="button"
              onClick={() => {
                if (hasRankings) return
                setShowAddResponse(!showAddResponse)
              }}
              disabled={hasRankings}
              className={`text-sepia hover:text-rust text-sm font-medium flex items-center gap-1 ${hasRankings ? 'opacity-50 cursor-not-allowed hover:text-sepia' : ''}`}
              title={hasRankings ? 'Disabled after rankings exist (adding responses would invalidate rankings)' : undefined}
            >
              <Plus className="w-4 h-4" /> Add
            </button>
          </div>
        </div>

        {showAutomatedResponse && (
          <AutomatedResponseForm
            experimentId={experimentId}
            questionId={question.id}
            question={question}
            onJobStarted={(jobId, type, completed) => {
              if (completed) {
                setShowAutomatedResponse(false)
                onRefresh()
              }
            }}
            onClose={() => setShowAutomatedResponse(false)}
          />
        )}

        {showDeliberation && (
          <AutomatedDeliberationForm
            experimentId={experimentId}
            questionId={question.id}
            question={question}
            onJobStarted={(jobId, type, completed) => {
              if (completed) {
                setShowDeliberation(false)
                onRefresh()
              }
            }}
            onClose={() => setShowDeliberation(false)}
          />
        )}

        {showAddResponse && (
          <div className="mb-4 p-4 bg-white rounded-lg border border-sepia/10 animate-slide-up">
            <div className="grid grid-cols-2 gap-3 mb-3">
              <select
                value={newResponse.model}
                onChange={(e) => setNewResponse({ ...newResponse, model: e.target.value })}
                className="text-sm"
              >
                <option value="">Select model...</option>
                {models.map(m => <option key={m} value={m}>{m}</option>)}
              </select>
            </div>
            <textarea
              value={newResponse.content}
              onChange={(e) => setNewResponse({ ...newResponse, content: e.target.value })}
              placeholder="Paste the model's response..."
              className="w-full text-sm mb-3"
              rows={4}
            />
            <button
              onClick={() => {
                if (hasRankings) return
                if (newResponse.model && newResponse.content) {
                  onAddResponse(question.id, newResponse)
                  setNewResponse({ model: '', content: '' })
                  setShowAddResponse(false)
                }
              }}
              className="btn btn-primary text-sm py-1.5"
              disabled={hasRankings || !newResponse.model || !newResponse.content}
              title={hasRankings ? 'Disabled after rankings exist (adding responses would invalidate rankings)' : undefined}
            >
              Save Response
            </button>
          </div>
        )}

        <div className="space-y-4">
          {responsesContent}
        </div>
      </div>

      {/* Rankings */}
      <div className="p-5 border-t border-sepia/10">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-sm font-semibold text-slate uppercase tracking-wide">
            Rankings ({question.rankings?.length || 0})
          </h4>
          <div className="flex gap-2">
            {question.responses?.length > 0 && (
              <button
                onClick={() => setShowAutomatedRanking(!showAutomatedRanking)}
                className="text-sepia hover:text-rust text-sm font-medium flex items-center gap-1"
                title="Automated collection via API"
              >
                <Zap className="w-4 h-4" /> Auto
              </button>
            )}
            <button
              onClick={() => setShowAddRanking(!showAddRanking)}
              className="text-sepia hover:text-rust text-sm font-medium flex items-center gap-1"
              disabled={!question.responses?.length}
            >
              <Plus className="w-4 h-4" /> Add
            </button>
          </div>
        </div>

        {showAutomatedRanking && question.responses?.length > 0 && (
          <AutomatedRankingForm
            experimentId={experimentId}
            questionId={question.id}
            question={question}
            onJobStarted={(jobId, type, completed) => {
              if (completed) {
                setShowAutomatedRanking(false)
                onRefresh()
              }
            }}
            onClose={() => setShowAutomatedRanking(false)}
          />
        )}

        {showAddRanking && question.responses?.length > 0 && (
          <div className="mb-4 p-4 bg-parchment/50 rounded-lg border border-sepia/10 animate-slide-up">
            <div className="mb-3">
              <label htmlFor="ranking-judge-model" className="block text-xs font-medium text-slate mb-1">Judge Model</label>
              <select
                id="ranking-judge-model"
                value={newRanking.judge}
                onChange={(e) => setNewRanking({ ...newRanking, judge: e.target.value })}
                className="w-full text-sm"
              >
                <option value="">Select judge...</option>
                {models.map(m => <option key={m} value={m}>{m}</option>)}
              </select>
            </div>
            <div className="mb-3">
              <label htmlFor="ranking-ids" className="block text-xs font-medium text-slate mb-1">
                Ranking (drag to reorder, or enter response IDs)
              </label>
              <input
                id="ranking-ids"
                type="text"
                value={newRanking.rankings.join(', ')}
                onChange={(e) => setNewRanking({
                  ...newRanking,
                  rankings: e.target.value.split(',').map(s => s.trim()).filter(Boolean)
                })}
                placeholder="e.g., abc123, def456, ghi789"
                className="w-full text-sm font-mono"
              />
              <p className="text-xs text-slate mt-1">
                Available IDs: {question.responses.map(r => r.id).join(', ')}
              </p>
            </div>
            <div className="mb-3">
              <label htmlFor="ranking-confidence" className="block text-xs font-medium text-slate mb-1">
                Confidence (0-1)
              </label>
              <input
                id="ranking-confidence"
                type="number"
                min="0"
                max="1"
                step="0.1"
                value={newRanking.confidence}
                onChange={(e) => setNewRanking({ ...newRanking, confidence: Number.parseFloat(e.target.value) })}
                className="w-24 text-sm"
              />
            </div>
            <button
              onClick={() => {
                if (newRanking.judge && newRanking.rankings.length > 0) {
                  onAddRanking(question.id, newRanking)
                  setNewRanking({ judge: '', rankings: [], confidence: 1 })
                  setShowAddRanking(false)
                }
              }}
              className="btn btn-primary text-sm py-1.5"
              disabled={!newRanking.judge || newRanking.rankings.length === 0}
            >
              Save Ranking
            </button>
          </div>
        )}

        {question.rankings?.length > 0 && (
          <div className="space-y-3 mb-4">
            {question.rankings.map((rank) => {
              const rankedResponses = rank.rankings.map((id) => {
                const response = question.responses?.find(r => r.id === id)
                if (!response) return { id, label: id, model: 'Unknown' }
                return { id, label: id, model: response.model }
              })

              return (
                <div key={rank.id} className="bg-white rounded-lg border border-sepia/20 shadow-sm p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold text-slate">Judge:</span>
                      <span className={`model-badge ${rank.judge.split('/')[0]}`}>
                        {rank.judge}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 text-xs">
                      <span className="text-slate">Confidence:</span>
                      <span className="font-semibold text-ink">
                        {(rank.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <div className="text-xs font-semibold text-slate uppercase tracking-wide">
                      Ranking (Best → Worst)
                    </div>
                    <div className="flex items-center gap-2 flex-wrap">
                      {rankedResponses.map((resp, idx) => (
                        <div key={resp.id} className="flex items-center gap-2">
                          <div className="flex items-center gap-1.5 bg-sepia/5 border border-sepia/20 rounded-lg px-3 py-2">
                            <span className="text-xs font-bold text-sepia">
                              #{idx + 1}
                            </span>
                            <span className="text-xs font-semibold text-sepia bg-sepia/10 px-2 py-0.5 rounded">
                              {resp.label}
                            </span>
                            <span className={`model-badge ${resp.model.split('/')[0]} text-xs`}>
                              {resp.model.split('/')[1] || resp.model}
                            </span>
                          </div>
                          {idx < rankedResponses.length - 1 && (
                            <span className="text-slate text-xs">→</span>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>

                  {rank.reasoning && (
                    <div className="mt-3 pt-3 border-t border-sepia/10">
                      <div className="text-xs font-semibold text-slate mb-1">Reasoning:</div>
                      <p className="text-xs text-ink leading-relaxed">{rank.reasoning}</p>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}

        {question.rankings?.length > 0 && (
          <div className="space-y-2">
            <button
              onClick={() => onCompare(question.id)}
              className="btn btn-secondary w-full flex items-center justify-center gap-2"
            >
              <BarChart3 className="w-4 h-4" />
              Compare Aggregation Methods
            </button>
            <button
              type="button"
              onClick={() => {
                if ((question.rankings?.length || 0) < 2) return
                setShowAgreementMatrix(true)
              }}
              disabled={(question.rankings?.length || 0) < 2}
              className={`btn btn-secondary w-full flex items-center justify-center gap-2 ${(question.rankings?.length || 0) < 2 ? 'opacity-50 cursor-not-allowed' : ''}`}
              title={(question.rankings?.length || 0) < 2 ? 'Need at least 2 rankings (two judges) to compute agreement' : 'View judge agreement matrix'}
            >
              <Users className="w-4 h-4" />
              View Ranking Agreement (Judges)
            </button>
          </div>
        )}

        {showAgreementMatrix && (
          <AgreementMatrixHeatmap
            experimentId={experimentId}
            questionId={question.id}
            onClose={() => setShowAgreementMatrix(false)}
          />
        )}
      </div>
    </div>
  )
}

function ComparisonResults({ results, onClose }) {
  if (!results) return null

  const methodNames = {
    plurality: 'Plurality',
    borda: 'Borda Count',
    weighted_borda: 'Weighted Borda',
    copeland: 'Copeland',
    ranked_pairs: 'Ranked Pairs'
  }

  const maxScore = Math.max(
    ...Object.values(results.methods).flatMap(m => Object.values(m.scores))
  )

  return (
    <div className="fixed inset-0 bg-ink/30 backdrop-blur-sm flex items-center justify-center z-50 animate-fade-in p-4">
      <div className="card rounded-2xl w-full max-w-3xl animate-slide-up max-h-[90vh] flex flex-col">
        <div className="p-6 border-b border-sepia/10 flex-shrink-0">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="font-display text-2xl font-semibold text-ink">Method Comparison</h2>
              <p className="text-sm text-slate mt-1">{results.question_text}</p>
            </div>
            {results.unanimous && (
              <span className="px-3 py-1 rounded-full bg-sage/10 text-sage text-sm font-medium">
                ✓ Unanimous
              </span>
            )}
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-6">
          <div className="space-y-6">
            {Object.entries(results.methods).map(([method, data]) => (
              <div key={method} className="p-4 bg-parchment/30 rounded-xl">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-display font-semibold text-ink">{methodNames[method]}</h3>
                  <div className="flex items-center gap-2">
                    <Award className="w-4 h-4 text-sepia" />
                    <span className="font-medium text-sepia">{data.winner}</span>
                  </div>
                </div>
                <div className="space-y-2">
                  {Object.entries(data.scores)
                    .sort((a, b) => b[1] - a[1])
                    .map(([model, score], i) => (
                      <div key={model} className="flex items-center gap-3">
                        {(() => {
                          const classes = ['gold', 'silver', 'bronze']
                          const rankClass = classes[i] || 'bronze'
                          return (
                            <span className={`rank-badge ${rankClass}`}>
                              {i + 1}
                            </span>
                          )
                        })()}
                        <span className="w-28 text-sm font-medium truncate">{model}</span>
                        <div className="flex-1 score-bar">
                          <div
                            className="score-bar-fill"
                            style={{ width: `${(score / maxScore) * 100}%` }}
                          />
                        </div>
                        <span className="w-12 text-right text-sm font-mono text-slate">
                          {score.toFixed(1)}
                        </span>
                      </div>
                    ))}
                </div>
              </div>
            ))}

            {results.ground_truth && (
              <div className="p-4 bg-sage/5 rounded-xl border border-sage/20">
                <h4 className="text-sm font-semibold text-sage mb-1">Ground Truth</h4>
                <p className="text-sm text-ink">{results.ground_truth}</p>
              </div>
            )}
          </div>
        </div>

        <div className="p-6 border-t border-sepia/10 flex-shrink-0">
          <button onClick={onClose} className="btn btn-primary w-full">
            Close
          </button>
        </div>
      </div>
    </div>
  )
}

ComparisonResults.propTypes = {
  results: PropTypes.object,
  onClose: PropTypes.func.isRequired
}

function CostDashboard({ experiment, onClose }) {
  const costs = useMemo(() => calculateCosts(experiment), [experiment])
  const [activeTab, setActiveTab] = useState('overview')

  return (
    <div className="fixed inset-0 bg-ink/30 backdrop-blur-sm flex items-center justify-center z-50 animate-fade-in p-4">
      <div className="card rounded-2xl w-full max-w-5xl animate-slide-up max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-sepia/10 flex-shrink-0">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="font-display text-2xl font-semibold text-ink">Cost Analysis</h2>
              <p className="text-sm text-slate mt-1">{experiment.name}</p>
            </div>
            <div className="text-right">
              <div className="text-xs text-slate mb-1">Total Experiment Cost</div>
              <div className="text-4xl font-display font-bold text-sepia">
                {formatCost(costs.total)}
              </div>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="border-b border-sepia/10 flex-shrink-0 overflow-x-auto">
          <div className="flex gap-1 px-6 min-w-max">
            {[
              { id: 'overview', label: 'Overview' },
              { id: 'byQuestion', label: 'By Question' },
              { id: 'byModel', label: 'By Model' },
              { id: 'byRound', label: 'By Round' },
              { id: 'byProvider', label: 'By Provider' }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-3 text-sm font-medium transition-colors border-b-2 whitespace-nowrap ${activeTab === tab.id
                  ? 'border-sepia text-sepia'
                  : 'border-transparent text-slate hover:text-sepia'
                  }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Tab Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {activeTab === 'overview' && (
            <div className="grid grid-cols-2 gap-4">
              <div className="card p-5 bg-gradient-to-br from-white to-parchment/30">
                <div className="text-xs text-slate mb-2 uppercase tracking-wide">Total Questions</div>
                <div className="text-3xl font-display font-semibold text-ink">
                  {experiment.questions?.length || 0}
                </div>
              </div>
              <div className="card p-5 bg-gradient-to-br from-white to-parchment/30">
                <div className="text-xs text-slate mb-2 uppercase tracking-wide">Total Responses</div>
                <div className="text-3xl font-display font-semibold text-ink">
                  {costs.totalResponses}
                </div>
              </div>
              <div className="card p-5 bg-gradient-to-br from-white to-sepia/5">
                <div className="text-xs text-slate mb-2 uppercase tracking-wide">Avg Cost per Response</div>
                <div className="text-3xl font-display font-semibold text-sepia">
                  {formatCost(costs.total / Math.max(1, costs.totalResponses))}
                </div>
              </div>
              <div className="card p-5 bg-gradient-to-br from-white to-parchment/30">
                <div className="text-xs text-slate mb-2 uppercase tracking-wide">Total Models Used</div>
                <div className="text-3xl font-display font-semibold text-ink">
                  {Object.keys(costs.byModel).length}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'byQuestion' && (
            <div className="space-y-2">
              {Object.entries(costs.byQuestion)
                .sort((a, b) => b[1].totalCost - a[1].totalCost)
                .map(([qid, data]) => (
                  <div key={qid} className="flex items-center gap-3 p-3 hover:bg-parchment/30 rounded-lg transition-colors">
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-ink line-clamp-1" title={data.questionText}>
                        {data.questionText}
                      </div>
                      <div className="text-xs text-slate">
                        {data.responseCount} response{data.responseCount === 1 ? '' : 's'}
                      </div>
                    </div>
                    <div className="flex items-center gap-3 flex-shrink-0">
                      <div className="text-right">
                        <div className="text-base font-mono font-semibold text-sepia">
                          {formatCost(data.totalCost)}
                        </div>
                        <div className="text-xs text-slate">
                          {((data.totalCost / Math.max(costs.total, 0.000001)) * 100).toFixed(0)}%
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
            </div>
          )}

          {activeTab === 'byModel' && (
            <div className="space-y-2">
              {Object.entries(costs.byModel)
                .sort((a, b) => b[1] - a[1])
                .map(([model, cost]) => (
                  <div key={model} className="flex items-center gap-3 p-2 hover:bg-parchment/30 rounded transition-colors">
                    <span className={`model-badge ${model.split('/')[0]} text-xs`}>{model}</span>
                    <div className="flex-1 h-2 bg-sepia/10 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-sepia transition-all duration-300"
                        style={{ width: `${(cost / Math.max(costs.total, 0.000001)) * 100}%` }}
                      />
                    </div>
                    <span className="text-sm font-mono font-semibold text-sepia w-20 text-right">
                      {formatCost(cost)}
                    </span>
                    <span className="text-xs text-slate w-10 text-right">
                      {((cost / Math.max(costs.total, 0.000001)) * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
            </div>
          )}

          {activeTab === 'byRound' && (
            <div className="space-y-2">
              {Object.entries(costs.byRound)
                .sort((a, b) => Number(a[0]) - Number(b[0]))
                .map(([round, cost]) => (
                  <div key={round} className="flex items-center gap-3 p-3 hover:bg-parchment/30 rounded transition-colors">
                    <div className="text-sm font-semibold text-ink w-20">Round {round}</div>
                    <div className="flex-1 h-2 bg-sepia/10 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-sepia transition-all duration-300"
                        style={{ width: `${(cost / Math.max(costs.total, 0.000001)) * 100}%` }}
                      />
                    </div>
                    <div className="text-sm font-mono font-semibold text-sepia w-20 text-right">
                      {formatCost(cost)}
                    </div>
                    <div className="text-xs text-slate w-10 text-right">
                      {((cost / Math.max(costs.total, 0.000001)) * 100).toFixed(0)}%
                    </div>
                  </div>
                ))}
            </div>
          )}

          {activeTab === 'byProvider' && (
            <div className="space-y-2">
              {Object.entries(costs.byProvider)
                .sort((a, b) => b[1] - a[1])
                .map(([provider, cost]) => (
                  <div key={provider} className="flex items-center gap-3 p-3 hover:bg-parchment/30 rounded transition-colors">
                    <div className="text-sm font-medium text-ink w-24 uppercase tracking-wide">
                      {provider}
                    </div>
                    <div className="flex-1 h-2 bg-sepia/10 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-sepia transition-all duration-300"
                        style={{ width: `${(cost / Math.max(costs.total, 0.000001)) * 100}%` }}
                      />
                    </div>
                    <div className="text-sm font-mono font-semibold text-sepia w-20 text-right">
                      {formatCost(cost)}
                    </div>
                    <div className="text-xs text-slate w-10 text-right">
                      {((cost / Math.max(costs.total, 0.000001)) * 100).toFixed(0)}%
                    </div>
                  </div>
                ))}
            </div>
          )}
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

CostDashboard.propTypes = {
  experiment: PropTypes.object.isRequired,
  onClose: PropTypes.func.isRequired
}

function ExperimentView({ experimentId, onBack }) {
  const [experiment, setExperiment] = useState(null)
  const [loading, setLoading] = useState(true)
  const [showAddQuestion, setShowAddQuestion] = useState(false)
  const [newQuestion, setNewQuestion] = useState({ text: '', question_type: 'reasoning', ground_truth: '' })
  const [comparisonResults, setComparisonResults] = useState(null)
  const [showCostDashboard, setShowCostDashboard] = useState(false)

  const handleExportExperiment = (format) => handleExport(experimentId, null, format, 'experiment')

  useEffect(() => {
    loadExperiment()
  }, [experimentId])

  async function loadExperiment() {
    setLoading(true)
    try {
      const data = await fetchAPI(`/experiments/${experimentId}`)
      setExperiment(data)
    } catch (err) {
      console.error('Failed to load experiment:', err)
    }
    setLoading(false)
  }

  async function handleAddQuestion() {
    if (!newQuestion.text) return
    try {
      await fetchAPI(`/experiments/${experimentId}/questions`, {
        method: 'POST',
        body: JSON.stringify({
          text: newQuestion.text,
          question_type: newQuestion.question_type,
          ground_truth: newQuestion.ground_truth || null
        })
      })
      setNewQuestion({ text: '', question_type: 'reasoning', ground_truth: '' })
      setShowAddQuestion(false)
      loadExperiment()
    } catch (err) {
      console.error('Failed to add question:', err)
    }
  }

  async function handleAddResponse(questionId, response) {
    try {
      await fetchAPI(`/experiments/${experimentId}/responses`, {
        method: 'POST',
        body: JSON.stringify({
          question_id: questionId,
          model: response.model,
          content: response.content
        })
      })
      loadExperiment()
    } catch (err) {
      console.error('Failed to add response:', err)
    }
  }

  async function handleAddRanking(questionId, ranking) {
    try {
      await fetchAPI(`/experiments/${experimentId}/rankings`, {
        method: 'POST',
        body: JSON.stringify({
          question_id: questionId,
          judge: ranking.judge,
          rankings: ranking.rankings,
          confidence: ranking.confidence
        })
      })
      loadExperiment()
    } catch (err) {
      console.error('Failed to add ranking:', err)
    }
  }

  async function handleCompare(questionId) {
    try {
      const results = await fetchAPI(`/experiments/${experimentId}/compare?question_id=${questionId}`)
      setComparisonResults(results)
    } catch (err) {
      console.error('Failed to compare methods:', err)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="animate-pulse-subtle text-sepia">Loading...</div>
      </div>
    )
  }

  if (!experiment) {
    return (
      <div className="text-center py-20">
        <p className="text-slate">Experiment not found</p>
        <button onClick={onBack} className="btn btn-secondary mt-4">Go Back</button>
      </div>
    )
  }

  return (
    <div className="animate-fade-in">
      <div className="flex items-start justify-between gap-4 mb-8">
        <div className="min-w-0">
          <div className="flex items-center gap-3 min-w-0">
            <button
              type="button"
              onClick={onBack}
              className="btn btn-secondary text-sm inline-flex items-center whitespace-nowrap"
              title="Back to experiments"
              aria-label="Back to experiments"
            >
              ← Back
            </button>
            <h2 className="font-display text-2xl font-semibold text-ink truncate">{experiment.name}</h2>
          </div>
          {experiment.description && (
            <p className="text-slate mt-2">{experiment.description}</p>
          )}
        </div>

        <button
          type="button"
          onClick={() => setShowAddQuestion(true)}
          className="btn btn-primary text-sm inline-flex items-center gap-2 whitespace-nowrap"
        >
          <Plus className="w-4 h-4" />
          Add Question
        </button>
      </div>

      {showAddQuestion && (
        <div className="card rounded-xl p-6 mb-6 animate-slide-up">
          <h3 className="font-display text-lg font-semibold text-ink mb-4">New Question</h3>
          <div className="space-y-4">
            <div>
              <label htmlFor="new-question-text" className="block text-sm font-medium text-slate mb-1.5">Question Text</label>
              <textarea
                id="new-question-text"
                value={newQuestion.text}
                onChange={(e) => setNewQuestion({ ...newQuestion, text: e.target.value })}
                placeholder="Enter your question..."
                className="w-full"
                rows={3}
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label htmlFor="new-question-type" className="block text-sm font-medium text-slate mb-1.5">Type</label>
                <select
                  id="new-question-type"
                  value={newQuestion.question_type}
                  onChange={(e) => setNewQuestion({ ...newQuestion, question_type: e.target.value })}
                  className="w-full"
                >
                  <option value="factual">Factual</option>
                  <option value="reasoning">Reasoning</option>
                  <option value="subjective">Subjective</option>
                  <option value="creative">Creative</option>
                </select>
              </div>
              <div>
                <label htmlFor="new-ground-truth" className="block text-sm font-medium text-slate mb-1.5">Ground Truth (optional)</label>
                <input
                  id="new-ground-truth"
                  type="text"
                  value={newQuestion.ground_truth}
                  onChange={(e) => setNewQuestion({ ...newQuestion, ground_truth: e.target.value })}
                  placeholder="Correct answer if known"
                  className="w-full"
                />
              </div>
            </div>
            <div className="flex gap-3">
              <button onClick={() => setShowAddQuestion(false)} className="btn btn-secondary">
                Cancel
              </button>
              <button onClick={handleAddQuestion} className="btn btn-primary" disabled={!newQuestion.text}>
                Add Question
              </button>
            </div>
          </div>
        </div>
      )}

      {experiment.questions?.length === 0 ? (
        <div className="text-center py-16 text-slate">
          <MessageSquare className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>No questions yet. Add one to get started!</p>
        </div>
      ) : (
        <div className="space-y-6">
          {experiment.questions.map((q, i) => (
            <QuestionCard
              key={q.id}
              experimentId={experimentId}
              question={q}
              onAddResponse={handleAddResponse}
              onAddRanking={handleAddRanking}
              onCompare={handleCompare}
              onRefresh={loadExperiment}
              showExperimentActions={i === 0}
              onViewCosts={() => setShowCostDashboard(true)}
              onExportExperiment={handleExportExperiment}
            />
          ))}
        </div>
      )}

      {comparisonResults && (
        <ComparisonResults
          results={comparisonResults}
          onClose={() => setComparisonResults(null)}
        />
      )}

      {showCostDashboard && experiment && (
        <CostDashboard
          experiment={experiment}
          onClose={() => setShowCostDashboard(false)}
        />
      )}
    </div>
  )
}

const responseShape = PropTypes.shape({
  id: PropTypes.string.isRequired,
  model: PropTypes.string.isRequired,
  content: PropTypes.string.isRequired,
  round: PropTypes.number
})

const rankingShape = PropTypes.shape({
  id: PropTypes.string.isRequired,
  judge: PropTypes.string.isRequired,
  rankings: PropTypes.arrayOf(PropTypes.string).isRequired,
  confidence: PropTypes.number,
  reasoning: PropTypes.string
})

QuestionCard.propTypes = {
  experimentId: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
  question: PropTypes.object.isRequired,
  onAddResponse: PropTypes.func.isRequired,
  onAddRanking: PropTypes.func.isRequired,
  onCompare: PropTypes.func.isRequired,
  onRefresh: PropTypes.func.isRequired,
  showExperimentActions: PropTypes.bool,
  onViewCosts: PropTypes.func,
  onExportExperiment: PropTypes.func,
}

QuestionCard.defaultProps = {
  showExperimentActions: false,
  onViewCosts: undefined,
  onExportExperiment: undefined,
}

const questionShape = PropTypes.shape({
  id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
  text: PropTypes.string.isRequired,
  question_type: PropTypes.string.isRequired,
  ground_truth: PropTypes.string,
  responses: PropTypes.arrayOf(responseShape),
  rankings: PropTypes.arrayOf(rankingShape)
})

const experimentShape = PropTypes.shape({
  id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
  name: PropTypes.string.isRequired,
  description: PropTypes.string,
  created_at: PropTypes.string,
  question_count: PropTypes.number,
  questions: PropTypes.arrayOf(questionShape)
})

const comparisonResultsShape = PropTypes.shape({
  question_text: PropTypes.string,
  unanimous: PropTypes.bool,
  ground_truth: PropTypes.string,
  methods: PropTypes.objectOf(PropTypes.shape({
    winner: PropTypes.string,
    scores: PropTypes.objectOf(PropTypes.number)
  }))
})

EmptyState.propTypes = {
  onCreateExperiment: PropTypes.func.isRequired
}

ExperimentList.propTypes = {
  experiments: PropTypes.arrayOf(experimentShape).isRequired,
  onSelect: PropTypes.func.isRequired,
  onDelete: PropTypes.func.isRequired
}

CreateExperimentModal.propTypes = {
  onClose: PropTypes.func.isRequired,
  onCreate: PropTypes.func.isRequired
}



ComparisonResults.propTypes = {
  results: comparisonResultsShape,
  onClose: PropTypes.func.isRequired
}

ExperimentView.propTypes = {
  experimentId: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
  onBack: PropTypes.func.isRequired
}

// === Main App ===

export default function App() {
  const [experiments, setExperiments] = useState([])
  const [loading, setLoading] = useState(true)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [selectedExperiment, setSelectedExperiment] = useState(null)

  useEffect(() => {
    loadExperiments()
  }, [])

  async function loadExperiments() {
    setLoading(true)
    try {
      const data = await fetchAPI('/experiments')
      setExperiments(data.experiments || [])
    } catch (err) {
      console.error('Failed to load experiments:', err)
    }
    setLoading(false)
  }

  async function handleCreateExperiment({ name, description }) {
    try {
      const result = await fetchAPI('/experiments', {
        method: 'POST',
        body: JSON.stringify({ name, description })
      })
      setShowCreateModal(false)
      loadExperiments()
      setSelectedExperiment(result.id)
    } catch (err) {
      console.error('Failed to create experiment:', err)
    }
  }

  async function handleDeleteExperiment(id) {
    if (!confirm('Delete this experiment?')) return
    try {
      await fetchAPI(`/experiments/${id}`, { method: 'DELETE' })
      loadExperiments()
    } catch (err) {
      console.error('Failed to delete experiment:', err)
    }
  }

  let mainContent

  if (selectedExperiment) {
    mainContent = (
      <ExperimentView
        experimentId={selectedExperiment}
        onBack={() => {
          setSelectedExperiment(null)
          loadExperiments()
        }}
      />
    )
  } else if (loading) {
    mainContent = (
      <div className="flex items-center justify-center py-20">
        <div className="animate-pulse-subtle text-sepia">Loading experiments...</div>
      </div>
    )
  } else if (experiments.length === 0) {
    mainContent = <EmptyState onCreateExperiment={() => setShowCreateModal(true)} />
  } else {
    mainContent = (
      <div className="animate-fade-in">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="font-display text-2xl font-semibold text-ink">Your Experiments</h2>
            <p className="text-slate mt-1">Select an experiment to continue or create a new one</p>
          </div>
          <button
            onClick={() => setShowCreateModal(true)}
            className="btn btn-primary inline-flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            New Experiment
          </button>
        </div>
        <ExperimentList
          experiments={experiments}
          onSelect={setSelectedExperiment}
          onDelete={handleDeleteExperiment}
        />
      </div>
    )
  }

  return (
    <div className="min-h-screen">
      <Header
        onHome={() => {
          setSelectedExperiment(null)
          setShowCreateModal(false)
          loadExperiments()
        }}
      />

      <main className="max-w-6xl mx-auto px-6 py-12">
        {mainContent}
      </main>

      {showCreateModal && (
        <CreateExperimentModal
          onClose={() => setShowCreateModal(false)}
          onCreate={handleCreateExperiment}
        />
      )}

      <footer className="border-t border-sepia/10 mt-20">
        <div className="max-w-6xl mx-auto px-6 py-8 text-center text-sm text-slate">
          <p className="flourish mb-4">❦</p>
          <p>
            Deliberate — LLM council deliberation and voting research tool.
          </p>
          <p className="mt-2">
            Built for exploring wisdom of crowds in AI systems.
          </p>
        </div>
      </footer>
    </div>
  )
}
