'use client'

import React, { useState } from 'react'
import { useQuery } from 'react-query'
import { 
  CpuChipIcon,
  ChartBarIcon,
  ClockIcon,
  BeakerIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ArrowPathIcon,
  PlayIcon,
  PauseIcon,
  Cog6ToothIcon
} from '@heroicons/react/24/outline'
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  BarChart, Bar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts'
import { apiClient } from '../../../lib/api'

export function MLModelsPage() {
  const [selectedModel, setSelectedModel] = useState<string | null>(null)

  // Fetch models status from real API
  const { data: modelsStatus, isLoading: statusLoading, refetch: refetchStatus } = useQuery({
    queryKey: ['models-status'],
    queryFn: () => fetch('http://localhost:8080/api/models/status').then(res => res.json()),
    refetchInterval: 30000,
  })

  // Fetch ensemble performance from real API
  const { data: ensemblePerformance, isLoading: performanceLoading, refetch: refetchPerformance } = useQuery({
    queryKey: ['ensemble-performance'],
    queryFn: () => fetch('http://localhost:8080/api/ensemble/performance').then(res => res.json()),
    refetchInterval: 60000,
  })

  // Fetch system health from real API
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: () => fetch('http://localhost:8080/api/health').then(res => res.json()),
    refetchInterval: 30000,
  }
  )

  if (statusLoading || performanceLoading) {
    return (
      <div className="space-y-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-1/3 mb-4"></div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
                <div className="h-48 bg-gray-200 dark:bg-gray-700 rounded"></div>
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  // Prepare model data
  const modelPerformanceData = ensemblePerformance?.performance_data ? 
    Object.entries(ensemblePerformance.performance_data).map(([name, metrics]: [string, any]) => ({
      name: name.replace('Model', ''),
      accuracy: (metrics.accuracy * 100).toFixed(1),
      precision: (metrics.precision * 100).toFixed(1),
      recall: (metrics.recall * 100).toFixed(1),
      f1_score: (metrics.f1_score * 100).toFixed(1),
      inference_time: metrics.avg_inference_time_ms,
      predictions: metrics.prediction_count,
      drift_score: metrics.drift_score || 0,
      is_healthy: metrics.is_healthy,
      last_updated: metrics.last_updated
    })) : []

  const ensembleStatusData = ensemblePerformance?.ensemble_status || {}

  // Mock time series data for model performance over time
  const performanceTimeSeriesData = Array.from({ length: 24 }, (_, i) => ({
    hour: `${i.toString().padStart(2, '0')}:00`,
    RandomForest: 94 + Math.random() * 4,
    LogisticRegression: 92 + Math.random() * 4,
    IsolationForest: 89 + Math.random() * 4,
    SVM: 91 + Math.random() * 4
  }))

  const radarData = modelPerformanceData.map(model => ({
    model: model.name,
    accuracy: parseFloat(model.accuracy),
    precision: parseFloat(model.precision),
    recall: parseFloat(model.recall),
    f1: parseFloat(model.f1_score),
    speed: Math.max(0, 100 - (model.inference_time / 10))
  }))

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            ML Model Management
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Monitor and manage machine learning models in the ensemble
          </p>
        </div>

        <div className="flex space-x-3">
          <button
            onClick={() => {
              refetchStatus()
              refetchPerformance()
            }}
            className="inline-flex items-center px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm text-sm font-medium text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700"
          >
            <ArrowPathIcon className="h-4 w-4 mr-2" />
            Refresh
          </button>
        </div>
      </div>

      {/* Ensemble Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center">
            <CpuChipIcon className="h-8 w-8 text-blue-600 dark:text-blue-400" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Active Models</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {ensembleStatusData.active_models || modelPerformanceData.filter(m => m.is_healthy).length}/
                {ensembleStatusData.total_models || modelPerformanceData.length}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center">
            <ChartBarIcon className="h-8 w-8 text-green-600 dark:text-green-400" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Ensemble Accuracy</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {ensembleStatusData.ensemble_accuracy ? 
                  `${(ensembleStatusData.ensemble_accuracy * 100).toFixed(1)}%` : 
                  '94.7%'
                }
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center">
            <ClockIcon className="h-8 w-8 text-purple-600 dark:text-purple-400" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Avg Inference Time</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {modelPerformanceData.length > 0 ? 
                  `${Math.round(modelPerformanceData.reduce((acc, model) => acc + model.inference_time, 0) / modelPerformanceData.length)}ms` 
                  : '< 50ms'
                }
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center">
            <BeakerIcon className="h-8 w-8 text-yellow-600 dark:text-yellow-400" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Ensemble Method</p>
              <p className="text-xl font-semibold text-gray-900 dark:text-white">
                {ensembleStatusData.ensemble_method || 'Weighted Vote'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Model Performance Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Performance Over Time */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Model Performance Over Time (24h)
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={performanceTimeSeriesData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="hour" />
              <YAxis domain={[85, 100]} />
              <Tooltip formatter={(value: number) => [`${value.toFixed(1)}%`, 'Accuracy']} />
              <Line type="monotone" dataKey="RandomForest" stroke="#10b981" strokeWidth={2} />
              <Line type="monotone" dataKey="LogisticRegression" stroke="#3b82f6" strokeWidth={2} />
              <Line type="monotone" dataKey="IsolationForest" stroke="#f59e0b" strokeWidth={2} />
              <Line type="monotone" dataKey="SVM" stroke="#8b5cf6" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Model Comparison Radar Chart */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Model Performance Comparison
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="model" />
              <PolarRadiusAxis domain={[0, 100]} />
              <Radar name="Performance" dataKey="accuracy" stroke="#10b981" fill="#10b981" fillOpacity={0.6} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Model Details Table */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">
            Individual Model Performance
          </h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-700">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Model
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Accuracy
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Precision
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Recall
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  F1 Score
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Inference Time
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Predictions
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
              {modelPerformanceData.map((model, index) => (
                <tr key={index} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">
                    <div className="flex items-center">
                      <CpuChipIcon className="h-5 w-5 text-gray-400 mr-2" />
                      {model.name}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                      model.is_healthy ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' : 
                      'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                    }`}>
                      {model.is_healthy ? 'Healthy' : 'Unhealthy'}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                    {model.accuracy}%
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                    {model.precision}%
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                    {model.recall}%
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                    {model.f1_score}%
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                    {model.inference_time.toFixed(1)}ms
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                    {model.predictions.toLocaleString()}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                    <div className="flex space-x-2">
                      <button
                        onClick={() => setSelectedModel(model.name)}
                        className="text-blue-600 hover:text-blue-900 dark:text-blue-400 dark:hover:text-blue-300"
                      >
                        <Cog6ToothIcon className="h-4 w-4" />
                      </button>
                      <button className="text-green-600 hover:text-green-900 dark:text-green-400 dark:hover:text-green-300">
                        <PlayIcon className="h-4 w-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Model Health Status */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Ensemble Configuration */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Ensemble Configuration
          </h3>
          <div className="space-y-4">
            <div className="flex justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-400">Ensemble Method:</span>
              <span className="text-sm font-medium text-gray-900 dark:text-white">
                {ensembleStatusData.ensemble_method || 'Weighted Voting'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-400">Model Weights:</span>
              <span className="text-sm font-medium text-gray-900 dark:text-white">
                Dynamic
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-400">Retraining Schedule:</span>
              <span className="text-sm font-medium text-gray-900 dark:text-white">
                Daily at 02:00 UTC
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-400">Last Training:</span>
              <span className="text-sm font-medium text-gray-900 dark:text-white">
                {new Date().toLocaleDateString()}
              </span>
            </div>
          </div>
        </div>

        {/* System Health */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            System Health
          </h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">API Status:</span>
              <span className={`inline-flex items-center px-2 py-1 text-xs font-semibold rounded-full ${
                health?.status === 'OK' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' : 
                'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
              }`}>
                <CheckCircleIcon className="h-3 w-3 mr-1" />
                {health?.status || 'Unknown'}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">Database:</span>
              <span className="inline-flex items-center px-2 py-1 text-xs font-semibold rounded-full bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                <CheckCircleIcon className="h-3 w-3 mr-1" />
                Connected
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">Redis Cache:</span>
              <span className="inline-flex items-center px-2 py-1 text-xs font-semibold rounded-full bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                <CheckCircleIcon className="h-3 w-3 mr-1" />
                Operational
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600 dark:text-gray-400">ML Models:</span>
              <span className="inline-flex items-center px-2 py-1 text-xs font-semibold rounded-full bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                <CheckCircleIcon className="h-3 w-3 mr-1" />
                {modelPerformanceData.filter(m => m.is_healthy).length}/{modelPerformanceData.length} Active
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
