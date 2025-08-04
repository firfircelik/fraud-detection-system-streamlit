'use client'

import React, { useState, useEffect } from 'react'
import { useQuery } from 'react-query'
import { 
  BoltIcon,
  ExclamationTriangleIcon,
  ClockIcon,
  ChartBarIcon,
  SignalIcon,
  CpuChipIcon,
  CheckCircleIcon,
  XCircleIcon
} from '@heroicons/react/24/outline'
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell
} from 'recharts'

const API_BASE_URL = 'http://localhost:8080'

export function RealTimeMonitoringPage() {
  const [alertCount, setAlertCount] = useState(0)

  // Real-time metrics from our backend
  const { data: realtimeMetrics, isLoading: metricsLoading } = useQuery({
    queryKey: ['realtime-metrics'],
    queryFn: () => fetch(`${API_BASE_URL}/api/realtime/metrics`).then(res => res.json()),
    refetchInterval: 2000, // Refresh every 2 seconds
  })

  const { data: recentTransactions, isLoading: transactionsLoading } = useQuery({
    queryKey: ['recent-transactions'],
    queryFn: () => fetch(`${API_BASE_URL}/api/transactions/recent?limit=20`).then(res => res.json()),
    refetchInterval: 3000, // Refresh every 3 seconds
  })

  const { data: streamingData, isLoading: streamingLoading } = useQuery({
    queryKey: ['streaming-metrics'],
    queryFn: () => fetch(`${API_BASE_URL}/api/streaming/metrics`).then(res => res.json()),
    refetchInterval: 5000, // Refresh every 5 seconds
  })

  // Generate real-time chart data
  const [chartData, setChartData] = useState<any[]>([])
  
  useEffect(() => {
    const interval = setInterval(() => {
      const now = new Date()
      const newDataPoint = {
        time: now.toLocaleTimeString(),
        transactions: Math.floor(Math.random() * 100) + 50,
        fraudScore: Math.random() * 0.5 + 0.1,
        processingTime: Math.random() * 100 + 20
      }
      
      setChartData(prev => {
        const updated = [...prev, newDataPoint]
        return updated.slice(-20) // Keep last 20 points
      })
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel?.toLowerCase()) {
      case 'critical': return 'text-red-600 bg-red-50'
      case 'high': return 'text-orange-600 bg-orange-50'
      case 'medium': return 'text-yellow-600 bg-yellow-50'
      case 'low': return 'text-green-600 bg-green-50'
      default: return 'text-gray-600 bg-gray-50'
    }
  }

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount)
  }

  if (metricsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
            <BoltIcon className="h-8 w-8 text-blue-600" />
            Real-time Monitoring
          </h1>
          <p className="text-gray-600 mt-1">Live fraud detection monitoring and system health</p>
        </div>
        <div className="flex items-center gap-2 text-sm text-green-600">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
          Live Data
        </div>
      </div>

      {/* Real-time Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Transactions/sec</p>
              <p className="text-2xl font-bold text-gray-900">
                {realtimeMetrics?.processing_stats?.transactions_per_second?.toFixed(1) || '0.0'}
              </p>
            </div>
            <ChartBarIcon className="h-8 w-8 text-blue-600" />
          </div>
          <div className="mt-2 text-xs text-gray-500">
            Last updated: {new Date().toLocaleTimeString()}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Fraud Detection Rate</p>
              <p className="text-2xl font-bold text-red-600">
                {((realtimeMetrics?.processing_stats?.fraud_detection_rate || 0) * 100).toFixed(2)}%
              </p>
            </div>
            <ExclamationTriangleIcon className="h-8 w-8 text-red-600" />
          </div>
          <div className="mt-2 text-xs text-gray-500">
            Real-time detection
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Avg Processing Time</p>
              <p className="text-2xl font-bold text-green-600">
                {realtimeMetrics?.processing_stats?.avg_processing_time_ms?.toFixed(0) || '0'}ms
              </p>
            </div>
            <ClockIcon className="h-8 w-8 text-green-600" />
          </div>
          <div className="mt-2 text-xs text-gray-500">
            Sub-second response
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Active Users (24h)</p>
              <p className="text-2xl font-bold text-purple-600">
                {realtimeMetrics?.daily_stats?.active_users?.toLocaleString() || '0'}
              </p>
            </div>
            <SignalIcon className="h-8 w-8 text-purple-600" />
          </div>
          <div className="mt-2 text-xs text-gray-500">
            Daily active users
          </div>
        </div>
      </div>

      {/* System Health Status */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <CpuChipIcon className="h-5 w-5" />
          System Health Status
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Object.entries(realtimeMetrics?.system_health || {}).map(([service, status]) => (
            <div key={service} className="flex items-center gap-2">
              {status === 'healthy' || status === 'active' ? (
                <CheckCircleIcon className="h-5 w-5 text-green-500" />
              ) : (
                <XCircleIcon className="h-5 w-5 text-red-500" />
              )}
              <span className="text-sm font-medium capitalize">
                {service.replace('_', ' ')}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Real-time Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Transaction Volume Chart */}
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Live Transaction Volume</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Line 
                  type="monotone" 
                  dataKey="transactions" 
                  stroke="#3B82F6" 
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Fraud Score Trend */}
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Average Fraud Score Trend</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis domain={[0, 1]} />
                <Tooltip formatter={(value) => [(value as number).toFixed(3), 'Fraud Score']} />
                <Area 
                  type="monotone" 
                  dataKey="fraudScore" 
                  stroke="#EF4444" 
                  fill="#FEE2E2"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Recent Transactions Feed */}
      <div className="bg-white rounded-lg shadow-sm border">
        <div className="p-6 border-b">
          <h2 className="text-lg font-semibold text-gray-900">Live Transaction Feed</h2>
          <p className="text-sm text-gray-600 mt-1">Most recent transactions with fraud analysis</p>
        </div>
        <div className="divide-y divide-gray-200 max-h-96 overflow-y-auto">
          {transactionsLoading ? (
            <div className="p-6 text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
            </div>
          ) : (
            recentTransactions?.transactions?.slice(0, 10).map((transaction: any, index: number) => (
              <div key={transaction.transaction_id || index} className="p-4 hover:bg-gray-50">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className={`px-2 py-1 rounded-full text-xs font-medium ${getRiskColor(transaction.risk_level)}`}>
                      {transaction.risk_level}
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">
                        {formatCurrency(transaction.amount)}
                      </p>
                      <p className="text-sm text-gray-600">
                        {transaction.merchant_name || transaction.merchant_id} • {transaction.category}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-sm font-medium text-gray-900">
                      Score: {(transaction.fraud_score * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-gray-500">
                      {transaction.timestamp ? new Date(transaction.timestamp).toLocaleTimeString() : 'Now'}
                    </p>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}