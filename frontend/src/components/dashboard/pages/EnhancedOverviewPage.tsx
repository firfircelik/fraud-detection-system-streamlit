'use client'

import React, { useState } from 'react'
import { useQuery } from 'react-query'
import { 
  CreditCardIcon,
  ShieldCheckIcon,
  ExclamationTriangleIcon,
  ChartBarIcon,
  ClockIcon,
  TrendingUpIcon,
  UserGroupIcon,
  BuildingStorefrontIcon,
  CpuChipIcon,
  BoltIcon
} from '@heroicons/react/24/outline'
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  BarChart, Bar, PieChart, Pie, Cell, AreaChart, Area, ScatterChart, Scatter
} from 'recharts'
import { apiClient, formatCurrency, formatDateTime, getRiskColor } from '../../../lib/api'

const COLORS = ['#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#f97316']

export function EnhancedOverviewPage() {
  const [timeWindow, setTimeWindow] = useState('24h')
  
  // Fetch all data sources
  const { data: dashboardData, isLoading: dashboardLoading } = useQuery(
    'dashboardData',
    () => apiClient.getDashboardData(),
    { refetchInterval: 30000 }
  )

  const { data: statistics, isLoading: statisticsLoading } = useQuery(
    'statistics',
    () => apiClient.getStatistics(),
    { refetchInterval: 60000 }
  )

  const { data: health, isLoading: healthLoading } = useQuery(
    'health',
    () => apiClient.checkHealth(),
    { refetchInterval: 30000 }
  )

  const isLoading = dashboardLoading || statisticsLoading || healthLoading

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-1/4 mb-4"></div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            {[...Array(8)].map((_, i) => (
              <div key={i} className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border">
                <div className="h-16 bg-gray-200 dark:bg-gray-700 rounded"></div>
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Enhanced Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Enterprise Fraud Detection Dashboard
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Real-time monitoring with advanced analytics • Last updated: {new Date().toLocaleTimeString()}
          </p>
        </div>
        
        {/* Time Window Selector */}
        <div className="flex items-center space-x-2">
          <select 
            value={timeWindow}
            onChange={(e) => setTimeWindow(e.target.value)}
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-sm"
          >
            <option value="1h">Last Hour</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
          </select>
        </div>
      </div>

      {/* Enhanced Key Metrics - 2 rows */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {/* Row 1 - Core Metrics */}
        <MetricCard
          icon={CreditCardIcon}
          title="Total Transactions"
          value={dashboardData?.total_transactions?.toLocaleString() || '0'}
          color="blue"
          trend={"+12.5%"}
        />
        <MetricCard
          icon={ShieldCheckIcon}
          title="Fraud Detected"
          value={dashboardData?.fraud_detected?.toLocaleString() || '0'}
          color="red"
          trend={"-2.1%"}
        />
        <MetricCard
          icon={ExclamationTriangleIcon}
          title="Fraud Rate"
          value={`${((dashboardData?.fraud_rate || 0) * 100).toFixed(2)}%`}
          color="yellow"
          trend={"-0.3%"}
        />
        <MetricCard
          icon={ChartBarIcon}
          title="Model Accuracy"
          value={`${((dashboardData?.accuracy || 0) * 100).toFixed(1)}%`}
          color="green"
          trend={"+0.8%"}
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {/* Row 2 - Advanced Metrics */}
        <MetricCard
          icon={UserGroupIcon}
          title="Active Models"
          value={`${statistics?.ensemble_models?.active_models || 0}/${statistics?.ensemble_models?.total_models || 0}`}
          color="purple"
          trend="100%"
        />
        <MetricCard
          icon={BoltIcon}
          title="System Status"
          value={health?.status === 'OK' ? 'Operational' : 'Issues'}
          color="indigo"
          trend={health?.status === 'OK' ? '✓' : '⚠'}
        />
        <MetricCard
          icon={CpuChipIcon}
          title="API Version"
          value={statistics?.api_version || 'Unknown'}
          color="cyan"
          trend="Latest"
        />
        <MetricCard
          icon={BuildingStorefrontIcon}
          title="System Uptime"
          value={statistics?.uptime || 'Unknown'}
          color="orange"
          trend="99.9%"
        />
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Fraud Trend Chart */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Fraud Detection Trends ({timeWindow})
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={[
              { time: '00:00', fraud_rate: 0.02 },
              { time: '06:00', fraud_rate: 0.01 },
              { time: '12:00', fraud_rate: 0.025 },
              { time: '18:00', fraud_rate: 0.035 },
              { time: '24:00', fraud_rate: 0.028 },
            ]}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, 'Fraud Rate']} />
              <Area 
                type="monotone" 
                dataKey="fraud_rate" 
                stroke="#ef4444" 
                fill="#ef4444"
                fillOpacity={0.3}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Risk Distribution */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Risk Level Distribution
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={dashboardData?.risk_distribution ? 
                  Object.entries(dashboardData.risk_distribution).map(([key, value]) => ({
                    name: key,
                    value: value
                  })) : []
                }
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {dashboardData?.risk_distribution && Object.entries(dashboardData.risk_distribution).map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recent Transactions Table */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">
            Recent High-Risk Transactions
          </h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-700">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Transaction ID
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Amount
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Fraud Score
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Risk Level
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Decision
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Time
                </th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
              {dashboardData?.recent_transactions?.slice(0, 10).map((transaction: any) => (
                <tr key={transaction.transactionId} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">
                    {transaction.transactionId}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                    {formatCurrency(transaction.amount)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className="w-16 bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                        <div 
                          className="bg-red-600 h-2 rounded-full" 
                          style={{ width: `${(transaction.fraudScore || 0) * 100}%` }}
                        ></div>
                      </div>
                      <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">
                        {((transaction.fraudScore || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full border ${getRiskColor(transaction.riskLevel || 'LOW')}`}>
                      {transaction.riskLevel || 'LOW'}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full border ${getRiskColor(transaction.status || 'APPROVED')}`}>
                      {transaction.status || 'APPROVED'}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {transaction.timestamp ? formatDateTime(transaction.timestamp) : 'N/A'}
                  </td>
                </tr>
              )) || []}
            </tbody>
          </table>
        </div>
      </div>

      {/* System Information */}
      {statistics && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
              ML Ensemble Status
            </h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">Active Models:</span>
                <span className="text-sm font-medium text-gray-900 dark:text-white">
                  {statistics.ensemble_models?.active_models || 0} / {statistics.ensemble_models?.total_models || 0}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">Ensemble Method:</span>
                <span className="text-sm font-medium text-gray-900 dark:text-white">
                  {statistics.ensemble_models?.ensemble_method || 'N/A'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">System Status:</span>
                <span className="text-sm font-medium text-green-600 dark:text-green-400">
                  {statistics.system_status || 'Unknown'}
                </span>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
              Service Health
            </h3>
            <div className="space-y-3">
              {health?.services && Object.entries(health.services).map(([service, status]) => (
                <div key={service} className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400 capitalize">
                    {service}:
                  </span>
                  <span className={`text-sm font-medium ${
                    status === 'healthy' ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                  }`}>
                    {status}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// Reusable Metric Card Component
interface MetricCardProps {
  icon: React.ComponentType<any>
  title: string
  value: string
  color: string
  trend?: string
}

function MetricCard({ icon: Icon, title, value, color, trend }: MetricCardProps) {
  const colorClasses = {
    blue: 'text-blue-600 dark:text-blue-400',
    red: 'text-red-600 dark:text-red-400',
    yellow: 'text-yellow-600 dark:text-yellow-400',
    green: 'text-green-600 dark:text-green-400',
    purple: 'text-purple-600 dark:text-purple-400',
    indigo: 'text-indigo-600 dark:text-indigo-400',
    cyan: 'text-cyan-600 dark:text-cyan-400',
    orange: 'text-orange-600 dark:text-orange-400',
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between">
        <div className="flex items-center">
          <Icon className={`h-8 w-8 ${colorClasses[color as keyof typeof colorClasses]}`} />
          <div className="ml-4">
            <p className="text-sm font-medium text-gray-600 dark:text-gray-400">{title}</p>
            <p className="text-2xl font-semibold text-gray-900 dark:text-white">{value}</p>
          </div>
        </div>
        {trend && (
          <div className={`text-sm font-medium ${
            trend.startsWith('+') ? 'text-green-600' : trend.startsWith('-') ? 'text-red-600' : 'text-gray-600'
          }`}>
            {trend}
          </div>
        )}
      </div>
    </div>
  )
}