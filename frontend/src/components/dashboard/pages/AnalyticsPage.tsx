'use client'

import { useState, useMemo } from 'react'
import { useQuery } from 'react-query'
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ComposedChart,
  ScatterChart,
  Scatter,
  LabelList
} from 'recharts'
import {
  ChartBarIcon,
  GlobeAltIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  ArrowTrendingUpIcon,
  UserGroupIcon,
  CalendarIcon,
  CurrencyDollarIcon,
  ShieldExclamationIcon,
  EyeIcon,
  AdjustmentsHorizontalIcon,
  ArrowTrendingDownIcon,
  FireIcon,
  BoltIcon,
  CpuChipIcon,
  SignalIcon
} from '@heroicons/react/24/outline'
import { apiClient } from '../../../lib/api'

const RISK_COLORS = {
  LOW: '#10b981',
  MEDIUM: '#f59e0b', 
  HIGH: '#ef4444',
  CRITICAL: '#dc2626'
}

// Advanced Chart Components
const GradientDefs = () => (
  <defs>
    <linearGradient id="fraudGradient" x1="0" y1="0" x2="0" y2="1">
      <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8}/>
      <stop offset="95%" stopColor="#ef4444" stopOpacity={0.1}/>
    </linearGradient>
    <linearGradient id="legitimateGradient" x1="0" y1="0" x2="0" y2="1">
      <stop offset="5%" stopColor="#10b981" stopOpacity={0.8}/>
      <stop offset="95%" stopColor="#10b981" stopOpacity={0.1}/>
    </linearGradient>
    <linearGradient id="volumeGradient" x1="0" y1="0" x2="0" y2="1">
      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.1}/>
    </linearGradient>
  </defs>
)

// Custom Tooltip Components
const CustomTooltip = ({ active, payload, label, formatter }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white dark:bg-gray-800 p-4 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg">
        <p className="text-sm font-medium text-gray-900 dark:text-white mb-2">{label}</p>
        {payload.map((entry: any, index: number) => (
          <div key={index} className="flex items-center space-x-2 mb-1">
            <div 
              className="w-3 h-3 rounded-full" 
              style={{ backgroundColor: entry.color }}
            />
            <span className="text-sm text-gray-600 dark:text-gray-400">{entry.name}:</span>
            <span className="text-sm font-semibold text-gray-900 dark:text-white">
              {formatter ? formatter(entry.value) : entry.value}
            </span>
          </div>
        ))}
      </div>
    )
  }
  return null
}

// Metric Card Component
interface MetricCardProps {
  title: string
  value: string | number
  change?: number
  icon: React.ElementType
  color: string
  subtitle?: string
  trend?: 'up' | 'down' | 'neutral'
}

const MetricCard = ({ title, value, change, icon: Icon, color, subtitle, trend }: MetricCardProps) => {
  const getTrendIcon = () => {
    if (trend === 'up') return ArrowTrendingUpIcon
    if (trend === 'down') return ArrowTrendingDownIcon
    return null
  }
  
  const TrendIcon = getTrendIcon()
  
  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700 hover:shadow-xl transition-shadow">
      <div className="flex items-center justify-between">
        <div className="flex items-center">
          <div className={`p-3 rounded-lg ${color}`}>
            <Icon className="h-6 w-6 text-white" />
          </div>
          <div className="ml-4">
            <p className="text-sm font-medium text-gray-600 dark:text-gray-400">{title}</p>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">{value}</p>
            {subtitle && (
              <p className="text-xs text-gray-500 dark:text-gray-500">{subtitle}</p>
            )}
          </div>
        </div>
        {change !== undefined && TrendIcon && (
          <div className={`flex items-center px-2 py-1 rounded-full text-sm font-medium ${
            trend === 'up' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
            trend === 'down' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' :
            'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
          }`}>
            <TrendIcon className="h-4 w-4 mr-1" />
            {Math.abs(change).toFixed(1)}%
          </div>
        )}
      </div>
    </div>
  )
}

export function AnalyticsPage() {
  const [timeRange, setTimeRange] = useState<'1h' | '24h' | '7d' | '30d'>('24h')
  const [selectedMetric, setSelectedMetric] = useState<'transactions' | 'fraud_rate' | 'amount'>('transactions')

  // Fetch dashboard data for analytics
  const { data: dashboardData, isLoading: dashboardLoading } = useQuery(
    'dashboardData',
    () => apiClient.getDashboardData(),
    {
      refetchInterval: 30000, // Refresh every 30 seconds
    }
  )

  // Fetch statistics
  const { data: statistics, isLoading: statisticsLoading } = useQuery(
    'statistics',
    () => apiClient.getStatistics(),
    {
      refetchInterval: 30000,
    }
  )

  // Fetch ensemble performance
  const { data: ensemblePerformance, isLoading: ensembleLoading } = useQuery(
    'ensemblePerformance',
    () => apiClient.getEnsemblePerformance(),
    {
      refetchInterval: 60000, // Refresh every minute
    }
  )

  // Generate advanced mock data for charts
  const timeSeriesData = useMemo(() => {
    const hours = timeRange === '1h' ? 1 : timeRange === '24h' ? 24 : timeRange === '7d' ? 168 : 720
    const interval = timeRange === '1h' ? 5 : timeRange === '24h' ? 60 : timeRange === '7d' ? 360 : 1440
    
    return Array.from({ length: Math.floor(hours * 60 / interval) }, (_, i) => {
      const time = new Date(Date.now() - (hours * 60 - i * interval) * 60000)
      const baseVolume = 100 + Math.sin(i * 0.5) * 20
      const fraudRate = 2 + Math.random() * 3
      
      return {
        time: timeRange === '1h' || timeRange === '24h' 
          ? time.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' })
          : time.toLocaleDateString('tr-TR', { month: 'short', day: 'numeric' }),
        transactions: Math.floor(baseVolume + Math.random() * 50),
        fraudulent: Math.floor(baseVolume * fraudRate / 100),
        legitimate: Math.floor(baseVolume * (100 - fraudRate) / 100),
        fraud_rate: fraudRate,
        total_amount: (baseVolume * 450 + Math.random() * 200000),
        fraud_amount: (baseVolume * fraudRate * 45),
        response_time: 25 + Math.random() * 25,
        cpu_usage: 40 + Math.random() * 30,
        memory_usage: 50 + Math.random() * 25
      }
    })
  }, [timeRange])

  const riskDistribution = useMemo(() => [
    { name: 'Low Risk', value: 78.5, count: 15700, color: RISK_COLORS.LOW },
    { name: 'Medium Risk', value: 16.2, count: 3240, color: RISK_COLORS.MEDIUM },
    { name: 'High Risk', value: 4.1, count: 820, color: RISK_COLORS.HIGH },
    { name: 'Critical Risk', value: 1.2, count: 240, color: RISK_COLORS.CRITICAL }
  ], [])

  const fraudPatterns = useMemo(() => [
    { pattern: 'Large Amount Transfer', frequency: 23, severity: 'HIGH', detection_rate: 98.5 },
    { pattern: 'Multiple Small Transactions', frequency: 45, severity: 'MEDIUM', detection_rate: 94.2 },
    { pattern: 'Off-hours Activity', frequency: 18, severity: 'HIGH', detection_rate: 96.8 },
    { pattern: 'Geographic Anomaly', frequency: 12, severity: 'CRITICAL', detection_rate: 99.1 },
    { pattern: 'Rapid Fire Transactions', frequency: 8, severity: 'CRITICAL', detection_rate: 97.3 },
    { pattern: 'New Device Login', frequency: 67, severity: 'LOW', detection_rate: 89.6 }
  ], [])

  if (dashboardLoading || statisticsLoading || ensembleLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      {/* Header with Time Range Selector */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Enterprise Analytics Dashboard
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Real-time fraud detection analytics and ML performance insights
          </p>
        </div>
        
        <div className="flex items-center space-x-3 mt-4 sm:mt-0">
          <div className="flex bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
            {(['1h', '24h', '7d', '30d'] as const).map((range) => (
              <button
                key={range}
                onClick={() => setTimeRange(range)}
                className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${
                  timeRange === range
                    ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
                }`}
              >
                {range}
              </button>
            ))}
          </div>
          <div className="flex items-center space-x-1 text-sm text-gray-500 dark:text-gray-400">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span>Live</span>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Total Transactions"
          value="247,586"
          change={12.5}
          trend="up"
          icon={ChartBarIcon}
          color="bg-blue-500"
          subtitle="Last 24 hours"
        />
        <MetricCard
          title="Fraud Detection Rate"
          value="96.8%"
          change={2.1}
          trend="up"
          icon={ShieldExclamationIcon}
          color="bg-green-500"
          subtitle="Average accuracy"
        />
        <MetricCard
          title="Flagged Transactions"
          value="4,872"
          change={-8.3}
          trend="down"
          icon={ExclamationTriangleIcon}
          color="bg-red-500"
          subtitle="Requiring review"
        />
        <MetricCard
          title="Response Time"
          value="23ms"
          change={-15.2}
          trend="down"
          icon={BoltIcon}
          color="bg-purple-500"
          subtitle="Average latency"
        />
      </div>

      {/* Main Analytics Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Transaction Volume Trends */}
        <div className="lg:col-span-2 bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Real-time Transaction Analysis
            </h3>
            <div className="flex space-x-2">
              {(['transactions', 'fraud_rate', 'amount'] as const).map((metric) => (
                <button
                  key={metric}
                  onClick={() => setSelectedMetric(metric)}
                  className={`px-3 py-1 text-xs font-medium rounded-full transition-colors ${
                    selectedMetric === metric
                      ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                      : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400'
                  }`}
                >
                  {metric.replace('_', ' ').toUpperCase()}
                </button>
              ))}
            </div>
          </div>
          <ResponsiveContainer width="100%" height={350}>
            <ComposedChart data={timeSeriesData}>
              <GradientDefs />
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
              <XAxis 
                dataKey="time" 
                stroke="#6b7280" 
                fontSize={12}
                tickLine={false}
              />
              <YAxis stroke="#6b7280" fontSize={12} tickLine={false} />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              
              {selectedMetric === 'transactions' && (
                <>
                  <Area
                    type="monotone"
                    dataKey="legitimate"
                    stackId="1"
                    stroke="#10b981"
                    fill="url(#legitimateGradient)"
                    name="Legitimate"
                  />
                  <Area
                    type="monotone"
                    dataKey="fraudulent"
                    stackId="1"
                    stroke="#ef4444"
                    fill="url(#fraudGradient)"
                    name="Fraudulent"
                  />
                </>
              )}
              
              {selectedMetric === 'fraud_rate' && (
                <Line
                  type="monotone"
                  dataKey="fraud_rate"
                  stroke="#f59e0b"
                  strokeWidth={3}
                  dot={{ fill: '#f59e0b', strokeWidth: 2, r: 4 }}
                  name="Fraud Rate (%)"
                />
              )}
              
              {selectedMetric === 'amount' && (
                <>
                  <Bar
                    dataKey="total_amount"
                    fill="url(#volumeGradient)"
                    name="Total Amount ($)"
                    radius={[2, 2, 0, 0]}
                  />
                  <Line
                    type="monotone"
                    dataKey="fraud_amount"
                    stroke="#ef4444"
                    strokeWidth={2}
                    name="Fraud Amount ($)"
                  />
                </>
              )}
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        {/* Risk Distribution */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
            Risk Distribution
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={riskDistribution}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={120}
                paddingAngle={2}
                dataKey="value"
              >
                {riskDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
                <LabelList 
                  dataKey="value" 
                  position="outside"
                  formatter={(value: number) => `${value}%`}
                  className="text-xs font-medium"
                />
              </Pie>
              <Tooltip 
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload
                    return (
                      <div className="bg-white dark:bg-gray-800 p-3 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg">
                        <p className="font-medium text-gray-900 dark:text-white">{data.name}</p>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {data.value}% ({data.count.toLocaleString()} transactions)
                        </p>
                      </div>
                    )
                  }
                  return null
                }}
              />
            </PieChart>
          </ResponsiveContainer>
          
          <div className="mt-4 space-y-2">
            {riskDistribution.map((item) => (
              <div key={item.name} className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <div 
                    className="w-3 h-3 rounded-full" 
                    style={{ backgroundColor: item.color }}
                  />
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    {item.name}
                  </span>
                </div>
                <span className="text-sm font-medium text-gray-900 dark:text-white">
                  {item.count.toLocaleString()}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ML Model Performance */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
            ML Ensemble Performance
          </h3>
          <ResponsiveContainer width="100%" height={350}>
            <RadarChart data={[
              { metric: 'Accuracy', RF: 96.2, LR: 93.8, IF: 91.5, SVM: 94.1 },
              { metric: 'Precision', RF: 94.8, LR: 91.2, IF: 88.9, SVM: 92.3 },
              { metric: 'Recall', RF: 97.1, LR: 96.4, IF: 94.7, SVM: 95.8 },
              { metric: 'F1-Score', RF: 95.9, LR: 93.7, IF: 91.7, SVM: 94.0 },
              { metric: 'AUC', RF: 98.3, LR: 95.8, IF: 93.2, SVM: 96.1 }
            ]}>
              <PolarGrid stroke="#374151" />
              <PolarAngleAxis dataKey="metric" className="text-xs" />
              <PolarRadiusAxis 
                angle={90} 
                domain={[85, 100]} 
                className="text-xs" 
                tickCount={4}
              />
              <Radar
                name="Random Forest"
                dataKey="RF"
                stroke="#3b82f6"
                fill="#3b82f6"
                fillOpacity={0.1}
                strokeWidth={2}
              />
              <Radar
                name="Logistic Regression"
                dataKey="LR"
                stroke="#8b5cf6"
                fill="#8b5cf6"
                fillOpacity={0.1}
                strokeWidth={2}
              />
              <Radar
                name="Isolation Forest"
                dataKey="IF"
                stroke="#10b981"
                fill="#10b981"
                fillOpacity={0.1}
                strokeWidth={2}
              />
              <Radar
                name="SVM"
                dataKey="SVM"
                stroke="#f59e0b"
                fill="#f59e0b"
                fillOpacity={0.1}
                strokeWidth={2}
              />
              <Legend />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Fraud Patterns Analysis */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
            Advanced Fraud Pattern Detection
          </h3>
          <div className="space-y-4">
            {fraudPatterns.map((pattern, index) => (
              <div key={index} className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors">
                <div className="flex-1">
                  <div className="flex items-center space-x-3">
                    <h4 className="font-medium text-gray-900 dark:text-white">
                      {pattern.pattern}
                    </h4>
                    <span className={`px-2 py-1 text-xs font-semibold rounded-full ${
                      pattern.severity === 'CRITICAL' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' :
                      pattern.severity === 'HIGH' ? 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200' :
                      pattern.severity === 'MEDIUM' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' :
                      'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                    }`}>
                      {pattern.severity}
                    </span>
                  </div>
                  <div className="mt-2 flex items-center space-x-4 text-sm text-gray-600 dark:text-gray-400">
                    <span>Frequency: {pattern.frequency}</span>
                    <span>Detection: {pattern.detection_rate}%</span>
                  </div>
                </div>
                <div className="w-16 h-2 bg-gray-200 dark:bg-gray-600 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-blue-500 rounded-full transition-all duration-300"
                    style={{ width: `${pattern.detection_rate}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Real-time Monitoring */}
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
          Real-time System Health Monitoring
        </h3>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={timeSeriesData.slice(-20)}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
            <XAxis 
              dataKey="time" 
              stroke="#6b7280" 
              fontSize={12}
              tickLine={false}
            />
            <YAxis stroke="#6b7280" fontSize={12} tickLine={false} />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Line
              type="monotone"
              dataKey="response_time"
              stroke="#3b82f6"
              strokeWidth={2}
              name="Response Time (ms)"
              dot={{ fill: '#3b82f6', strokeWidth: 2, r: 3 }}
            />
            <Line
              type="monotone"
              dataKey="cpu_usage"
              stroke="#10b981"
              strokeWidth={2}
              name="CPU Usage (%)"
              dot={{ fill: '#10b981', strokeWidth: 2, r: 3 }}
            />
            <Line
              type="monotone"
              dataKey="memory_usage"
              stroke="#f59e0b"
              strokeWidth={2}
              name="Memory Usage (%)"
              dot={{ fill: '#f59e0b', strokeWidth: 2, r: 3 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
