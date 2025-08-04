'use client'

import React, { useState } from 'react'
import { useQuery } from 'react-query'
import { 
  UserGroupIcon,
  ShieldExclamationIcon,
  MapIcon,
  ChartBarIcon,
  MagnifyingGlassIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline'
import { 
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line
} from 'recharts'

const API_BASE_URL = 'http://localhost:8080'

export function GraphAnalyticsPage() {
  const [selectedEntity, setSelectedEntity] = useState<any>(null)

  // API calls with React Query
  const { data: graphData, isLoading: graphLoading } = useQuery({
    queryKey: ['graph-analytics'],
    queryFn: () => fetch(`${API_BASE_URL}/api/graph/analytics`).then(res => res.json()),
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  const { data: fraudRings, isLoading: ringsLoading } = useQuery({
    queryKey: ['fraud-rings'],
    queryFn: () => fetch(`${API_BASE_URL}/api/graph/fraud-rings`).then(res => res.json()),
    refetchInterval: 60000, // Refresh every minute
  })

  const { data: analyticsData, isLoading: analyticsLoading } = useQuery({
    queryKey: ['analytics-trends'],
    queryFn: () => fetch(`${API_BASE_URL}/api/analytics/trends`).then(res => res.json()),
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  const getRiskColor = (riskLevel: string | number) => {
    if (typeof riskLevel === 'number') {
      if (riskLevel >= 0.8) return '#EF4444' // red
      if (riskLevel >= 0.6) return '#F97316' // orange
      if (riskLevel >= 0.4) return '#EAB308' // yellow
      return '#22C55E' // green
    }
    
    switch (riskLevel?.toLowerCase()) {
      case 'critical': return '#EF4444'
      case 'high': return '#F97316'
      case 'medium': return '#EAB308'
      case 'low': return '#22C55E'
      default: return '#6B7280'
    }
  }

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount)
  }

  if (graphLoading || analyticsLoading) {
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
            <UserGroupIcon className="h-8 w-8 text-purple-600" />
            Graph Analytics
          </h1>
          <p className="text-gray-600 mt-1">Fraud ring detection and network analysis</p>
        </div>
      </div>

      {/* Network Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Network Nodes</p>
              <p className="text-2xl font-bold text-gray-900">
                {graphData?.network_stats?.total_nodes?.toLocaleString() || '0'}
              </p>
            </div>
            <UserGroupIcon className="h-8 w-8 text-blue-600" />
          </div>
          <div className="mt-2 text-xs text-gray-500">
            Users, merchants, devices
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Suspicious Clusters</p>
              <p className="text-2xl font-bold text-red-600">
                {graphData?.network_stats?.suspicious_clusters || '0'}
              </p>
            </div>
            <ShieldExclamationIcon className="h-8 w-8 text-red-600" />
          </div>
          <div className="mt-2 text-xs text-gray-500">
            Detected fraud rings
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Relationships</p>
              <p className="text-2xl font-bold text-green-600">
                {graphData?.network_stats?.total_relationships?.toLocaleString() || '0'}
              </p>
            </div>
            <ChartBarIcon className="h-8 w-8 text-green-600" />
          </div>
          <div className="mt-2 text-xs text-gray-500">
            Transaction connections
          </div>
        </div>
      </div>

      {/* Fraud Rings Detection */}
      <div className="bg-white rounded-lg shadow-sm border">
        <div className="p-6 border-b">
          <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
            <ExclamationTriangleIcon className="h-5 w-5 text-red-600" />
            Detected Fraud Rings
          </h2>
          <p className="text-sm text-gray-600 mt-1">Suspicious entity clusters and fraud networks</p>
        </div>
        <div className="p-6">
          {ringsLoading ? (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
            </div>
          ) : fraudRings?.fraud_rings?.length > 0 ? (
            <div className="space-y-4">
              {fraudRings.fraud_rings.map((ring: any, index: number) => (
                <div key={ring.ring_id || index} className="border rounded-lg p-4 hover:bg-gray-50">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                        ring.risk_level === 'HIGH' ? 'bg-red-100 text-red-800' : 'bg-orange-100 text-orange-800'
                      }`}>
                        {ring.risk_level || 'HIGH'} RISK
                      </div>
                      <span className="font-medium">Ring #{ring.ring_id || index + 1}</span>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-medium">
                        {ring.size || ring.members?.length || 0} Members
                      </p>
                      <p className="text-xs text-gray-500">
                        {formatCurrency(ring.total_amount || 0)}
                      </p>
                    </div>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                    <div>
                      <span className="text-gray-600">Detection Date:</span>
                      <p className="font-medium">
                        {ring.detection_date ? new Date(ring.detection_date).toLocaleDateString() : 'Today'}
                      </p>
                    </div>
                    <div>
                      <span className="text-gray-600">Status:</span>
                      <p className="font-medium">{ring.status || 'ACTIVE'}</p>
                    </div>
                    <div>
                      <span className="text-gray-600">Algorithm:</span>
                      <p className="font-medium">{ring.detection_algorithm || 'Community Detection'}</p>
                    </div>
                  </div>
                  {ring.members && ring.members.length > 0 && (
                    <div className="mt-3 pt-3 border-t">
                      <p className="text-sm text-gray-600 mb-2">Key Members:</p>
                      <div className="flex flex-wrap gap-2">
                        {ring.members.slice(0, 5).map((member: any, idx: number) => (
                          <span key={idx} className="px-2 py-1 bg-gray-100 rounded text-xs">
                            {member.user_id || member.node_id} ({member.role || 'member'})
                          </span>
                        ))}
                        {ring.members.length > 5 && (
                          <span className="px-2 py-1 bg-gray-100 rounded text-xs">
                            +{ring.members.length - 5} more
                          </span>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <UserGroupIcon className="h-12 w-12 mx-auto mb-3 text-gray-300" />
              <p>No fraud rings detected</p>
              <p className="text-sm">Network analysis in progress...</p>
            </div>
          )}
        </div>
      </div>

      {/* Top Risky Entities */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Risky Users */}
        <div className="bg-white rounded-lg shadow-sm border">
          <div className="p-4 border-b">
            <h3 className="font-semibold text-gray-900">Top Risky Users</h3>
          </div>
          <div className="p-4 space-y-3">
            {graphData?.top_risky_entities?.users?.slice(0, 5).map((user: any, index: number) => (
              <div key={user.id || index} className="flex items-center justify-between">
                <div>
                  <p className="font-medium text-sm">{user.id}</p>
                  <p className="text-xs text-gray-500">{user.transaction_count} transactions</p>
                </div>
                <div className="text-right">
                  <div 
                    className="px-2 py-1 rounded text-xs font-medium"
                    style={{ 
                      backgroundColor: getRiskColor(user.risk_score) + '20',
                      color: getRiskColor(user.risk_score)
                    }}
                  >
                    {(user.risk_score * 100).toFixed(0)}%
                  </div>
                </div>
              </div>
            )) || (
              <p className="text-sm text-gray-500 text-center py-4">No risky users detected</p>
            )}
          </div>
        </div>

        {/* Risky Merchants */}
        <div className="bg-white rounded-lg shadow-sm border">
          <div className="p-4 border-b">
            <h3 className="font-semibold text-gray-900">Top Risky Merchants</h3>
          </div>
          <div className="p-4 space-y-3">
            {analyticsData?.merchant_risks?.slice(0, 5).map((merchant: any, index: number) => (
              <div key={merchant.merchant_id || index} className="flex items-center justify-between">
                <div>
                  <p className="font-medium text-sm">{merchant.merchant_name}</p>
                  <p className="text-xs text-gray-500">{merchant.category}</p>
                </div>
                <div className="text-right">
                  <div 
                    className="px-2 py-1 rounded text-xs font-medium"
                    style={{ 
                      backgroundColor: getRiskColor(merchant.fraud_rate) + '20',
                      color: getRiskColor(merchant.fraud_rate)
                    }}
                  >
                    {(merchant.fraud_rate * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            )) || (
              <p className="text-sm text-gray-500 text-center py-4">No risky merchants detected</p>
            )}
          </div>
        </div>

        {/* Risky Devices */}
        <div className="bg-white rounded-lg shadow-sm border">
          <div className="p-4 border-b">
            <h3 className="font-semibold text-gray-900">Top Risky Devices</h3>
          </div>
          <div className="p-4 space-y-3">
            {graphData?.top_risky_entities?.devices?.slice(0, 5).map((device: any, index: number) => (
              <div key={device.id || index} className="flex items-center justify-between">
                <div>
                  <p className="font-medium text-sm">{device.id}</p>
                  <p className="text-xs text-gray-500">{device.user_count} users</p>
                </div>
                <div className="text-right">
                  <div 
                    className="px-2 py-1 rounded text-xs font-medium"
                    style={{ 
                      backgroundColor: getRiskColor(device.risk_score) + '20',
                      color: getRiskColor(device.risk_score)
                    }}
                  >
                    {(device.risk_score * 100).toFixed(0)}%
                  </div>
                </div>
              </div>
            )) || (
              <p className="text-sm text-gray-500 text-center py-4">No risky devices detected</p>
            )}
          </div>
        </div>
      </div>

      {/* Fraud Patterns Analysis */}
      <div className="bg-white rounded-lg shadow-sm border">
        <div className="p-6 border-b">
          <h2 className="text-lg font-semibold text-gray-900">Fraud Pattern Analysis</h2>
          <p className="text-sm text-gray-600 mt-1">Detected patterns and behavioral anomalies</p>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <h4 className="font-medium text-gray-900 mb-3">Peak Fraud Hours</h4>
              <div className="space-y-2">
                {analyticsData?.fraud_patterns?.peak_hours?.map((hour: number) => (
                  <div key={hour} className="flex items-center justify-between text-sm">
                    <span>{hour}:00 - {hour + 1}:00</span>
                    <span className="text-red-600 font-medium">High Risk</span>
                  </div>
                )) || (
                  <p className="text-sm text-gray-500">No patterns detected</p>
                )}
              </div>
            </div>
            
            <div>
              <h4 className="font-medium text-gray-900 mb-3">High Risk Categories</h4>
              <div className="space-y-2">
                {analyticsData?.fraud_patterns?.high_risk_categories?.map((category: string) => (
                  <div key={category} className="flex items-center justify-between text-sm">
                    <span className="capitalize">{category}</span>
                    <span className="text-orange-600 font-medium">Monitor</span>
                  </div>
                )) || (
                  <p className="text-sm text-gray-500">No high-risk categories</p>
                )}
              </div>
            </div>
            
            <div>
              <h4 className="font-medium text-gray-900 mb-3">Velocity Patterns</h4>
              <div className="space-y-2">
                {analyticsData?.fraud_patterns?.velocity_patterns?.map((pattern: string) => (
                  <div key={pattern} className="flex items-center justify-between text-sm">
                    <span className="capitalize">{pattern.replace('_', ' ')}</span>
                    <span className="text-yellow-600 font-medium">Watch</span>
                  </div>
                )) || (
                  <p className="text-sm text-gray-500">No velocity patterns</p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}