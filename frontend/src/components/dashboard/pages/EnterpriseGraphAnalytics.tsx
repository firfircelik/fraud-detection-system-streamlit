'use client'

import React, { useState, useEffect, useRef } from 'react'
import { 
  ShareIcon,
  UserGroupIcon,
  BuildingStorefrontIcon,
  DevicePhoneMobileIcon,
  ExclamationTriangleIcon,
  ShieldExclamationIcon,
  EyeIcon,
  MagnifyingGlassIcon,
  ChartBarIcon,
  CpuChipIcon,
  GlobeAltIcon,
  BoltIcon,
  FireIcon,
  MapPinIcon
} from '@heroicons/react/24/outline'

interface FraudRing {
  ring_id: string
  size: number
  risk_level: string
  total_amount: number
  detection_date: string
  status: string
  members: Array<{
    user_id: string
    role: string
    risk_score: number
  }>
  detection_method: string
}

interface GraphNode {
  id: string
  type: 'user' | 'merchant' | 'device' | 'transaction'
  risk_score: number
  transaction_count?: number
  total_amount?: number
  country?: string
}

interface GraphAnalytics {
  fraud_rings: FraudRing[]
  network_stats: {
    total_nodes: number
    total_relationships: number
    suspicious_clusters: number
  }
  top_risky_entities: {
    users: Array<{
      id: string
      risk_score: number
      transaction_count: number
    }>
    merchants: Array<{
      id: string
      risk_score: number
      transaction_count: number
    }>
    devices: Array<{
      id: string
      risk_score: number
      user_count: number
    }>
  }
}

export function EnterpriseGraphAnalytics() {
  const [graphData, setGraphData] = useState<GraphAnalytics | null>(null)
  const [fraudRings, setFraudRings] = useState<FraudRing[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedRing, setSelectedRing] = useState<FraudRing | null>(null)
  const [networkView, setNetworkView] = useState<'2d' | '3d' | 'force'>('2d')
  const [realTimeEnabled, setRealTimeEnabled] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedEntity, setSelectedEntity] = useState<string | null>(null)
  
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)

  useEffect(() => {
    fetchGraphAnalytics()
    fetchFraudRings()
    
    if (realTimeEnabled) {
      intervalRef.current = setInterval(() => {
        fetchGraphAnalytics()
        fetchFraudRings()
      }, 10000) // Update every 10 seconds
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [realTimeEnabled])

  const fetchGraphAnalytics = async () => {
    try {
      const response = await fetch('http://localhost:8080/api/graph/analytics')
      const data = await response.json()
      setGraphData(data)
      setLoading(false)
    } catch (error) {
      console.error('Failed to fetch graph analytics:', error)
      setLoading(false)
    }
  }

  const fetchFraudRings = async () => {
    try {
      const response = await fetch('http://localhost:8080/api/graph/fraud-rings')
      const data = await response.json()
      setFraudRings(data.fraud_rings || [])
    } catch (error) {
      console.error('Failed to fetch fraud rings:', error)
    }
  }

  const getRiskLevelColor = (riskLevel: string) => {
    switch (riskLevel.toUpperCase()) {
      case 'CRITICAL': return 'bg-red-600 text-white'
      case 'HIGH': return 'bg-red-500 text-white'
      case 'MEDIUM': return 'bg-yellow-500 text-white'
      case 'LOW': return 'bg-green-500 text-white'
      default: return 'bg-gray-500 text-white'
    }
  }

  const getStatusColor = (status: string) => {
    switch (status.toUpperCase()) {
      case 'ACTIVE': return 'bg-red-100 text-red-800 border-red-200'
      case 'INVESTIGATING': return 'bg-yellow-100 text-yellow-800 border-yellow-200'
      case 'RESOLVED': return 'bg-green-100 text-green-800 border-green-200'
      default: return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }

  return (
    <div className="space-y-6">
      {/* Enhanced Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center">
            <ShareIcon className="h-8 w-8 mr-3 text-purple-600" />
            Enterprise Graph Intelligence
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Advanced network analysis with Neo4j-powered fraud ring detection and 3D relationship mapping
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <button
            onClick={() => setRealTimeEnabled(!realTimeEnabled)}
            className={`flex items-center px-4 py-2 rounded-lg font-medium transition-colors ${
              realTimeEnabled 
                ? 'bg-green-100 text-green-800 border border-green-200' 
                : 'bg-gray-100 text-gray-600 border border-gray-200'
            }`}
          >
            <BoltIcon className="h-4 w-4 mr-2" />
            {realTimeEnabled ? 'Live Analysis' : 'Paused'}
          </button>
          
          <div className="flex bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
            {['2d', '3d', 'force'].map((view) => (
              <button
                key={view}
                onClick={() => setNetworkView(view as any)}
                className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                  networkView === view
                    ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
                    : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white'
                }`}
              >
                {view.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Network Statistics Dashboard */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-gradient-to-r from-purple-500 to-purple-600 rounded-lg p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-purple-100">Total Network Nodes</p>
              <p className="text-3xl font-bold">
                {graphData?.network_stats?.total_nodes?.toLocaleString() || '0'}
              </p>
            </div>
            <ShareIcon className="h-12 w-12 text-purple-200" />
          </div>
        </div>

        <div className="bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-blue-100">Relationships</p>
              <p className="text-3xl font-bold">
                {graphData?.network_stats?.total_relationships?.toLocaleString() || '0'}
              </p>
            </div>
            <ChartBarIcon className="h-12 w-12 text-blue-200" />
          </div>
        </div>

        <div className="bg-gradient-to-r from-red-500 to-red-600 rounded-lg p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-red-100">Fraud Rings Detected</p>
              <p className="text-3xl font-bold">{fraudRings.length}</p>
            </div>
            <FireIcon className="h-12 w-12 text-red-200" />
          </div>
        </div>

        <div className="bg-gradient-to-r from-orange-500 to-orange-600 rounded-lg p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-orange-100">Suspicious Clusters</p>
              <p className="text-3xl font-bold">
                {graphData?.network_stats?.suspicious_clusters || 0}
              </p>
            </div>
            <ExclamationTriangleIcon className="h-12 w-12 text-orange-200" />
          </div>
        </div>
      </div>

      {/* Advanced Search & Entity Explorer */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex items-center space-x-4 mb-4">
          <div className="flex-1">
            <div className="relative">
              <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search entities, fraud rings, or network patterns..."
                className="w-full pl-10 pr-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 dark:bg-gray-700 dark:text-white"
              />
            </div>
          </div>
          <button className="bg-purple-600 hover:bg-purple-700 text-white px-6 py-3 rounded-lg font-medium">
            Analyze Network
          </button>
        </div>

        {/* Top Risky Entities */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Risky Users */}
          <div>
            <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-3 flex items-center">
              <UserGroupIcon className="h-5 w-5 mr-2 text-red-600" />
              High-Risk Users
            </h4>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {graphData?.top_risky_entities?.users?.slice(0, 10).map((user, index) => (
                <div 
                  key={user.id} 
                  className="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 cursor-pointer hover:bg-red-100 dark:hover:bg-red-900/30"
                  onClick={() => setSelectedEntity(user.id)}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-red-800 dark:text-red-200">{user.id}</p>
                      <p className="text-sm text-red-600 dark:text-red-300">
                        {user.transaction_count} transactions
                      </p>
                    </div>
                    <div className="text-right">
                      <div className="w-12 h-2 bg-red-200 rounded-full">
                        <div 
                          className="h-2 bg-red-600 rounded-full"
                          style={{ width: `${user.risk_score * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-xs text-red-600 dark:text-red-300">
                        {(user.risk_score * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Risky Merchants */}
          <div>
            <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-3 flex items-center">
              <BuildingStorefrontIcon className="h-5 w-5 mr-2 text-orange-600" />
              High-Risk Merchants
            </h4>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {graphData?.top_risky_entities?.merchants?.slice(0, 10).map((merchant, index) => (
                <div 
                  key={merchant.id} 
                  className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg border border-orange-200 cursor-pointer hover:bg-orange-100 dark:hover:bg-orange-900/30"
                  onClick={() => setSelectedEntity(merchant.id)}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-orange-800 dark:text-orange-200">{merchant.id}</p>
                      <p className="text-sm text-orange-600 dark:text-orange-300">
                        {merchant.transaction_count} transactions
                      </p>
                    </div>
                    <div className="text-right">
                      <div className="w-12 h-2 bg-orange-200 rounded-full">
                        <div 
                          className="h-2 bg-orange-600 rounded-full"
                          style={{ width: `${merchant.risk_score * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-xs text-orange-600 dark:text-orange-300">
                        {(merchant.risk_score * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Risky Devices */}
          <div>
            <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-3 flex items-center">
              <DevicePhoneMobileIcon className="h-5 w-5 mr-2 text-purple-600" />
              Suspicious Devices
            </h4>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {graphData?.top_risky_entities?.devices?.slice(0, 10).map((device, index) => (
                <div 
                  key={device.id} 
                  className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 cursor-pointer hover:bg-purple-100 dark:hover:bg-purple-900/30"
                  onClick={() => setSelectedEntity(device.id)}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-purple-800 dark:text-purple-200">{device.id}</p>
                      <p className="text-sm text-purple-600 dark:text-purple-300">
                        {device.user_count} users
                      </p>
                    </div>
                    <div className="text-right">
                      <div className="w-12 h-2 bg-purple-200 rounded-full">
                        <div 
                          className="h-2 bg-purple-600 rounded-full"
                          style={{ width: `${device.risk_score * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-xs text-purple-600 dark:text-purple-300">
                        {(device.risk_score * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* 3D Network Visualization */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white flex items-center">
            <CpuChipIcon className="h-6 w-6 mr-2 text-blue-600" />
            Network Topology Visualization ({networkView.toUpperCase()})
          </h3>
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">
              Powered by Neo4j Graph Database
            </span>
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
          </div>
        </div>
        
        <div className="relative">
          <canvas 
            ref={canvasRef}
            className="w-full h-96 bg-gray-900 rounded-lg border-2 border-gray-300 dark:border-gray-600"
            style={{ background: 'radial-gradient(circle, #1a1a2e 0%, #16213e 50%, #0f3460 100%)' }}
          >
            {/* Placeholder for 3D visualization */}
            <div className="absolute inset-0 flex items-center justify-center text-white">
              <div className="text-center">
                <CpuChipIcon className="h-16 w-16 mx-auto mb-4 text-blue-400 animate-pulse" />
                <p className="text-lg font-semibold">3D Network Visualization</p>
                <p className="text-sm text-gray-300 mt-2">
                  Interactive graph showing {graphData?.network_stats?.total_nodes || 0} nodes and {graphData?.network_stats?.total_relationships || 0} relationships
                </p>
                <div className="mt-4 flex justify-center space-x-4">
                  <div className="flex items-center">
                    <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
                    <span className="text-xs">High Risk</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-3 h-3 bg-yellow-500 rounded-full mr-2"></div>
                    <span className="text-xs">Medium Risk</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
                    <span className="text-xs">Low Risk</span>
                  </div>
                </div>
              </div>
            </div>
          </canvas>
          
          {/* Network Controls */}
          <div className="absolute top-4 right-4 bg-black/50 rounded-lg p-2 space-y-2">
            <button className="block w-full text-white text-xs px-3 py-1 bg-blue-600 rounded hover:bg-blue-700">
              Zoom In
            </button>
            <button className="block w-full text-white text-xs px-3 py-1 bg-blue-600 rounded hover:bg-blue-700">
              Zoom Out
            </button>
            <button className="block w-full text-white text-xs px-3 py-1 bg-blue-600 rounded hover:bg-blue-700">
              Reset View
            </button>
            <button className="block w-full text-white text-xs px-3 py-1 bg-purple-600 rounded hover:bg-purple-700">
              Auto Layout
            </button>
          </div>
        </div>
      </div>

      {/* Fraud Rings Detection */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white flex items-center">
            <ShieldExclamationIcon className="h-6 w-6 mr-2 text-red-600" />
            Active Fraud Rings ({fraudRings.length} detected)
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            AI-powered fraud ring detection using graph algorithms and behavioral analysis
          </p>
        </div>
        
        {loading ? (
          <div className="p-6 space-y-4">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="animate-pulse flex items-center space-x-4">
                <div className="h-16 w-16 bg-gray-200 dark:bg-gray-700 rounded-lg"></div>
                <div className="flex-1 space-y-2">
                  <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
                  <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-1/2"></div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="divide-y divide-gray-200 dark:divide-gray-700">
            {fraudRings.map((ring) => (
              <div 
                key={ring.ring_id} 
                className="p-6 hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer"
                onClick={() => setSelectedRing(selectedRing?.ring_id === ring.ring_id ? null : ring)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className={`w-16 h-16 rounded-lg flex items-center justify-center ${getRiskLevelColor(ring.risk_level)}`}>
                      <FireIcon className="h-8 w-8" />
                    </div>
                    <div>
                      <h4 className="text-lg font-semibold text-gray-900 dark:text-white">
                        {ring.ring_id}
                      </h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {ring.size} members • ${ring.total_amount.toLocaleString()} total amount
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-500">
                        Detected: {new Date(ring.detection_date).toLocaleDateString()} • Method: {ring.detection_method}
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <span className={`px-3 py-1 text-sm font-medium rounded-full border ${getStatusColor(ring.status)}`}>
                      {ring.status}
                    </span>
                    <span className={`px-3 py-1 text-sm font-medium rounded-full ${getRiskLevelColor(ring.risk_level)}`}>
                      {ring.risk_level}
                    </span>
                    <button className="text-blue-600 hover:text-blue-800 dark:text-blue-400">
                      <EyeIcon className="h-5 w-5" />
                    </button>
                  </div>
                </div>
                
                {selectedRing?.ring_id === ring.ring_id && (
                  <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-600">
                    <h5 className="font-medium text-gray-900 dark:text-white mb-3">Ring Members:</h5>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                      {ring.members.map((member, index) => (
                        <div key={index} className="p-3 bg-gray-50 dark:bg-gray-600 rounded-lg">
                          <div className="flex items-center justify-between">
                            <div>
                              <p className="font-medium text-gray-900 dark:text-white text-sm">
                                {member.user_id}
                              </p>
                              <p className="text-xs text-gray-600 dark:text-gray-400">
                                Role: {member.role}
                              </p>
                            </div>
                            <div className="text-right">
                              <div className="w-8 h-1 bg-gray-200 rounded-full">
                                <div 
                                  className="h-1 bg-red-600 rounded-full"
                                  style={{ width: `${member.risk_score * 100}%` }}
                                ></div>
                              </div>
                              <span className="text-xs text-gray-600 dark:text-gray-400">
                                {(member.risk_score * 100).toFixed(0)}%
                              </span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}