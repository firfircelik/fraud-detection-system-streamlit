'use client'

import React, { useState, useEffect, useRef } from 'react'
import { 
  CreditCardIcon,
  FunnelIcon,
  ArrowDownTrayIcon,
  ClockIcon,
  UserIcon,
  BuildingStorefrontIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  XCircleIcon,
  MapPinIcon,
  MagnifyingGlassIcon,
  ChartBarIcon,
  GlobeAltIcon,
  BoltIcon,
  EyeIcon
} from '@heroicons/react/24/outline'

interface Transaction {
  transaction_id: string
  user_id: string
  merchant_id: string
  amount: number
  currency: string
  category: string
  timestamp: string
  fraud_score: number
  risk_level: 'MINIMAL' | 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
  decision: 'APPROVED' | 'REVIEW' | 'DECLINED'
  device_type?: string
  country?: string
  latitude?: number
  longitude?: number
  ip_address?: string
  seconds_ago: number
  merchant_name?: string
}

interface GeospatialData {
  country_analytics: Array<{
    country: string
    total_transactions: number
    fraud_transactions: number
    fraud_rate: number
    avg_fraud_score: number
    total_amount: number
    coordinates: { lat: number; lng: number }
    risk_level: string
  }>
  fraud_hotspots: Array<{
    lat: number
    lng: number
    transaction_count: number
    fraud_count: number
    fraud_rate: number
    country: string
    intensity: number
  }>
  velocity_anomalies: Array<{
    user_id: string
    current_location: { lat: number; lng: number }
    previous_location: { lat: number; lng: number }
    time_diff_hours: number
    velocity: number
    risk_level: string
  }>
}

export function EnterpriseTransactionsPage() {
  const [transactions, setTransactions] = useState<Transaction[]>([])
  const [geospatialData, setGeospatialData] = useState<GeospatialData | null>(null)
  const [loading, setLoading] = useState(true)
  const [realTimeEnabled, setRealTimeEnabled] = useState(true)
  const [currentPage, setCurrentPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<any[]>([])
  const [showMap, setShowMap] = useState(true)
  const [filters, setFilters] = useState({
    risk_level: '',
    country: '',
    decision: '',
    category: '',
    amount_min: '',
    amount_max: ''
  })

  const intervalRef = useRef<NodeJS.Timeout | null>(null)

  useEffect(() => {
    fetchTransactions()
    fetchGeospatialData()
    
    if (realTimeEnabled) {
      intervalRef.current = setInterval(() => {
        fetchTransactions()
        fetchGeospatialData()
      }, 5000) // Update every 5 seconds
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [filters, currentPage, realTimeEnabled])

  const fetchTransactions = async () => {
    try {
      const params = new URLSearchParams({
        limit: '500',
        page: currentPage.toString(),
        ...(filters.risk_level && { filter_risk: filters.risk_level }),
        ...(filters.country && { filter_country: filters.country })
      })

      const response = await fetch(`http://localhost:8080/api/transactions/recent?${params}`)
      const data = await response.json()
      
      if (data.transactions) {
        setTransactions(data.transactions)
        setTotalPages(data.pagination?.total_pages || 1)
      }
      setLoading(false)
    } catch (error) {
      console.error('Failed to fetch transactions:', error)
      setLoading(false)
    }
  }

  const fetchGeospatialData = async () => {
    try {
      const response = await fetch('http://localhost:8080/api/geospatial/analytics')
      const data = await response.json()
      setGeospatialData(data)
    } catch (error) {
      console.error('Failed to fetch geospatial data:', error)
    }
  }

  const performSearch = async () => {
    if (!searchQuery.trim()) return
    
    try {
      const response = await fetch(`http://localhost:8080/api/elasticsearch/search?query=${encodeURIComponent(searchQuery)}&size=20`)
      const data = await response.json()
      setSearchResults(data.results || [])
    } catch (error) {
      console.error('Search failed:', error)
    }
  }

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'MINIMAL': return 'text-green-600 bg-green-50 border-green-200'
      case 'LOW': return 'text-yellow-600 bg-yellow-50 border-yellow-200'
      case 'MEDIUM': return 'text-orange-600 bg-orange-50 border-orange-200'
      case 'HIGH': return 'text-red-600 bg-red-50 border-red-200'
      case 'CRITICAL': return 'text-red-800 bg-red-100 border-red-300'
      default: return 'text-gray-600 bg-gray-50 border-gray-200'
    }
  }

  const getDecisionIcon = (decision: string) => {
    switch (decision) {
      case 'APPROVED': return CheckCircleIcon
      case 'REVIEW': return ExclamationTriangleIcon
      case 'DECLINED': return XCircleIcon
      default: return CheckCircleIcon
    }
  }

  const formatTimeAgo = (secondsAgo: number) => {
    if (secondsAgo < 60) return `${Math.floor(secondsAgo)}s ago`
    if (secondsAgo < 3600) return `${Math.floor(secondsAgo / 60)}m ago`
    if (secondsAgo < 86400) return `${Math.floor(secondsAgo / 3600)}h ago`
    return `${Math.floor(secondsAgo / 86400)}d ago`
  }

  return (
    <div className="space-y-6">
      {/* Enhanced Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center">
            <GlobeAltIcon className="h-8 w-8 mr-3 text-blue-600" />
            Enterprise Transaction Intelligence
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Real-time global transaction monitoring with geospatial analytics and ML-powered fraud detection
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
            {realTimeEnabled ? 'Live' : 'Paused'}
          </button>
          
          <button
            onClick={() => setShowMap(!showMap)}
            className="flex items-center px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium"
          >
            <MapPinIcon className="h-4 w-4 mr-2" />
            {showMap ? 'Hide Map' : 'Show Map'}
          </button>
          
          <button className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg flex items-center space-x-2">
            <ArrowDownTrayIcon className="h-4 w-4" />
            <span>Export Analytics</span>
          </button>
        </div>
      </div>

      {/* Real-time Stats Dashboard */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-6">
        <div className="bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-blue-100">Total Transactions</p>
              <p className="text-3xl font-bold">{transactions.length.toLocaleString()}</p>
            </div>
            <CreditCardIcon className="h-12 w-12 text-blue-200" />
          </div>
        </div>

        <div className="bg-gradient-to-r from-green-500 to-green-600 rounded-lg p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-green-100">Approved</p>
              <p className="text-3xl font-bold">
                {transactions.filter(t => t.decision === 'APPROVED').length}
              </p>
            </div>
            <CheckCircleIcon className="h-12 w-12 text-green-200" />
          </div>
        </div>

        <div className="bg-gradient-to-r from-yellow-500 to-yellow-600 rounded-lg p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-yellow-100">Under Review</p>
              <p className="text-3xl font-bold">
                {transactions.filter(t => t.decision === 'REVIEW').length}
              </p>
            </div>
            <ExclamationTriangleIcon className="h-12 w-12 text-yellow-200" />
          </div>
        </div>

        <div className="bg-gradient-to-r from-red-500 to-red-600 rounded-lg p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-red-100">Declined</p>
              <p className="text-3xl font-bold">
                {transactions.filter(t => t.decision === 'DECLINED').length}
              </p>
            </div>
            <XCircleIcon className="h-12 w-12 text-red-200" />
          </div>
        </div>

        <div className="bg-gradient-to-r from-purple-500 to-purple-600 rounded-lg p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-purple-100">Countries Active</p>
              <p className="text-3xl font-bold">
                {geospatialData?.country_analytics?.length || 0}
              </p>
            </div>
            <GlobeAltIcon className="h-12 w-12 text-purple-200" />
          </div>
        </div>
      </div>

      {/* Advanced Search */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex items-center space-x-4 mb-4">
          <div className="flex-1">
            <div className="relative">
              <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && performSearch()}
                placeholder="Search transactions, users, merchants, countries..."
                className="w-full pl-10 pr-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
              />
            </div>
          </div>
          <button
            onClick={performSearch}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-medium"
          >
            Search
          </button>
        </div>

        {/* Advanced Filters */}
        <div className="grid grid-cols-1 md:grid-cols-6 gap-4">
          <select
            value={filters.risk_level}
            onChange={(e) => setFilters({...filters, risk_level: e.target.value})}
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          >
            <option value="">All Risk Levels</option>
            <option value="MINIMAL">Minimal</option>
            <option value="LOW">Low</option>
            <option value="MEDIUM">Medium</option>
            <option value="HIGH">High</option>
            <option value="CRITICAL">Critical</option>
          </select>

          <select
            value={filters.country}
            onChange={(e) => setFilters({...filters, country: e.target.value})}
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          >
            <option value="">All Countries</option>
            {geospatialData?.country_analytics?.map(country => (
              <option key={country.country} value={country.country}>
                {country.country} ({country.total_transactions})
              </option>
            ))}
          </select>

          <select
            value={filters.decision}
            onChange={(e) => setFilters({...filters, decision: e.target.value})}
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          >
            <option value="">All Decisions</option>
            <option value="APPROVED">Approved</option>
            <option value="REVIEW">Review</option>
            <option value="DECLINED">Declined</option>
          </select>

          <input
            type="number"
            value={filters.amount_min}
            onChange={(e) => setFilters({...filters, amount_min: e.target.value})}
            placeholder="Min Amount"
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          />

          <input
            type="number"
            value={filters.amount_max}
            onChange={(e) => setFilters({...filters, amount_max: e.target.value})}
            placeholder="Max Amount"
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          />

          <button
            onClick={() => setFilters({
              risk_level: '',
              country: '',
              decision: '',
              category: '',
              amount_min: '',
              amount_max: ''
            })}
            className="bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded-md"
          >
            Clear Filters
          </button>
        </div>
      </div>

      {/* Geospatial Analytics */}
      {showMap && geospatialData && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
            <MapPinIcon className="h-6 w-6 mr-2 text-blue-600" />
            Global Fraud Intelligence Map
          </h3>
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Country Analytics */}
            <div className="lg:col-span-2">
              <h4 className="text-lg font-medium text-gray-900 dark:text-white mb-3">Country Risk Analysis</h4>
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {geospatialData.country_analytics.map((country, index) => (
                  <div key={country.country} className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className={`w-3 h-3 rounded-full ${
                        country.risk_level === 'HIGH' ? 'bg-red-500' :
                        country.risk_level === 'MEDIUM' ? 'bg-yellow-500' : 'bg-green-500'
                      }`}></div>
                      <div>
                        <p className="font-medium text-gray-900 dark:text-white">{country.country}</p>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {country.total_transactions.toLocaleString()} transactions
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="font-semibold text-gray-900 dark:text-white">
                        {country.fraud_rate.toFixed(2)}% fraud rate
                      </p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        ${country.total_amount.toLocaleString()}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Fraud Hotspots & Velocity Anomalies */}
            <div>
              <h4 className="text-lg font-medium text-gray-900 dark:text-white mb-3">Real-time Alerts</h4>
              
              {geospatialData.fraud_hotspots.length > 0 && (
                <div className="mb-4">
                  <h5 className="text-md font-medium text-red-600 mb-2">🔥 Fraud Hotspots</h5>
                  <div className="space-y-2">
                    {geospatialData.fraud_hotspots.slice(0, 5).map((hotspot, index) => (
                      <div key={index} className="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200">
                        <p className="font-medium text-red-800 dark:text-red-200">
                          {hotspot.country} ({hotspot.lat.toFixed(2)}, {hotspot.lng.toFixed(2)})
                        </p>
                        <p className="text-sm text-red-600 dark:text-red-300">
                          {hotspot.fraud_count}/{hotspot.transaction_count} transactions ({hotspot.fraud_rate.toFixed(1)}% fraud)
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {geospatialData.velocity_anomalies.length > 0 && (
                <div>
                  <h5 className="text-md font-medium text-orange-600 mb-2">⚡ Velocity Anomalies</h5>
                  <div className="space-y-2">
                    {geospatialData.velocity_anomalies.slice(0, 3).map((anomaly, index) => (
                      <div key={index} className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg border border-orange-200">
                        <p className="font-medium text-orange-800 dark:text-orange-200">
                          User: {anomaly.user_id}
                        </p>
                        <p className="text-sm text-orange-600 dark:text-orange-300">
                          Velocity: {anomaly.velocity.toFixed(0)} km/h in {anomaly.time_diff_hours}h
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {geospatialData.fraud_hotspots.length === 0 && geospatialData.velocity_anomalies.length === 0 && (
                <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200">
                  <p className="text-green-800 dark:text-green-200 font-medium">✅ No active alerts</p>
                  <p className="text-sm text-green-600 dark:text-green-300">All systems operating normally</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Search Results */}
      {searchResults.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Search Results ({searchResults.length} found)
          </h3>
          <div className="space-y-3">
            {searchResults.slice(0, 10).map((result, index) => (
              <div key={index} className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-gray-900 dark:text-white">
                      {result.transaction_id}
                    </p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {result.merchant_name} • {result.country} • ${result.amount}
                    </p>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-1 text-xs font-semibold rounded-full ${getRiskColor(result.risk_level)}`}>
                      {result.risk_level}
                    </span>
                    <span className="text-sm text-gray-500">
                      Score: {result.relevance_score?.toFixed(2)}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Enhanced Transactions Table */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">
            Live Transaction Stream ({transactions.length} transactions)
          </h3>
          <div className="flex items-center space-x-4">
            <span className="text-sm text-gray-600 dark:text-gray-400">
              Page {currentPage} of {totalPages}
            </span>
            <div className="flex space-x-2">
              <button
                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
                className="px-3 py-1 bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-300 rounded disabled:opacity-50"
              >
                Previous
              </button>
              <button
                onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                disabled={currentPage === totalPages}
                className="px-3 py-1 bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-300 rounded disabled:opacity-50"
              >
                Next
              </button>
            </div>
          </div>
        </div>
        
        {loading ? (
          <div className="p-6 space-y-4">
            {[...Array(10)].map((_, i) => (
              <div key={i} className="animate-pulse flex items-center space-x-4">
                <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/4"></div>
                <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/6"></div>
                <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/6"></div>
                <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/4"></div>
              </div>
            ))}
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-700">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Transaction Details
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Amount & Location
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Risk Analysis
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Decision & Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Time & Device
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                {transactions.map((transaction) => {
                  const DecisionIcon = getDecisionIcon(transaction.decision)
                  return (
                    <tr key={transaction.transaction_id} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div>
                          <div className="text-sm font-medium text-gray-900 dark:text-white">
                            {transaction.transaction_id}
                          </div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">
                            {transaction.user_id} → {transaction.merchant_name || transaction.merchant_id}
                          </div>
                          <div className="text-xs text-gray-400 dark:text-gray-500">
                            {transaction.category}
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div>
                          <div className="text-sm font-medium text-gray-900 dark:text-white">
                            {transaction.amount.toLocaleString()} {transaction.currency}
                          </div>
                          <div className="text-sm text-gray-500 dark:text-gray-400 flex items-center">
                            <MapPinIcon className="h-3 w-3 mr-1" />
                            {transaction.country}
                            {transaction.latitude && transaction.longitude && (
                              <span className="ml-1 text-xs">
                                ({transaction.latitude.toFixed(2)}, {transaction.longitude.toFixed(2)})
                              </span>
                            )}
                          </div>
                          {transaction.ip_address && (
                            <div className="text-xs text-gray-400 dark:text-gray-500">
                              IP: {transaction.ip_address}
                            </div>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <div className="w-16 bg-gray-200 dark:bg-gray-600 rounded-full h-2 mr-2">
                            <div 
                              className={`h-2 rounded-full ${
                                transaction.fraud_score > 0.7 ? 'bg-red-600' :
                                transaction.fraud_score > 0.4 ? 'bg-yellow-600' : 'bg-green-600'
                              }`}
                              style={{ width: `${transaction.fraud_score * 100}%` }}
                            ></div>
                          </div>
                          <span className="text-sm text-gray-600 dark:text-gray-400">
                            {(transaction.fraud_score * 100).toFixed(1)}%
                          </span>
                        </div>
                        <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full border ${getRiskColor(transaction.risk_level)}`}>
                          {transaction.risk_level}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <DecisionIcon className="h-4 w-4 mr-2" />
                          <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full border ${
                            transaction.decision === 'APPROVED' ? 'text-green-600 bg-green-50 border-green-200' :
                            transaction.decision === 'REVIEW' ? 'text-yellow-600 bg-yellow-50 border-yellow-200' :
                            'text-red-600 bg-red-50 border-red-200'
                          }`}>
                            {transaction.decision}
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div>
                          <div className="text-sm text-gray-900 dark:text-white">
                            {formatTimeAgo(transaction.seconds_ago)}
                          </div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">
                            {new Date(transaction.timestamp).toLocaleString()}
                          </div>
                          {transaction.device_type && (
                            <div className="text-xs text-gray-400 dark:text-gray-500">
                              {transaction.device_type}
                            </div>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex space-x-2">
                          <button className="text-blue-600 hover:text-blue-900 dark:text-blue-400 dark:hover:text-blue-300">
                            <EyeIcon className="h-4 w-4" />
                          </button>
                          <button className="text-purple-600 hover:text-purple-900 dark:text-purple-400 dark:hover:text-purple-300">
                            <ChartBarIcon className="h-4 w-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}