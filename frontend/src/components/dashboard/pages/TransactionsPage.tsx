'use client'

import React, { useState, useEffect } from 'react'
import { 
  CreditCardIcon,
  FunnelIcon,
  ArrowDownTrayIcon,
  ClockIcon,
  UserIcon,
  BuildingStorefrontIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  XCircleIcon
} from '@heroicons/react/24/outline'

interface Transaction {
  id: string
  user_id: string
  merchant_id: string
  amount: number
  currency: string
  category: string
  timestamp: string
  fraud_score: number
  risk_level: 'MINIMAL' | 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
  decision: 'APPROVED' | 'REVIEW' | 'DECLINED'
  device_id?: string
  location?: string
}

export function TransactionsPage() {
  const [transactions, setTransactions] = useState<Transaction[]>([])
  const [loading, setLoading] = useState(true)
  const [filters, setFilters] = useState({
    risk_level: '',
    decision: '',
    category: '',
    date_range: '24h'
  })
  const [searchTerm, setSearchTerm] = useState('')

  useEffect(() => {
    fetchTransactions()
  }, [filters])

  const fetchTransactions = async () => {
    setLoading(true)
    try {
      // Fetch real data from backend API
      const response = await fetch('http://localhost:8080/api/transactions/recent?limit=100')
      const data = await response.json()
      
      if (data.transactions) {
        setTransactions(data.transactions.map((t: any) => ({
          id: t.transaction_id,
          user_id: t.user_id,
          merchant_id: t.merchant_id,
          amount: t.amount,
          currency: t.currency || 'USD',
          category: t.category || 'retail',
          timestamp: t.timestamp,
          fraud_score: t.fraud_score,
          risk_level: t.risk_level,
          decision: t.decision,
          device_id: t.device_id,
          location: t.country
        })))
      }
      setLoading(false)
    } catch (error) {
      console.error('Failed to fetch transactions:', error)
      // Fallback to mock data
      setTimeout(() => {
        const mockTransactions: Transaction[] = Array.from({ length: 50 }, (_, i) => {
          const riskLevels: Transaction['risk_level'][] = ['MINIMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
          const decisions: Transaction['decision'][] = ['APPROVED', 'REVIEW', 'DECLINED']
          const categories = ['grocery', 'gas', 'restaurant', 'online', 'gambling', 'atm']
          const currencies = ['USD', 'EUR', 'GBP', 'CAD']
          
          const riskLevel = riskLevels[Math.floor(Math.random() * riskLevels.length)]
          
          return {
            id: `tx_${Date.now()}_${i}`,
            user_id: `user_${Math.floor(Math.random() * 1000)}`,
            merchant_id: `merchant_${Math.floor(Math.random() * 500)}`,
            amount: Math.floor(Math.random() * 5000) + 10,
            currency: currencies[Math.floor(Math.random() * currencies.length)],
            category: categories[Math.floor(Math.random() * categories.length)],
            timestamp: new Date(Date.now() - Math.random() * 24 * 60 * 60 * 1000).toISOString(),
            fraud_score: Math.random(),
            risk_level: riskLevel,
            decision: decisions[Math.floor(Math.random() * decisions.length)],
            device_id: `device_${Math.floor(Math.random() * 100)}`,
            location: `${Math.floor(Math.random() * 90)}°N, ${Math.floor(Math.random() * 180)}°W`
          }
        })
        setTransactions(mockTransactions);
        setLoading(false);
      }, 1000);
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

  const getDecisionColor = (decision: string) => {
    switch (decision) {
      case 'APPROVED': return 'text-green-600 bg-green-50 border-green-200'
      case 'REVIEW': return 'text-yellow-600 bg-yellow-50 border-yellow-200'
      case 'DECLINED': return 'text-red-600 bg-red-50 border-red-200'
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

  const filteredTransactions = transactions.filter(transaction => {
    const matchesSearch = searchTerm === '' || 
      transaction.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      transaction.user_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      transaction.merchant_id.toLowerCase().includes(searchTerm.toLowerCase())

    const matchesRiskLevel = filters.risk_level === '' || transaction.risk_level === filters.risk_level
    const matchesDecision = filters.decision === '' || transaction.decision === filters.decision
    const matchesCategory = filters.category === '' || transaction.category === filters.category

    return matchesSearch && matchesRiskLevel && matchesDecision && matchesCategory
  })

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Transaction Monitoring
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Real-time transaction monitoring and fraud detection
          </p>
        </div>
        <button className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg flex items-center space-x-2">
          <ArrowDownTrayIcon className="h-4 w-4" />
          <span>Export</span>
        </button>
      </div>

      {/* Filters */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          {/* Search */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Search
            </label>
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Transaction ID, User, Merchant..."
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
            />
          </div>

          {/* Risk Level Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Risk Level
            </label>
            <select
              value={filters.risk_level}
              onChange={(e) => setFilters({...filters, risk_level: e.target.value})}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
            >
              <option value="">All Risk Levels</option>
              <option value="MINIMAL">Minimal</option>
              <option value="LOW">Low</option>
              <option value="MEDIUM">Medium</option>
              <option value="HIGH">High</option>
              <option value="CRITICAL">Critical</option>
            </select>
          </div>

          {/* Decision Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Decision
            </label>
            <select
              value={filters.decision}
              onChange={(e) => setFilters({...filters, decision: e.target.value})}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
            >
              <option value="">All Decisions</option>
              <option value="APPROVED">Approved</option>
              <option value="REVIEW">Review</option>
              <option value="DECLINED">Declined</option>
            </select>
          </div>

          {/* Category Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Category
            </label>
            <select
              value={filters.category}
              onChange={(e) => setFilters({...filters, category: e.target.value})}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
            >
              <option value="">All Categories</option>
              <option value="grocery">Grocery</option>
              <option value="gas">Gas Station</option>
              <option value="restaurant">Restaurant</option>
              <option value="online">Online</option>
              <option value="gambling">Gambling</option>
              <option value="atm">ATM</option>
            </select>
          </div>

          {/* Date Range Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Time Range
            </label>
            <select
              value={filters.date_range}
              onChange={(e) => setFilters({...filters, date_range: e.target.value})}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
            >
              <option value="1h">Last Hour</option>
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
            </select>
          </div>
        </div>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center">
            <CreditCardIcon className="h-8 w-8 text-blue-600 dark:text-blue-400" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Total Transactions</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">{filteredTransactions.length}</p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center">
            <CheckCircleIcon className="h-8 w-8 text-green-600 dark:text-green-400" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Approved</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {filteredTransactions.filter(t => t.decision === 'APPROVED').length}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center">
            <ExclamationTriangleIcon className="h-8 w-8 text-yellow-600 dark:text-yellow-400" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Under Review</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {filteredTransactions.filter(t => t.decision === 'REVIEW').length}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center">
            <XCircleIcon className="h-8 w-8 text-red-600 dark:text-red-400" />
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Declined</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {filteredTransactions.filter(t => t.decision === 'DECLINED').length}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Transactions Table */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">
            Recent Transactions ({filteredTransactions.length})
          </h3>
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
                    Transaction
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Amount
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                    Risk Score
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
                {filteredTransactions.slice(0, 20).map((transaction) => {
                  const DecisionIcon = getDecisionIcon(transaction.decision)
                  return (
                    <tr key={transaction.id} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div>
                          <div className="text-sm font-medium text-gray-900 dark:text-white">
                            {transaction.id}
                          </div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">
                            {transaction.user_id} → {transaction.merchant_id}
                          </div>
                          <div className="text-xs text-gray-400 dark:text-gray-500">
                            {transaction.category}
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm font-medium text-gray-900 dark:text-white">
                          {transaction.amount.toLocaleString()} {transaction.currency}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <div className="w-16 bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                            <div 
                              className="bg-blue-600 h-2 rounded-full" 
                              style={{ width: `${transaction.fraud_score * 100}%` }}
                            ></div>
                          </div>
                          <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">
                            {(transaction.fraud_score * 100).toFixed(1)}%
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full border ${getRiskColor(transaction.risk_level)}`}>
                          {transaction.risk_level}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <DecisionIcon className="h-4 w-4 mr-2" />
                          <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full border ${getDecisionColor(transaction.decision)}`}>
                            {transaction.decision}
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                        {new Date(transaction.timestamp).toLocaleString()}
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
