'use client'

import React, { useState } from 'react'
import { 
  ShieldCheckIcon,
  DocumentArrowUpIcon,
  PlayIcon,
  ChartBarIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline'
import { apiClient, TransactionRequest, formatCurrency, getRiskColor, getDecisionColor } from '../../../lib/api'

export function DetectionPage() {
  const [file, setFile] = useState<File | null>(null)
  const [activeTab, setActiveTab] = useState<'single' | 'batch'>('single')
  const [transactionData, setTransactionData] = useState<Partial<TransactionRequest>>({
    transaction_id: `tx_${Date.now()}`,
    amount: '',
    merchant_id: '',
    user_id: '',
    category: '',
    currency: 'USD',
    timestamp: new Date().toISOString(),
    device_id: '',
    ip_address: '',
    lat: undefined,
    lon: undefined
  })
  const [result, setResult] = useState<any>(null)
  const [batchResult, setBatchResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [batchLoading, setBatchLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0]
    if (selectedFile) {
      setFile(selectedFile)
      setBatchResult(null)
    }
  }

  const handleSingleTransaction = async () => {
    if (!transactionData.amount || !transactionData.user_id || !transactionData.merchant_id) {
      setError('Please fill in all required fields')
      return
    }

    setLoading(true)
    setError(null)
    
    try {
      const transaction: TransactionRequest = {
        transaction_id: transactionData.transaction_id || `tx_${Date.now()}`,
        user_id: transactionData.user_id || '',
        merchant_id: transactionData.merchant_id || '',
        amount: parseFloat(transactionData.amount as string),
        currency: transactionData.currency || 'USD',
        category: transactionData.category,
        timestamp: transactionData.timestamp || new Date().toISOString(),
        device_id: transactionData.device_id,
        ip_address: transactionData.ip_address,
        lat: transactionData.lat,
        lon: transactionData.lon
      }

      const response = await apiClient.analyzeTransaction(transaction)
      setResult(response)
    } catch (error: any) {
      console.error('Error analyzing transaction:', error)
      setError(error.message || 'Failed to analyze transaction')
    } finally {
      setLoading(false)
    }
  }

  const handleBatchProcessing = async () => {
    if (!file) {
      setError('Please select a CSV file')
      return
    }

    setBatchLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/transactions/batch`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Failed to process batch')
      }

      const result = await response.json()
      setBatchResult(result)
    } catch (error: any) {
      console.error('Error processing batch:', error)
      setError(error.message || 'Failed to process batch file')
    } finally {
      setBatchLoading(false)
    }
  }

  const generateSampleData = () => {
    const sampleTransactions = [
      {
        transaction_id: `tx_${Date.now()}_1`,
        user_id: 'user_sample_001',
        merchant_id: 'merchant_grocery_001',
        amount: 45.67,
        currency: 'USD',
        category: 'grocery',
        timestamp: new Date().toISOString()
      },
      {
        transaction_id: `tx_${Date.now()}_2`,
        user_id: 'user_sample_002',
        merchant_id: 'merchant_gambling_001',
        amount: 2500.00,
        currency: 'USD',
        category: 'gambling',
        timestamp: new Date(Date.now() - 3600000).toISOString()
      }
    ]

    setTransactionData(sampleTransactions[Math.floor(Math.random() * sampleTransactions.length)])
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              Real-time Fraud Detection
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Analyze individual transactions or process batch files for fraud detection
            </p>
          </div>
          <ShieldCheckIcon className="h-12 w-12 text-blue-600 dark:text-blue-400" />
        </div>
        
        {/* Error Display */}
        {error && (
          <div className="mb-6 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
            <div className="flex items-center">
              <ExclamationTriangleIcon className="h-6 w-6 text-red-600 dark:text-red-400 mr-2" />
              <p className="text-red-800 dark:text-red-200">{error}</p>
            </div>
          </div>
        )}

        {/* Tabs */}
        <div className="border-b border-gray-200 dark:border-gray-700">
          <nav className="-mb-px flex space-x-8">
            <button 
              onClick={() => setActiveTab('single')}
              className={`border-b-2 py-2 px-1 text-sm font-medium ${
                activeTab === 'single' 
                  ? 'border-blue-500 text-blue-600 dark:text-blue-400' 
                  : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
              }`}
            >
              Single Transaction
            </button>
            <button 
              onClick={() => setActiveTab('batch')}
              className={`border-b-2 py-2 px-1 text-sm font-medium ${
                activeTab === 'batch' 
                  ? 'border-blue-500 text-blue-600 dark:text-blue-400' 
                  : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
              }`}
            >
              Batch Processing
            </button>
          </nav>
        </div>

        {/* Single Transaction Tab */}
        {activeTab === 'single' && (
          <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                  Transaction Details
                </h3>
                <button
                  onClick={generateSampleData}
                  className="text-sm text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300"
                >
                  Use Sample Data
                </button>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="col-span-2">
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Transaction ID
                  </label>
                  <input
                    type="text"
                    value={transactionData.transaction_id || ''}
                    onChange={(e) => setTransactionData({...transactionData, transaction_id: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                    placeholder={`tx_${Date.now()}`}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Amount *
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    value={transactionData.amount}
                    onChange={(e) => setTransactionData({...transactionData, amount: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                    placeholder="1000.00"
                    required
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Currency
                  </label>
                  <select
                    value={transactionData.currency || 'USD'}
                    onChange={(e) => setTransactionData({...transactionData, currency: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  >
                    <option value="USD">USD</option>
                    <option value="EUR">EUR</option>
                    <option value="GBP">GBP</option>
                    <option value="CAD">CAD</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    User ID *
                  </label>
                  <input
                    type="text"
                    value={transactionData.user_id || ''}
                    onChange={(e) => setTransactionData({...transactionData, user_id: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                    placeholder="user_456"
                    required
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Merchant ID *
                  </label>
                  <input
                    type="text"
                    value={transactionData.merchant_id || ''}
                    onChange={(e) => setTransactionData({...transactionData, merchant_id: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                    placeholder="merchant_123"
                    required
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Category
                  </label>
                  <select
                    value={transactionData.category || ''}
                    onChange={(e) => setTransactionData({...transactionData, category: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  >
                    <option value="">Select category</option>
                    <option value="grocery">Grocery</option>
                    <option value="gas">Gas Station</option>
                    <option value="restaurant">Restaurant</option>
                    <option value="online">Online Purchase</option>
                    <option value="gambling">Gambling</option>
                    <option value="atm">ATM Withdrawal</option>
                    <option value="retail">Retail</option>
                    <option value="entertainment">Entertainment</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Device ID
                  </label>
                  <input
                    type="text"
                    value={transactionData.device_id || ''}
                    onChange={(e) => setTransactionData({...transactionData, device_id: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                    placeholder="device_789"
                  />
                </div>
              </div>
              
              <button
                onClick={handleSingleTransaction}
                disabled={loading}
                className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white px-4 py-2 rounded-md flex items-center justify-center space-x-2"
              >
                {loading ? (
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                ) : (
                  <PlayIcon className="h-4 w-4" />
                )}
                <span>{loading ? 'Analyzing...' : 'Detect Fraud'}</span>
              </button>
            </div>

            {/* Results Panel */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                Detection Results
              </h3>
              
              {result ? (
                <div className="space-y-4">
                  {/* Overall Score */}
                  <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        Fraud Score
                      </span>
                      <span className="text-lg font-bold text-gray-900 dark:text-white">
                        {(result.fraud_score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-3">
                      <div 
                        className={`h-3 rounded-full transition-all duration-500 ${
                          result.fraud_score > 0.7 ? 'bg-red-500' :
                          result.fraud_score > 0.4 ? 'bg-yellow-500' : 'bg-green-500'
                        }`}
                        style={{ width: `${result.fraud_score * 100}%` }}
                      ></div>
                    </div>
                  </div>
                  
                  {/* Risk Level and Decision */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center">
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300 block mb-2">
                        Risk Level
                      </span>
                      <span className={`px-3 py-2 rounded-full text-sm font-medium ${getRiskColor(result.risk_level)}`}>
                        {result.risk_level}
                      </span>
                    </div>
                    
                    <div className="text-center">
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300 block mb-2">
                        Decision
                      </span>
                      <span className={`px-3 py-2 rounded-full text-sm font-medium ${getDecisionColor(result.decision)}`}>
                        {result.decision}
                      </span>
                    </div>
                  </div>

                  {/* Risk Factors */}
                  {result.risk_factors && result.risk_factors.length > 0 && (
                    <div className="border-t border-gray-200 dark:border-gray-600 pt-4">
                      <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3 flex items-center">
                        <ExclamationTriangleIcon className="h-4 w-4 mr-2 text-yellow-500" />
                        Risk Factors
                      </h4>
                      <div className="space-y-2">
                        {result.risk_factors.map((factor: string, index: number) => (
                          <div key={index} className="text-sm text-gray-600 dark:text-gray-400 bg-yellow-50 dark:bg-yellow-900/20 p-2 rounded">
                            • {factor}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Recommendations */}
                  {result.recommendations && result.recommendations.length > 0 && (
                    <div className="border-t border-gray-200 dark:border-gray-600 pt-4">
                      <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3 flex items-center">
                        <InformationCircleIcon className="h-4 w-4 mr-2 text-blue-500" />
                        Recommendations
                      </h4>
                      <div className="space-y-2">
                        {result.recommendations.map((rec: string, index: number) => (
                          <div key={index} className="text-sm text-gray-600 dark:text-gray-400 bg-blue-50 dark:bg-blue-900/20 p-2 rounded">
                            • {rec}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-8 text-center">
                  <ChartBarIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-500 dark:text-gray-400">
                    Enter transaction details and click "Detect Fraud" to see results
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Batch Processing Tab */}
        {activeTab === 'batch' && (
          <div className="mt-6 space-y-6">
            <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-6">
              <div className="text-center">
                <DocumentArrowUpIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600 dark:text-gray-400 mb-2">
                  Upload CSV file for batch fraud detection
                </p>
                <p className="text-sm text-gray-500 dark:text-gray-500 mb-4">
                  Supports files up to 500MB with millions of transactions
                </p>
                <div className="flex flex-col sm:flex-row gap-4 items-center justify-center">
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleFileUpload}
                    className="hidden"
                    id="csv-upload"
                  />
                  <label
                    htmlFor="csv-upload"
                    className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md cursor-pointer inline-flex items-center space-x-2"
                  >
                    <DocumentArrowUpIcon className="h-4 w-4" />
                    <span>Choose File</span>
                  </label>
                  
                  {file && (
                    <button
                      onClick={handleBatchProcessing}
                      disabled={batchLoading}
                      className="bg-green-600 hover:bg-green-700 disabled:bg-green-400 text-white px-4 py-2 rounded-md inline-flex items-center space-x-2"
                    >
                      {batchLoading ? (
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                      ) : (
                        <PlayIcon className="h-4 w-4" />
                      )}
                      <span>{batchLoading ? 'Processing...' : 'Process Batch'}</span>
                    </button>
                  )}
                </div>
                
                {file && (
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-4">
                    Selected: {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
                  </p>
                )}
              </div>
            </div>

            {/* Batch Results */}
            {batchResult && (
              <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4 flex items-center">
                  <CheckCircleIcon className="h-6 w-6 text-green-500 mr-2" />
                  Batch Processing Results
                </h3>
                
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-gray-900 dark:text-white">
                      {batchResult.total_processed || 0}
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">Total Processed</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-red-600">
                      {batchResult.fraud_detected || 0}
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">Fraud Detected</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-yellow-600">
                      {batchResult.flagged_for_review || 0}
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">Flagged for Review</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">
                      {((batchResult.fraud_detected || 0) / (batchResult.total_processed || 1) * 100).toFixed(2)}%
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">Fraud Rate</div>
                  </div>
                </div>

                {batchResult.download_url && (
                  <div className="text-center">
                    <a
                      href={batchResult.download_url}
                      download
                      className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-md inline-flex items-center space-x-2"
                    >
                      <DocumentArrowUpIcon className="h-4 w-4" />
                      <span>Download Results</span>
                    </a>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
