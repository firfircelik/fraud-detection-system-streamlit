'use client'

import { useState } from 'react'
import { useQuery } from 'react-query'
import { 
  Cog6ToothIcon,
  ShieldCheckIcon,
  BellIcon,
  ServerStackIcon,
  UserGroupIcon,
  KeyIcon,
  ChartBarIcon,
  CheckCircleIcon,
  InformationCircleIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline'

interface SettingsSectionProps {
  title: string
  description: string
  icon: React.ElementType
  children: React.ReactNode
}

function SettingsSection({ title, description, icon: Icon, children }: SettingsSectionProps) {
  return (
    <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
      <div className="flex items-center mb-4">
        <Icon className="h-6 w-6 text-blue-600 mr-3" />
        <div>
          <h3 className="text-lg font-medium text-gray-900">{title}</h3>
          <p className="text-sm text-gray-600">{description}</p>
        </div>
      </div>
      {children}
    </div>
  )
}

export function SettingsPage() {
  const [activeTab, setActiveTab] = useState<'general' | 'security' | 'models' | 'notifications'>('general')
  const [settings, setSettings] = useState({
    // General Settings
    autoRetraining: true,
    batchProcessing: true,
    realTimeAlerts: true,
    dataRetention: '90',
    // Security Settings
    twoFactorAuth: false,
    sessionTimeout: '30',
    ipWhitelist: '',
    // Model Settings
    fraudThreshold: '0.7',
    ensembleWeights: {
      randomForest: 0.3,
      logisticRegression: 0.25,
      isolationForest: 0.25,
      svm: 0.2
    },
    // Notification Settings
    emailAlerts: true,
    slackIntegration: false,
    webhookUrl: ''
  })

  // Fetch system health
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: () => fetch('http://localhost:8080/api/health').then(res => res.json()),
    refetchInterval: 30000,
  })

  const handleSettingChange = (key: string, value: any) => {
    setSettings(prev => ({
      ...prev,
      [key]: value
    }))
  }

  const handleSave = async () => {
    // In a real app, this would save to backend
    console.log('Saving settings:', settings)
    alert('Settings saved successfully!')
  }

  const tabs = [
    { id: 'general', name: 'General', icon: Cog6ToothIcon },
    { id: 'security', name: 'Security', icon: ShieldCheckIcon },
    { id: 'models', name: 'ML Models', icon: ChartBarIcon },
    { id: 'notifications', name: 'Notifications', icon: BellIcon },
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
            <Cog6ToothIcon className="h-8 w-8 text-gray-600" />
            System Settings
          </h1>
          <p className="text-gray-600 mt-1">Configure fraud detection system parameters</p>
        </div>
        <button
          onClick={handleSave}
          className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
        >
          Save Changes
        </button>
      </div>

      {/* System Status */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <ServerStackIcon className="h-5 w-5" />
          System Status
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {Object.entries(health?.services || {}).map(([service, status]) => (
            <div key={service} className="flex items-center gap-2">
              {status === 'healthy' ? (
                <CheckCircleIcon className="h-5 w-5 text-green-500" />
              ) : (
                <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />
              )}
              <span className="text-sm font-medium capitalize">
                {service.replace('_', ' ')}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`py-2 px-1 border-b-2 font-medium text-sm flex items-center gap-2 ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <tab.icon className="h-4 w-4" />
              {tab.name}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="space-y-6">
        {activeTab === 'general' && (
          <>
            <SettingsSection
              title="General Configuration"
              description="Basic system settings and preferences"
              icon={Cog6ToothIcon}
            >
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-sm font-medium text-gray-900">Auto Model Retraining</label>
                    <p className="text-sm text-gray-600">Automatically retrain models when performance drops</p>
                  </div>
                  <input
                    type="checkbox"
                    checked={settings.autoRetraining}
                    onChange={(e) => handleSettingChange('autoRetraining', e.target.checked)}
                    className="h-4 w-4 text-blue-600 rounded border-gray-300"
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-sm font-medium text-gray-900">Batch Processing</label>
                    <p className="text-sm text-gray-600">Enable batch processing for large datasets</p>
                  </div>
                  <input
                    type="checkbox"
                    checked={settings.batchProcessing}
                    onChange={(e) => handleSettingChange('batchProcessing', e.target.checked)}
                    className="h-4 w-4 text-blue-600 rounded border-gray-300"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-900 mb-2">Data Retention (days)</label>
                  <input
                    type="number"
                    value={settings.dataRetention}
                    onChange={(e) => handleSettingChange('dataRetention', e.target.value)}
                    className="w-32 px-3 py-2 border border-gray-300 rounded-md text-sm"
                    min="1"
                    max="365"
                  />
                </div>
              </div>
            </SettingsSection>
          </>
        )}

        {activeTab === 'security' && (
          <>
            <SettingsSection
              title="Security Settings"
              description="Authentication and access control settings"
              icon={ShieldCheckIcon}
            >
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-sm font-medium text-gray-900">Two-Factor Authentication</label>
                    <p className="text-sm text-gray-600">Require 2FA for admin access</p>
                  </div>
                  <input
                    type="checkbox"
                    checked={settings.twoFactorAuth}
                    onChange={(e) => handleSettingChange('twoFactorAuth', e.target.checked)}
                    className="h-4 w-4 text-blue-600 rounded border-gray-300"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-900 mb-2">Session Timeout (minutes)</label>
                  <input
                    type="number"
                    value={settings.sessionTimeout}
                    onChange={(e) => handleSettingChange('sessionTimeout', e.target.value)}
                    className="w-32 px-3 py-2 border border-gray-300 rounded-md text-sm"
                    min="5"
                    max="480"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-900 mb-2">IP Whitelist</label>
                  <textarea
                    value={settings.ipWhitelist}
                    onChange={(e) => handleSettingChange('ipWhitelist', e.target.value)}
                    placeholder="Enter IP addresses, one per line"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                    rows={3}
                  />
                </div>
              </div>
            </SettingsSection>
          </>
        )}

        {activeTab === 'models' && (
          <>
            <SettingsSection
              title="ML Model Configuration"
              description="Machine learning model parameters and thresholds"
              icon={ChartBarIcon}
            >
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-900 mb-2">Fraud Detection Threshold</label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={settings.fraudThreshold}
                    onChange={(e) => handleSettingChange('fraudThreshold', e.target.value)}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>0.0 (Lenient)</span>
                    <span>Current: {settings.fraudThreshold}</span>
                    <span>1.0 (Strict)</span>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-900 mb-3">Ensemble Model Weights</label>
                  <div className="space-y-3">
                    {Object.entries(settings.ensembleWeights).map(([model, weight]) => (
                      <div key={model} className="flex items-center justify-between">
                        <span className="text-sm capitalize">{model.replace(/([A-Z])/g, ' $1')}</span>
                        <div className="flex items-center gap-2">
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={weight}
                            onChange={(e) => handleSettingChange('ensembleWeights', {
                              ...settings.ensembleWeights,
                              [model]: parseFloat(e.target.value)
                            })}
                            className="w-24"
                          />
                          <span className="text-sm w-12">{weight}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </SettingsSection>
          </>
        )}

        {activeTab === 'notifications' && (
          <>
            <SettingsSection
              title="Notification Settings"
              description="Configure alerts and notification channels"
              icon={BellIcon}
            >
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-sm font-medium text-gray-900">Email Alerts</label>
                    <p className="text-sm text-gray-600">Send fraud alerts via email</p>
                  </div>
                  <input
                    type="checkbox"
                    checked={settings.emailAlerts}
                    onChange={(e) => handleSettingChange('emailAlerts', e.target.checked)}
                    className="h-4 w-4 text-blue-600 rounded border-gray-300"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-sm font-medium text-gray-900">Slack Integration</label>
                    <p className="text-sm text-gray-600">Send alerts to Slack channels</p>
                  </div>
                  <input
                    type="checkbox"
                    checked={settings.slackIntegration}
                    onChange={(e) => handleSettingChange('slackIntegration', e.target.checked)}
                    className="h-4 w-4 text-blue-600 rounded border-gray-300"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-900 mb-2">Webhook URL</label>
                  <input
                    type="url"
                    value={settings.webhookUrl}
                    onChange={(e) => handleSettingChange('webhookUrl', e.target.value)}
                    placeholder="https://your-webhook-url.com/fraud-alerts"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                  />
                </div>
              </div>
            </SettingsSection>
          </>
        )}
      </div>
    </div>
  )
}