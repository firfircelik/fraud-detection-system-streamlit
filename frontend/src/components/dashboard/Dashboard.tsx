'use client'

import React, { useState } from 'react'
import { Sidebar } from './Sidebar'
import { TopNav } from './TopNav'
import { OverviewPage } from './pages/OverviewPage'
import { DetectionPage } from './pages/DetectionPage'
import { AnalyticsPage } from './pages/AnalyticsPage'
import { GraphAnalyticsPage } from './pages/GraphAnalyticsPage'
import { TransactionsPage } from './pages/TransactionsPage'
import { SettingsPage } from './pages/SettingsPage'
import { MLModelsPage } from './pages/MLModelsPage'

type DashboardPage = 
  | 'overview' 
  | 'detection' 
  | 'analytics' 
  | 'enterprise-analytics'
  | 'transactions' 
  | 'ml-models' 
  | 'settings'

export function Dashboard() {
  const [currentPage, setCurrentPage] = useState<DashboardPage>('overview')
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const renderPage = () => {
    switch (currentPage) {
      case 'overview':
        return <OverviewPage />
      case 'detection':
        return <DetectionPage />
      case 'analytics':
        return <AnalyticsPage />
      case 'enterprise-analytics':
        return <GraphAnalyticsPage />
      case 'transactions':
        return <TransactionsPage />
      case 'ml-models':
        return <MLModelsPage />
      case 'settings':
        return <SettingsPage />
      default:
        return <OverviewPage />
    }
  }

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-900">
      {/* Sidebar */}
      <Sidebar 
        currentPage={currentPage}
        onPageChange={setCurrentPage}
        isOpen={sidebarOpen}
        onOpenChange={setSidebarOpen}
      />

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top navigation */}
        <TopNav 
          onMenuClick={() => setSidebarOpen(!sidebarOpen)}
          currentPage={currentPage}
        />

        {/* Page content */}
        <main className="flex-1 overflow-x-hidden overflow-y-auto bg-gray-50 dark:bg-gray-900">
          <div className="container mx-auto px-6 py-8">
            {renderPage()}
          </div>
        </main>
      </div>
    </div>
  )
}
