'use client'

import React from 'react'
import Link from 'next/link'
import { 
  HomeIcon, 
  ShieldCheckIcon, 
  ChartBarIcon, 
  CreditCardIcon,
  CpuChipIcon,
  CogIcon,
  XMarkIcon,
  Bars3Icon
} from '@heroicons/react/24/outline'

type DashboardPage = 
  | 'overview' 
  | 'detection' 
  | 'analytics' 
  | 'enterprise-analytics'
  | 'transactions' 
  | 'ml-models' 
  | 'settings'

interface SidebarProps {
  currentPage: DashboardPage
  onPageChange: (page: DashboardPage) => void
  isOpen: boolean
  onOpenChange: (open: boolean) => void
}

const navigationItems = [
  {
    id: 'overview' as DashboardPage,
    name: 'Overview',
    icon: HomeIcon,
    description: 'Dashboard overview and key metrics'
  },
  {
    id: 'detection' as DashboardPage,
    name: 'Fraud Detection',
    icon: ShieldCheckIcon,
    description: 'Real-time fraud detection and analysis'
  },
  {
    id: 'analytics' as DashboardPage,
    name: 'Analytics',
    icon: ChartBarIcon,
    description: 'Advanced analytics and reporting'
  },
  {
    id: 'enterprise-analytics' as DashboardPage,
    name: 'Enterprise Analytics',
    icon: ChartBarIcon,
    description: 'Enterprise-grade analytics with Elasticsearch, geospatial data, and system monitoring'
  },
  {
    id: 'transactions' as DashboardPage,
    name: 'Transactions',
    icon: CreditCardIcon,
    description: 'Transaction monitoring and history'
  },
  {
    id: 'ml-models' as DashboardPage,
    name: 'ML Models',
    icon: CpuChipIcon,
    description: 'Machine learning model management'
  },
  {
    id: 'settings' as DashboardPage,
    name: 'Settings',
    icon: CogIcon,
    description: 'System configuration and preferences'
  }
]

export function Sidebar({ currentPage, onPageChange, isOpen, onOpenChange }: SidebarProps) {
  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 z-50 bg-black bg-opacity-50 lg:hidden"
          onClick={() => onOpenChange(false)}
        />
      )}

      {/* Sidebar */}
      <div className={`
        fixed inset-y-0 left-0 z-50 w-64 bg-white dark:bg-gray-800 transform transition-transform duration-200 ease-in-out lg:translate-x-0 lg:static lg:inset-0
        ${isOpen ? 'translate-x-0' : '-translate-x-full'}
      `}>
        <div className="flex items-center justify-between h-16 px-6 bg-gradient-to-r from-blue-600 to-purple-600">
          <div className="flex items-center">
            <ShieldCheckIcon className="h-8 w-8 text-white" />
            <span className="ml-2 text-lg font-semibold text-white">
              Fraud Guard
            </span>
          </div>
          <button
            onClick={() => onOpenChange(false)}
            className="lg:hidden text-white hover:text-gray-200"
          >
            <XMarkIcon className="h-6 w-6" />
          </button>
        </div>

        <nav className="mt-8 px-4">
          <div className="space-y-2">
            {navigationItems.map((item) => {
              const Icon = item.icon
              const isActive = currentPage === item.id
              
              return (
                <button
                  key={item.id}
                  onClick={() => {
                    onPageChange(item.id)
                    onOpenChange(false)
                  }}
                  className={`
                    w-full flex items-center px-4 py-3 text-sm font-medium rounded-lg transition-colors
                    ${isActive 
                      ? 'bg-blue-50 text-blue-700 border-r-2 border-blue-700 dark:bg-blue-900 dark:text-blue-200' 
                      : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900 dark:text-gray-300 dark:hover:bg-gray-700 dark:hover:text-white'
                    }
                  `}
                >
                  <Icon className="h-5 w-5 mr-3" />
                  <div className="text-left">
                    <div className="font-medium">{item.name}</div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      {item.description}
                    </div>
                  </div>
                </button>
              )
            })}
          </div>
        </nav>

        {/* System status */}
        <div className="absolute bottom-4 left-4 right-4">
          <div className="bg-green-50 dark:bg-green-900 border border-green-200 dark:border-green-800 rounded-lg p-3">
            <div className="flex items-center">
              <div className="h-2 w-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="ml-2 text-sm text-green-700 dark:text-green-300 font-medium">
                System Online
              </span>
            </div>
            <div className="text-xs text-green-600 dark:text-green-400 mt-1">
              4 ML models active
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
