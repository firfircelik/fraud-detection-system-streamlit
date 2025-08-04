'use client'

import React from 'react'
import { 
  Bars3Icon,
  BellIcon,
  UserCircleIcon,
  MoonIcon,
  SunIcon
} from '@heroicons/react/24/outline'
import { useTheme } from 'next-themes'

type DashboardPage = 
  | 'overview' 
  | 'detection' 
  | 'analytics' 
  | 'enterprise-analytics'
  | 'transactions' 
  | 'ml-models' 
  | 'settings'

interface TopNavProps {
  onMenuClick: () => void
  currentPage: DashboardPage
}

const pageTitle: Record<DashboardPage, string> = {
  overview: 'Dashboard Overview',
  detection: 'Fraud Detection',
  analytics: 'Advanced Analytics',
  'enterprise-analytics': 'Enterprise Analytics Dashboard',
  transactions: 'Transaction Monitoring',
  'ml-models': 'ML Model Management',
  settings: 'System Settings'
}

export function TopNav({ onMenuClick, currentPage }: TopNavProps) {
  const { theme, setTheme } = useTheme()

  return (
    <header className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between h-16 px-6">
        {/* Left side */}
        <div className="flex items-center">
          <button
            onClick={onMenuClick}
            className="lg:hidden p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <Bars3Icon className="h-6 w-6" />
          </button>
          
          <div className="ml-4 lg:ml-0">
            <h1 className="text-xl font-semibold text-gray-900 dark:text-white">
              {pageTitle[currentPage]}
            </h1>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Enterprise Fraud Detection System
            </p>
          </div>
        </div>

        {/* Right side */}
        <div className="flex items-center space-x-4">
          {/* System status indicator */}
          <div className="hidden sm:flex items-center space-x-2">
            <div className="h-2 w-2 bg-green-400 rounded-full animate-pulse"></div>
            <span className="text-sm text-gray-600 dark:text-gray-300">
              Online
            </span>
          </div>

          {/* Theme toggle */}
          <button
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            className="p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {theme === 'dark' ? (
              <SunIcon className="h-5 w-5" />
            ) : (
              <MoonIcon className="h-5 w-5" />
            )}
          </button>

          {/* Notifications */}
          <button className="p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 relative">
            <BellIcon className="h-5 w-5" />
            <span className="absolute top-1 right-1 h-2 w-2 bg-red-400 rounded-full"></span>
          </button>

          {/* User menu */}
          <div className="relative">
            <button className="flex items-center space-x-2 p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500">
              <UserCircleIcon className="h-6 w-6" />
              <span className="hidden sm:block text-sm font-medium text-gray-700 dark:text-gray-200">
                Admin
              </span>
            </button>
          </div>
        </div>
      </div>
    </header>
  )
}
