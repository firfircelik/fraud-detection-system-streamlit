import './globals.css'
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import { Providers } from './providers'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Enterprise Fraud Detection System',
  description: 'Advanced real-time fraud detection with ML ensemble and enterprise analytics',
  keywords: ['fraud detection', 'machine learning', 'enterprise', 'security', 'analytics'],
  authors: [{ name: 'Fraud Detection Team' }],
  openGraph: {
    title: 'Enterprise Fraud Detection System',
    description: 'Advanced real-time fraud detection with ML ensemble and enterprise analytics',
    type: 'website',
  },
  robots: {
    index: false,
    follow: false,
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <Providers>
          {children}
        </Providers>
      </body>
    </html>
  )
}
