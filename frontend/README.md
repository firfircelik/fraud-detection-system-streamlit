# Enterprise Fraud Detection Frontend

Modern Next.js 14 frontend for the Enterprise Fraud Detection System with TypeScript, Tailwind CSS, and enterprise-grade components.

## 🚀 Features

- **Modern Stack**: Next.js 14 + TypeScript + Tailwind CSS
- **Enterprise UI**: Professional dashboard with 6 specialized modules
- **Real-time Detection**: Live fraud detection interface
- **Responsive Design**: Mobile-first responsive layout
- **Dark Mode**: Built-in light/dark theme support
- **Performance**: Optimized for speed and scalability
- **Accessibility**: WCAG compliant components

## 📦 Tech Stack

### Core Technologies
- **Framework**: Next.js 14 with App Router
- **Language**: TypeScript 5.4+
- **Styling**: Tailwind CSS 3.4+
- **Icons**: Heroicons + Lucide React
- **Charts**: Recharts for data visualization
- **State**: React Query for server state
- **Forms**: React Hook Form + Zod validation
- **Notifications**: React Hot Toast

### Development Tools
- **Linting**: ESLint + TypeScript ESLint
- **Formatting**: Prettier with Tailwind plugin
- **Type Checking**: Strict TypeScript configuration

## 🏗️ Project Structure

```
frontend/
├── src/
│   ├── app/                    # Next.js App Router
│   │   ├── globals.css         # Global styles
│   │   ├── layout.tsx          # Root layout
│   │   ├── page.tsx            # Home page
│   │   ├── providers.tsx       # Context providers
│   │   └── theme-provider.tsx  # Theme provider
│   └── components/
│       └── dashboard/          # Dashboard components
│           ├── Dashboard.tsx   # Main dashboard
│           ├── Sidebar.tsx     # Navigation sidebar
│           ├── TopNav.tsx      # Top navigation
│           └── pages/          # Dashboard pages
│               ├── OverviewPage.tsx     # Dashboard overview
│               ├── DetectionPage.tsx    # Fraud detection
│               ├── AnalyticsPage.tsx    # Advanced analytics
│               ├── TransactionsPage.tsx # Transaction monitoring
│               ├── MLModelsPage.tsx     # ML model management
│               └── SettingsPage.tsx     # System settings
├── public/                     # Static assets
├── package.json               # Dependencies
├── tailwind.config.js         # Tailwind configuration
├── tsconfig.json             # TypeScript configuration
└── next.config.js            # Next.js configuration
```

## 🚀 Getting Started

### Prerequisites
- Node.js 18.0.0 or higher
- npm 8.0.0 or higher
- Backend API running on http://localhost:8000

### Installation

```bash
# Install dependencies
npm install

# Create environment file
cp .env.local.example .env.local

# Start development server
npm run dev
```

### Available Scripts

```bash
npm run dev          # Start development server (port 3000)
npm run build        # Build for production
npm run start        # Start production server
npm run lint         # Run ESLint
npm run type-check   # Run TypeScript type checking
npm run format       # Format code with Prettier
```

## 🎨 Dashboard Modules

### 1. 📊 Overview Dashboard
- Real-time system metrics
- Transaction volume charts
- Fraud detection statistics
- Risk distribution analytics
- Recent fraud alerts

### 2. 🛡️ Fraud Detection
- Single transaction analysis
- Batch CSV processing
- Real-time risk scoring
- Feature importance analysis
- Decision explanations

### 3. 📈 Advanced Analytics
- Time-series fraud patterns
- Geographic risk analysis
- Behavioral analytics
- ROI tracking
- Custom reporting

### 4. 💳 Transaction Monitoring
- Real-time transaction feeds
- Transaction history
- Risk categorization
- Alert management
- Export capabilities

### 5. 🤖 ML Model Management
- Model performance metrics
- Ensemble configuration
- Model versioning
- A/B testing
- Feature monitoring

### 6. ⚙️ System Settings
- User management
- System configuration
- Alert thresholds
- Integration settings
- Audit logs

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME="Enterprise Fraud Detection System"
NEXT_PUBLIC_APP_VERSION="1.0.0"

# Feature Flags
NEXT_PUBLIC_ENABLE_DARK_MODE=true
NEXT_PUBLIC_ENABLE_ANALYTICS=true
NEXT_PUBLIC_ENABLE_REAL_TIME=true
```

### Theme Configuration

The application supports both light and dark themes with automatic system detection. Theme preferences are persisted across sessions.

### API Integration

Frontend communicates with the FastAPI backend through:
- REST API endpoints
- Real-time WebSocket connections
- File upload handling
- Authentication tokens

## 📱 Responsive Design

The dashboard is fully responsive and optimized for:
- **Desktop**: Full featured dashboard experience
- **Tablet**: Adapted layout with collapsible sidebar
- **Mobile**: Mobile-first design with touch-friendly interfaces

## 🚀 Performance

### Optimizations
- **Code Splitting**: Automatic route-based code splitting
- **Image Optimization**: Next.js Image component with lazy loading
- **Bundle Analysis**: Webpack bundle analyzer integration
- **Caching**: Smart caching strategies for API responses

### Benchmarks
- **Lighthouse Score**: 95+ for Performance, Accessibility, Best Practices
- **First Contentful Paint**: <1.5s
- **Largest Contentful Paint**: <2.5s
- **Cumulative Layout Shift**: <0.1

## 🔒 Security

### Security Features
- **CSP Headers**: Content Security Policy headers
- **XSS Protection**: Built-in XSS protection
- **CSRF Protection**: Cross-site request forgery protection
- **Secure Headers**: Security headers configuration

### Authentication
- JWT token-based authentication
- Automatic token refresh
- Secure token storage
- Role-based access control

## 🧪 Testing

```bash
# Run unit tests
npm run test

# Run integration tests
npm run test:integration

# Run E2E tests
npm run test:e2e

# Generate coverage report
npm run test:coverage
```

## 📦 Deployment

### Production Build

```bash
# Build for production
npm run build

# Start production server
npm start
```

### Docker Deployment

```bash
# Build Docker image
docker build -t fraud-frontend .

# Run container
docker run -p 3000:3000 fraud-frontend
```

### Environment-specific Builds

```bash
# Development build
npm run build:dev

# Staging build
npm run build:staging

# Production build
npm run build:prod
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

### Development Guidelines
- Follow TypeScript strict mode
- Use Tailwind CSS for styling
- Implement responsive design
- Add proper error handling
- Include unit tests
- Follow accessibility guidelines

## 📄 License

MIT License - See LICENSE file for details.

---

🎨 **Built with Next.js 14 + TypeScript + Tailwind CSS** - Modern, fast, and enterprise-ready!
