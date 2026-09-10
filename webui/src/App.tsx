import { Navigate, Route, Routes } from 'react-router-dom'
import { TopBar } from './components/TopBar'
import { BacktestPage } from './pages/BacktestPage'
import { DataPage } from './pages/DataPage'
import { OptimizePage } from './pages/OptimizePage'
import { ProductsPage } from './pages/ProductsPage'

export default function App() {
  return (
    <>
      <TopBar />
      <main style={{ flex: 1 }}>
        <Routes>
          <Route path="/" element={<Navigate to="/products" replace />} />
          <Route path="/products" element={<ProductsPage />} />
          <Route path="/data" element={<DataPage />} />
          <Route path="/backtest" element={<BacktestPage />} />
          <Route path="/optimize" element={<OptimizePage />} />
        </Routes>
      </main>
    </>
  )
}
