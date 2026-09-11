import { Navigate, Route, Routes } from 'react-router-dom'
import { ChartWorkspace } from './shell/ChartWorkspace'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<ChartWorkspace />} />
      {/* Every old routed page (Products, Data, Backtest, Optimize) lives
          inside the workspace now -- see ChartWorkspace / BottomDrawer /
          SymbolSidebar. A bookmarked or typed old path lands here rather
          than an empty <Routes>. */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
