import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App.tsx'
import './components/ui.css'
import './i18n'
import './index.css'
import { JobsProvider } from './shell/JobsProvider.tsx'
import { migrateStickyState } from './shell/migrateStickyState.ts'
import './shell/workspace.css'
import { WorkspaceProvider } from './shell/WorkspaceContext.tsx'

// Split the old shared `ft.symbols` universe key into `ft.chartSymbol` +
// `ft.extraSymbols` before anything reads either -- see the function's own
// doc comment. A no-op once it has run once.
migrateStickyState()

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, refetchOnWindowFocus: false },
  },
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <WorkspaceProvider>
          <JobsProvider>
            <App />
          </JobsProvider>
        </WorkspaceProvider>
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
)
