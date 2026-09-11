import { useQuery } from '@tanstack/react-query'
import { strategiesApi } from '../api/endpoints'
import type { RunSummary } from '../api/types'
import { RunHistory } from '../components/RunHistory'
import { useWorkspace } from '../shell/WorkspaceContext'

/** The drawer's History tab. Opening a row overlays that run on the main
 * chart immediately (via `runId`, the same field the Backtest tab's result
 * tables read) -- there is no separate results view to navigate to. It also
 * prefills the Backtest tab's form and switches to it, so reopening a past
 * run shows the strategy, parameters, and universe that produced it -- the
 * main chart follows, re-charting to that run's first symbol -- rather than
 * whatever the form and chart happened to be holding. */
export function HistoryPanel() {
  const { runId, setRunId, prefillBacktest } = useWorkspace()
  // `RunSummary.strategy` is the class name the engine ran (see
  // web/routers/backtest.py), not the registry key the form's dropdown and
  // `BacktestPrefill.strategy` use -- this is what maps one to the other.
  const { data: strategies } = useQuery({ queryKey: ['strategies'], queryFn: strategiesApi.list })

  const openRun = (run: RunSummary) => {
    const strategyKey = strategies?.find((s) => s.class_name === run.strategy)?.key
    if (strategyKey && run.start && run.end && run.cash !== null && run.slippage !== null) {
      // `run.id` overlays this run in the same URL update as the prefill --
      // see `prefillBacktest`'s own comment for why a separate `setRunId`
      // call here would silently lose one or the other.
      prefillBacktest(
        {
          strategy: strategyKey,
          symbols: run.symbols,
          start: run.start,
          end: run.end,
          cash: run.cash,
          slippage: run.slippage,
          params: run.params,
        },
        run.id,
      )
    } else {
      setRunId(run.id)
    }
  }

  return (
    <RunHistory
      onOpen={openRun}
      onDeleted={(id) => {
        if (runId === id) setRunId(null)
      }}
      openRunId={runId}
    />
  )
}
