import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { factorsApi } from '../../api/endpoints'
import type { FactorReportSummary } from '../../api/types'
import { Drawer } from '../../components/Drawer'
import type { Column } from '../../components/Table'
import { Table } from '../../components/Table'
import { FactorReportView } from './FactorReportView'

export function FactorReportsList() {
  const { t } = useTranslation()
  const { data: reports } = useQuery({ queryKey: ['factor-reports'], queryFn: factorsApi.reports })

  const [viewName, setViewName] = useState<string | null>(null)
  const { data: viewReport, isLoading: viewLoading } = useQuery({
    queryKey: ['factor-report', viewName],
    queryFn: () => factorsApi.report(viewName as string),
    enabled: viewName !== null,
  })

  const columns: Column<FactorReportSummary>[] = [
    { key: 'factor', header: t('factors.factor'), render: (r) => r.factor ?? '—' },
    { key: 'symbols', header: t('common.symbols'), render: (r) => (r.symbols ?? []).join(', ') },
    { key: 'start', header: t('common.start'), render: (r) => r.start ?? '—' },
    { key: 'end', header: t('common.end'), render: (r) => r.end ?? '—' },
    { key: 'return_source', header: t('factors.returnSource'), render: (r) => r.return_source ?? '—' },
    {
      key: 'actions',
      header: t('common.actions'),
      render: (r) => (
        <button className="btn btn-sm" onClick={() => setViewName(r.name)}>
          {t('common.view')}
        </button>
      ),
    },
  ]

  return (
    <>
      <Table columns={columns} rows={reports ?? []} rowKey={(r) => r.name} />
      <Drawer open={viewName !== null} onClose={() => setViewName(null)} title={viewName ?? ''}>
        {viewLoading && <p>{t('common.loading')}</p>}
        {viewReport && <FactorReportView report={viewReport} />}
      </Drawer>
    </>
  )
}
