import { useMutation, useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { errorMessage } from '../../api/client'
import { productsApi } from '../../api/endpoints'
import type { Product, ProductInput } from '../../api/types'
import { candlestickOption } from '../../charts/builders'
import { EChart } from '../../components/EChart'
import { Drawer } from '../../components/Drawer'
import { useIsDarkMode } from '../../theme/useIsDarkMode'

type CommissionMode = 'rate' | 'per_lot'

interface FormState {
  code: string
  exchange: string
  name: string
  name_zh: string
  start_year: string
  multiplier: string
  tick_size: string
  margin_rate: string
  commission_mode: CommissionMode
  commission_value: string
  main_months: string
  roll_lead_months: string
}

function emptyForm(defaultExchange: string): FormState {
  return {
    code: '',
    exchange: defaultExchange,
    name: '',
    name_zh: '',
    start_year: String(new Date().getFullYear()),
    multiplier: '10',
    tick_size: '1',
    margin_rate: '0.12',
    commission_mode: 'rate',
    commission_value: '0.0001',
    main_months: '1,5,9',
    roll_lead_months: '1',
  }
}

function formFromProduct(p: Product): FormState {
  const mode: CommissionMode = p.costs.commission_mode
  return {
    code: p.code,
    exchange: p.exchange,
    name: p.name,
    name_zh: p.name_zh,
    start_year: String(p.start_year),
    multiplier: String(p.multiplier),
    tick_size: String(p.tick_size),
    margin_rate: String(p.costs.margin_rate),
    commission_mode: mode,
    commission_value: String(mode === 'rate' ? p.costs.commission_rate : p.costs.commission_per_lot),
    main_months: p.roll.main_months.join(','),
    roll_lead_months: String(p.roll.lead_months),
  }
}

function toApiInput(form: FormState): ProductInput {
  const months = form.main_months
    .split(',')
    .map((s) => parseInt(s.trim(), 10))
    .filter((n) => !Number.isNaN(n))
  return {
    exchange: form.exchange,
    name: form.name,
    name_zh: form.name_zh,
    start_year: parseInt(form.start_year, 10),
    multiplier: parseFloat(form.multiplier),
    tick_size: parseFloat(form.tick_size),
    margin_rate: parseFloat(form.margin_rate),
    commission_rate: form.commission_mode === 'rate' ? parseFloat(form.commission_value) : null,
    commission_per_lot: form.commission_mode === 'per_lot' ? parseFloat(form.commission_value) : null,
    main_months: months.length ? months : null,
    roll_lead_months: form.roll_lead_months ? parseInt(form.roll_lead_months, 10) : null,
  }
}

export function ProductDrawer({ code, onClose }: { code: string | null; onClose: () => void }) {
  const { t } = useTranslation()
  const isEdit = code !== null
  const dark = useIsDarkMode()

  const { data: exchangeMeta } = useQuery({ queryKey: ['meta', 'exchanges'], queryFn: productsApi.exchanges })
  const { data: product } = useQuery({
    queryKey: ['product', code],
    queryFn: () => productsApi.get(code as string),
    enabled: isEdit,
  })

  const [form, setForm] = useState<FormState>(() => emptyForm(''))
  const [showPurge, setShowPurge] = useState(false)
  const [purgeData, setPurgeData] = useState(false)
  const [formError, setFormError] = useState<string | null>(null)

  useEffect(() => {
    if (product) setForm(formFromProduct(product))
    else if (!isEdit && exchangeMeta) setForm(emptyForm(exchangeMeta.exchanges[0] ?? 'CZCE'))
  }, [product, isEdit, exchangeMeta])

  const set = <K extends keyof FormState>(key: K, value: FormState[K]) => setForm((f) => ({ ...f, [key]: value }))

  const saveMutation = useMutation({
    mutationFn: () => {
      const input = toApiInput(form)
      return isEdit ? productsApi.update(code as string, input) : productsApi.create(form.code.toUpperCase(), input)
    },
    onSuccess: () => onClose(),
    onError: (err) => setFormError(errorMessage(err)),
  })

  const deleteMutation = useMutation({
    mutationFn: () => productsApi.remove(code as string, purgeData),
    onSuccess: () => onClose(),
    onError: (err) => setFormError(errorMessage(err)),
  })

  const { data: barsData } = useQuery({
    queryKey: ['product-bars', code],
    queryFn: () => productsApi.bars(code as string),
    enabled: isEdit,
  })
  const { data: rollData } = useQuery({
    queryKey: ['product-roll', code],
    queryFn: () => productsApi.roll(code as string),
    enabled: isEdit,
  })

  return (
    <Drawer
      open
      onClose={onClose}
      title={isEdit ? `${t('products.editProduct')} — ${code}` : t('products.addProduct')}
      footer={
        <>
          {isEdit && !showPurge && (
            <button className="btn btn-danger" onClick={() => setShowPurge(true)}>
              {t('common.delete')}
            </button>
          )}
          {isEdit && showPurge && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, flex: 1 }}>
              <label className="checkbox-row" style={{ margin: 0 }}>
                <input type="checkbox" checked={purgeData} onChange={(e) => setPurgeData(e.target.checked)} />
                {t('products.purgeData')}
              </label>
              <div className="spacer" />
              <button className="btn btn-sm" onClick={() => setShowPurge(false)}>
                {t('common.cancel')}
              </button>
              <button className="btn btn-sm btn-danger" onClick={() => deleteMutation.mutate()} disabled={deleteMutation.isPending}>
                {t('common.confirm')}
              </button>
            </div>
          )}
          <div className="spacer" />
          <button className="btn" onClick={onClose}>
            {t('common.cancel')}
          </button>
          <button className="btn btn-primary" onClick={() => saveMutation.mutate()} disabled={saveMutation.isPending}>
            {t('common.save')}
          </button>
        </>
      }
    >
      {formError && <div className="hint-banner warning">{formError}</div>}

      {!isEdit && (
        <div className="field">
          <label>{t('products.code')}</label>
          <input value={form.code} onChange={(e) => set('code', e.target.value.toUpperCase())} placeholder="e.g. AP" />
        </div>
      )}

      <div className="form-grid">
        <div className="field">
          <label>{t('products.exchange')}</label>
          <select value={form.exchange} onChange={(e) => set('exchange', e.target.value)}>
            {(exchangeMeta?.exchanges ?? [form.exchange]).map((ex) => (
              <option key={ex} value={ex}>
                {ex}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>{t('products.startYear')}</label>
          <input type="number" value={form.start_year} onChange={(e) => set('start_year', e.target.value)} />
        </div>
        <div className="field">
          <label>{t('products.name')}</label>
          <input value={form.name} onChange={(e) => set('name', e.target.value)} />
        </div>
        <div className="field">
          <label>{t('products.nameZh')}</label>
          <input value={form.name_zh} onChange={(e) => set('name_zh', e.target.value)} />
        </div>
        <div className="field">
          <label>{t('products.multiplier')}</label>
          <input type="number" value={form.multiplier} onChange={(e) => set('multiplier', e.target.value)} />
        </div>
        <div className="field">
          <label>{t('products.tickSize')}</label>
          <input type="number" step="any" value={form.tick_size} onChange={(e) => set('tick_size', e.target.value)} />
        </div>
        <div className="field">
          <label>{t('products.marginRate')}</label>
          <input type="number" step="0.01" value={form.margin_rate} onChange={(e) => set('margin_rate', e.target.value)} />
        </div>
        <div className="field">
          <label>{t('products.commissionMode')}</label>
          <select value={form.commission_mode} onChange={(e) => set('commission_mode', e.target.value as CommissionMode)}>
            <option value="rate">{t('products.commissionRate')}</option>
            <option value="per_lot">{t('products.commissionPerLot')}</option>
          </select>
        </div>
        <div className="field">
          <label>&nbsp;</label>
          <input type="number" step="any" value={form.commission_value} onChange={(e) => set('commission_value', e.target.value)} />
        </div>
        <div className="field">
          <label>{t('products.mainMonths')}</label>
          <input value={form.main_months} onChange={(e) => set('main_months', e.target.value)} placeholder="1,5,9" />
        </div>
        <div className="field">
          <label>{t('products.rollLeadMonths')}</label>
          <input type="number" value={form.roll_lead_months} onChange={(e) => set('roll_lead_months', e.target.value)} />
        </div>
      </div>

      {isEdit && (
        <div style={{ marginTop: 20 }}>
          <h3 style={{ marginBottom: 10 }}>{t('products.priceChart')}</h3>
          {barsData ? (
            <EChart
              option={candlestickOption(dark, barsData.bars, { roll: rollData?.roll })}
              height={360}
            />
          ) : (
            <div className="empty-state">{t('common.loading')}</div>
          )}
        </div>
      )}
    </Drawer>
  )
}
