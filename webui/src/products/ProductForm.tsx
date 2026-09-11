import { useMutation, useQuery } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { errorMessage } from '../api/client'
import { productsApi } from '../api/endpoints'
import type { Product, ProductInput } from '../api/types'
import { candlestickOption } from '../charts/builders'
import { EChart } from '../components/EChart'
import { useJobs } from '../shell/JobsProvider'
import { useIsDarkMode } from '../theme/useIsDarkMode'

/**
 * The product create/edit form, split out of `ProductDrawer` so a future
 * non-modal presentation (the workspace's product sidebar) can render the
 * same fields without the modal `Drawer` chrome around them.
 *
 * `useProductForm` owns all state and mutations, including the footer
 * actions (save/delete), because the footer buttons live in the caller's
 * chrome (`Drawer`'s pinned footer slot today) while the fields render
 * through `ProductFormFields` -- splitting them this way keeps the footer
 * pinned to the bottom of a scrolling drawer body instead of scrolling away
 * with the fields.
 */

type CommissionMode = 'rate' | 'per_lot'

/** `seededFor` sentinel for the create form, which has no product code.
 *
 * Any string that is not a valid product code will do -- the registry holds
 * codes to `[A-Z0-9]+` (`datafeed/products.py`), so the angle brackets alone
 * make a collision impossible. It was a literal NUL character, which worked
 * as a sentinel and turned this file binary: git, grep and every diff view
 * stopped treating it as source, so nothing here matched a search and the
 * file's changes showed up as "Binary files differ".
 */
const NEW_PRODUCT = '<new>'

// Rates are stored as fractions but entered in friendlier units: commission in ‱
// (per ten-thousand) or yuan per lot; margin rates in %.
const RATE_SCALE = 10000
const PERCENT_SCALE = 100
const MONTHS = Array.from({ length: 12 }, (_, i) => i + 1)

/** Scale a stored fraction into its display unit without inventing or losing
 * precision.
 *
 * Both of these used to round to a fixed shape -- margin through
 * `Math.round`, commission through `toFixed(1)` -- which made the form lossy
 * in the one direction that matters: open a product, save it unchanged, and a
 * 12.5% margin came back 13%. The stored values happen to be whole percents
 * today, so nothing had gone wrong yet, but the panel is where they get
 * edited and the margin rate is what sizes every position.
 *
 * The rounding is still needed, just at a scale that only removes binary
 * floating-point noise: 0.11 * 100 is 11.000000000000002, and displaying that
 * in a number input is its own kind of wrong.
 */
function scaledToDisplay(value: number, scale: number): string {
  return String(Number((value * scale).toFixed(6)))
}

function marginToDisplay(margin: number): string {
  return scaledToDisplay(margin, PERCENT_SCALE)
}

function commissionToDisplay(mode: CommissionMode, p: Product): string {
  return mode === 'rate'
    ? scaledToDisplay(p.costs.commission_rate ?? 0, RATE_SCALE)
    : scaledToDisplay(p.costs.commission_per_lot ?? 0, 1)
}

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
  main_months: number[]
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
    margin_rate: '12',
    commission_mode: 'rate',
    commission_value: '1.0',
    main_months: [1, 5, 9],
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
    margin_rate: marginToDisplay(p.costs.margin_rate),
    commission_mode: mode,
    commission_value: commissionToDisplay(mode, p),
    main_months: [...p.roll.main_months].sort((a, b) => a - b),
    roll_lead_months: String(p.roll.lead_months),
  }
}

function toApiInput(form: FormState): ProductInput {
  return {
    exchange: form.exchange,
    name: form.name,
    name_zh: form.name_zh,
    start_year: parseInt(form.start_year, 10),
    multiplier: parseFloat(form.multiplier),
    tick_size: parseFloat(form.tick_size),
    margin_rate: parseFloat(form.margin_rate) / PERCENT_SCALE,
    commission_rate: form.commission_mode === 'rate' ? parseFloat(form.commission_value) / RATE_SCALE : null,
    commission_per_lot: form.commission_mode === 'per_lot' ? parseFloat(form.commission_value) : null,
    main_months: form.main_months.length ? form.main_months : null,
    roll_lead_months: form.roll_lead_months ? parseInt(form.roll_lead_months, 10) : null,
  }
}

export interface ProductFormHandle {
  isEdit: boolean
  form: FormState
  set: <K extends keyof FormState>(key: K, value: FormState[K]) => void
  toggleMonth: (m: number) => void
  exchanges: string[]
  formError: string | null
  showPurge: boolean
  setShowPurge: (v: boolean) => void
  purgeData: boolean
  setPurgeData: (v: boolean) => void
  save: () => void
  saving: boolean
  remove: () => void
  deleting: boolean
  barsReady: boolean
  chartOption: ReturnType<typeof candlestickOption> | null
  /** A backtest or data update is running. The backend refuses a registry
   * write while any job is active -- it reads costs and roll rules live,
   * per fill, so an edit landing mid-run would silently corrupt that run's
   * numbers (see web/routers/products.py). Save/Delete are disabled on this
   * rather than left to fail into a 409 after the user has already filled
   * the form in, now that the editor sits one click from the Run button in
   * the sidebar instead of on a separate page. The 409 path stays as the
   * backstop for a job started from another tab or the CLI, which this flag
   * cannot see. */
  lockedByJob: boolean
}

export function useProductForm(code: string | null, onClose: () => void): ProductFormHandle {
  const isEdit = code !== null
  const dark = useIsDarkMode()
  const jobs = useJobs()
  const lockedByJob = jobs.backtest.isActive || jobs.data.isActive

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

  // Seed the form from the loaded product once, not on every change to the
  // query's data. `coverage` rides along on the product payload and moves
  // whenever a download finishes, and re-seeding on that threw away whatever
  // the user had typed but not yet saved.
  const seededFor = useRef<string | null>(null)
  useEffect(() => {
    if (isEdit) {
      if (!product || seededFor.current === product.code) return
      seededFor.current = product.code
      setForm(formFromProduct(product))
    } else {
      if (!exchangeMeta || seededFor.current === NEW_PRODUCT) return
      seededFor.current = NEW_PRODUCT
      setForm(emptyForm(exchangeMeta.exchanges[0] ?? 'CZCE'))
    }
  }, [product, isEdit, exchangeMeta])

  const set = <K extends keyof FormState>(key: K, value: FormState[K]) => setForm((f) => ({ ...f, [key]: value }))

  const toggleMonth = (m: number) =>
    setForm((f) => ({
      ...f,
      main_months: f.main_months.includes(m)
        ? f.main_months.filter((x) => x !== m)
        : [...f.main_months, m].sort((a, b) => a - b),
    }))

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

  return {
    isEdit,
    form,
    set,
    toggleMonth,
    exchanges: exchangeMeta?.exchanges ?? [form.exchange],
    formError,
    showPurge,
    setShowPurge,
    purgeData,
    setPurgeData,
    save: () => saveMutation.mutate(),
    saving: saveMutation.isPending,
    remove: () => deleteMutation.mutate(),
    deleting: deleteMutation.isPending,
    barsReady: barsData !== undefined,
    chartOption: barsData ? candlestickOption(dark, barsData.bars, { roll: rollData?.roll }) : null,
    lockedByJob,
  }
}

export { MONTHS }

export function ProductFormFields({ f }: { f: ProductFormHandle }) {
  const { t } = useTranslation()

  return (
    <>
      {f.formError && <div className="hint-banner warning">{f.formError}</div>}

      {!f.isEdit && (
        <div className="field">
          <label>{t('products.code')}</label>
          <input value={f.form.code} onChange={(e) => f.set('code', e.target.value.toUpperCase())} placeholder="e.g. AP" />
        </div>
      )}

      <div className="form-grid">
        <div className="field">
          <label>{t('products.exchange')}</label>
          <select value={f.form.exchange} onChange={(e) => f.set('exchange', e.target.value)}>
            {f.exchanges.map((ex) => (
              <option key={ex} value={ex}>
                {ex}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>{t('products.startYear')}</label>
          <input type="number" value={f.form.start_year} onChange={(e) => f.set('start_year', e.target.value)} />
        </div>
        <div className="field">
          <label>{t('products.name')}</label>
          <input value={f.form.name} onChange={(e) => f.set('name', e.target.value)} />
        </div>
        <div className="field">
          <label>{t('products.nameZh')}</label>
          <input value={f.form.name_zh} onChange={(e) => f.set('name_zh', e.target.value)} />
        </div>
        <div className="field">
          <label>{t('products.multiplier')}</label>
          <input type="number" value={f.form.multiplier} onChange={(e) => f.set('multiplier', e.target.value)} />
        </div>
        <div className="field">
          <label>{t('products.tickSize')}</label>
          <input type="number" step="any" value={f.form.tick_size} onChange={(e) => f.set('tick_size', e.target.value)} />
        </div>
        <div className="field">
          <label>{t('products.marginRate')}</label>
          <div className="input-group">
            <input
              type="number"
              step="any"
              min="0"
              max="100"
              value={f.form.margin_rate}
              onChange={(e) => f.set('margin_rate', e.target.value)}
            />
            <span className="input-unit">%</span>
          </div>
        </div>
        <div className="field">
          <label>{t('products.commissionMode')}</label>
          <div className="input-group">
            <input type="number" step="0.1" value={f.form.commission_value} onChange={(e) => f.set('commission_value', e.target.value)} />
            <select value={f.form.commission_mode} onChange={(e) => f.set('commission_mode', e.target.value as CommissionMode)}>
              <option value="rate">{t('products.commissionModeRate')}</option>
              <option value="per_lot">{t('products.commissionModePerLot')}</option>
            </select>
          </div>
        </div>
        <div className="field field-wide">
          <label>{t('products.mainMonths')}</label>
          <div className="month-picker">
            {MONTHS.map((m) => (
              <label key={m} className={`month-chip${f.form.main_months.includes(m) ? ' active' : ''}`}>
                <input type="checkbox" checked={f.form.main_months.includes(m)} onChange={() => f.toggleMonth(m)} />
                <span>{m}</span>
              </label>
            ))}
          </div>
        </div>
        <div className="field">
          <label>{t('products.rollLeadMonths')}</label>
          <input type="number" value={f.form.roll_lead_months} onChange={(e) => f.set('roll_lead_months', e.target.value)} />
        </div>
      </div>

      {f.isEdit && (
        <div style={{ marginTop: 20 }}>
          <h3 style={{ marginBottom: 10 }}>{t('products.priceChart')}</h3>
          {f.chartOption ? (
            <EChart option={f.chartOption} height={360} />
          ) : (
            <div className="empty-state">{t('common.loading')}</div>
          )}
        </div>
      )}
    </>
  )
}
