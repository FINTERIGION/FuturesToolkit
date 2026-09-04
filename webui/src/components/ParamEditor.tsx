import { useTranslation } from 'react-i18next'
import type { SpaceSpec } from '../api/types'

/** Per-param validity against the strategy's declared search space (see
 * research/space.py::resolve_space on the backend, mirrored here via
 * StrategyInfo.space). Only int/float (range) and categorical (choices)
 * params carry a declared range; anything else has nothing to check. */
export type ParamErrorKind = 'below_min' | 'above_max' | 'not_number' | 'invalid_choice'

export function paramErrors(
  defaults: Record<string, unknown>,
  space: Record<string, SpaceSpec>,
  values: Record<string, unknown>,
): Record<string, ParamErrorKind> {
  const errors: Record<string, ParamErrorKind> = {}
  for (const name of Object.keys(defaults)) {
    const spec = space[name]
    const current = values[name] ?? defaults[name]

    if (spec?.kind === 'int' || spec?.kind === 'float') {
      const n = Number(current)
      if (!Number.isFinite(n)) {
        errors[name] = 'not_number'
      } else if (spec.low !== undefined && n < spec.low) {
        errors[name] = 'below_min'
      } else if (spec.high !== undefined && n > spec.high) {
        errors[name] = 'above_max'
      }
    } else if (spec?.kind === 'categorical') {
      const choices = spec.choices ?? []
      if (choices.length > 0 && !choices.some((c) => String(c) === String(current))) {
        errors[name] = 'invalid_choice'
      }
    }
  }
  return errors
}

export function paramsAreValid(
  defaults: Record<string, unknown>,
  space: Record<string, SpaceSpec>,
  values: Record<string, unknown>,
): boolean {
  return Object.keys(paramErrors(defaults, space, values)).length === 0
}

/** One control per declared param, typed off its default value and the
 * strategy's declared/inferred search-space spec. A numeric param with an
 * Int/Float space gets a number input constrained to that range, a
 * Categorical gets a select; anything else (a param the space couldn't infer
 * a range for) falls back to a plain text input, cast back to the default's
 * type on change. Params outside their declared range are flagged inline. */
export function ParamEditor({
  defaults,
  space,
  fixedParams,
  values,
  onChange,
}: {
  defaults: Record<string, unknown>
  space: Record<string, SpaceSpec>
  fixedParams: string[]
  values: Record<string, unknown>
  onChange: (name: string, value: unknown) => void
}) {
  const { t } = useTranslation()
  const names = Object.keys(defaults)
  if (names.length === 0) return null

  const errors = paramErrors(defaults, space, values)

  const errorLabel = (name: string, spec: SpaceSpec | undefined) => {
    const kind = errors[name]
    if (!kind) return null
    if (kind === 'below_min') return t('strategy.paramBelowMin', { min: spec?.low })
    if (kind === 'above_max') return t('strategy.paramAboveMax', { max: spec?.high })
    if (kind === 'not_number') return t('strategy.paramNotNumber')
    return t('strategy.paramInvalidChoice')
  }

  return (
    <div className="form-grid">
      {names.map((name) => {
        const spec = space[name]
        const isFixed = fixedParams.includes(name)
        const current = values[name] ?? defaults[name]
        const error = errorLabel(name, spec)

        if (spec?.kind === 'categorical') {
          return (
            <div className="field" key={name}>
              <label>
                {name} {isFixed && <span className="field-hint">(fixed)</span>}
              </label>
              <select
                className={error ? 'invalid' : undefined}
                value={String(current)}
                onChange={(e) => onChange(name, coerce(e.target.value, defaults[name]))}
              >
                {(spec.choices ?? []).map((choice) => (
                  <option key={String(choice)} value={String(choice)}>
                    {String(choice)}
                  </option>
                ))}
              </select>
              {error && <span className="field-error">{error}</span>}
            </div>
          )
        }

        if (spec?.kind === 'int' || spec?.kind === 'float') {
          return (
            <div className="field" key={name}>
              <label>{name}</label>
              <input
                type="number"
                className={error ? 'invalid' : undefined}
                value={String(current)}
                step={spec.kind === 'int' ? spec.step ?? 1 : spec.step ?? 'any'}
                min={spec.low}
                max={spec.high}
                onChange={(e) => onChange(name, spec.kind === 'int' ? parseInt(e.target.value, 10) : parseFloat(e.target.value))}
              />
              <span className="field-hint">
                {error ?? `range ${spec.low}–${spec.high}`}
              </span>
            </div>
          )
        }

        if (typeof defaults[name] === 'boolean') {
          return (
            <div className="checkbox-row" key={name}>
              <input
                type="checkbox"
                id={`param-${name}`}
                checked={Boolean(current)}
                onChange={(e) => onChange(name, e.target.checked)}
              />
              <label htmlFor={`param-${name}`}>{name}</label>
            </div>
          )
        }

        return (
          <div className="field" key={name}>
            <label>
              {name} {isFixed && <span className="field-hint">(fixed)</span>}
            </label>
            <input
              type="text"
              value={String(current)}
              onChange={(e) => onChange(name, coerce(e.target.value, defaults[name]))}
            />
          </div>
        )
      })}
    </div>
  )
}

function coerce(raw: string, likeDefault: unknown): unknown {
  if (typeof likeDefault === 'number') {
    const n = Number(raw)
    return Number.isNaN(n) ? raw : n
  }
  if (typeof likeDefault === 'boolean') return raw === 'true'
  return raw
}
