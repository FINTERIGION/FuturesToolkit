import { useTranslation } from 'react-i18next'
import { NavLink } from 'react-router-dom'

const LANGS: { code: string; label: string }[] = [
  { code: 'en', label: 'EN' },
  { code: 'zh', label: '中文' },
]

export function TopBar() {
  const { t, i18n } = useTranslation()

  return (
    <header className="topbar">
      <div className="topbar-inner">
        <span className="brand">FuturesToolkit</span>
        <nav className="topnav">
          <NavLink to="/products">{t('nav.products')}</NavLink>
          <NavLink to="/data">{t('nav.data')}</NavLink>
          <NavLink to="/backtest">{t('nav.backtest')}</NavLink>
          <NavLink to="/optimize">{t('nav.optimize')}</NavLink>
          <NavLink to="/signals">{t('nav.signals')}</NavLink>
          <NavLink to="/runs">{t('nav.runs')}</NavLink>
        </nav>
        <div className="spacer" />
        <div className="lang-switch">
          {LANGS.map((l) => (
            <button
              key={l.code}
              className={`lang-btn ${i18n.resolvedLanguage === l.code ? 'active' : ''}`}
              onClick={() => void i18n.changeLanguage(l.code)}
            >
              {l.label}
            </button>
          ))}
        </div>
      </div>
    </header>
  )
}
