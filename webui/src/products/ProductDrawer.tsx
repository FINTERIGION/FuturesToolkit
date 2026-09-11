import { useTranslation } from 'react-i18next'
import { Drawer } from '../components/Drawer'
import { ProductFormFields, useProductForm } from './ProductForm'

export function ProductDrawer({ code, onClose }: { code: string | null; onClose: () => void }) {
  const { t } = useTranslation()
  const isEdit = code !== null
  const f = useProductForm(code, onClose)

  return (
    <Drawer
      open
      onClose={onClose}
      title={isEdit ? `${t('products.editProduct')} — ${code}` : t('products.addProduct')}
      footer={
        <>
          {isEdit && !f.showPurge && (
            <button
              className="btn btn-danger"
              onClick={() => f.setShowPurge(true)}
              disabled={f.lockedByJob}
              title={f.lockedByJob ? t('products.lockedByJob') : undefined}
            >
              {t('common.delete')}
            </button>
          )}
          {isEdit && f.showPurge && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, flex: 1 }}>
              <label className="checkbox-row" style={{ margin: 0 }}>
                <input type="checkbox" checked={f.purgeData} onChange={(e) => f.setPurgeData(e.target.checked)} />
                {t('products.purgeData')}
              </label>
              <div className="spacer" />
              <button className="btn btn-sm" onClick={() => f.setShowPurge(false)}>
                {t('common.cancel')}
              </button>
              <button className="btn btn-sm btn-danger" onClick={f.remove} disabled={f.deleting || f.lockedByJob}>
                {t('common.confirm')}
              </button>
            </div>
          )}
          <div className="spacer" />
          <button className="btn" onClick={onClose}>
            {t('common.cancel')}
          </button>
          <button
            className="btn btn-primary"
            onClick={f.save}
            disabled={f.saving || f.lockedByJob}
            title={f.lockedByJob ? t('products.lockedByJob') : undefined}
          >
            {t('common.save')}
          </button>
        </>
      }
    >
      {f.lockedByJob && <div className="hint-banner warning">{t('products.lockedByJob')}</div>}
      <ProductFormFields f={f} />
    </Drawer>
  )
}
