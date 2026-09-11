import { useCallback, useRef } from 'react'
import type { PointerEvent as ReactPointerEvent } from 'react'

const MIN_HEIGHT = 160
const MAX_RATIO = 0.7

/** Pure clamp, tested without synthesizing pointer events: never smaller
 * than a form actually needs, never so tall it swallows the chart above it. */
export function clampDrawerHeight(height: number, viewportHeight: number): number {
  const max = Math.max(MIN_HEIGHT, viewportHeight * MAX_RATIO)
  return Math.min(Math.max(height, MIN_HEIGHT), max)
}

/**
 * Drag-to-resize for the drawer's top edge. The live drag writes straight to
 * a CSS custom property via `document.documentElement.style`, not React
 * state -- state updates on every `pointermove` would re-render the whole
 * workspace tree at drag speed. Only the final height, on `pointerup`,
 * reaches `onCommit` (and from there, sticky storage).
 */
export function useDrawerResize(height: number, onCommit: (height: number) => void) {
  const draggingRef = useRef(false)

  const onPointerDown = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      const handle = e.currentTarget
      const startY = e.clientY
      const startHeight = height
      const root = document.documentElement
      draggingRef.current = true
      handle.setPointerCapture(e.pointerId)
      document.body.classList.add('drawer-resizing')

      const apply = (clientY: number) => {
        const next = clampDrawerHeight(startHeight + (startY - clientY), window.innerHeight)
        root.style.setProperty('--drawer-h', `${next}px`)
        return next
      }

      const move = (ev: PointerEvent) => {
        apply(ev.clientY)
      }
      const up = (ev: PointerEvent) => {
        draggingRef.current = false
        handle.releasePointerCapture(e.pointerId)
        document.body.classList.remove('drawer-resizing')
        window.removeEventListener('pointermove', move)
        window.removeEventListener('pointerup', up)
        onCommit(apply(ev.clientY))
      }
      window.addEventListener('pointermove', move)
      window.addEventListener('pointerup', up)
    },
    [height, onCommit],
  )

  return { onPointerDown }
}
