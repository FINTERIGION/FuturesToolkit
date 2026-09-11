import { describe, expect, it } from 'vitest'
import { clampDrawerHeight } from './useDrawerResize'

describe('clampDrawerHeight', () => {
  it('never goes below the minimum, even for a huge upward drag', () => {
    expect(clampDrawerHeight(10, 900)).toBe(160)
    expect(clampDrawerHeight(-500, 900)).toBe(160)
  })

  it('never exceeds 70% of the viewport height', () => {
    expect(clampDrawerHeight(10000, 900)).toBe(630)
  })

  it('passes through an in-range height unchanged', () => {
    expect(clampDrawerHeight(400, 900)).toBe(400)
  })

  it('falls back to the minimum on a very short viewport rather than an even smaller cap', () => {
    expect(clampDrawerHeight(50, 100)).toBe(160)
  })
})
