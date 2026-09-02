import { DATA } from './generated.ts'

export interface ValidationRecord {
  model: string
  gpu: string
  engine: string
  engineVersion: string
  /** Signed relative error of our prediction vs. the engine's own startup log, per component. */
  errors: Record<string, number>
  caseFile: string
}

/**
 * The validation record for a (model, gpu, engine) triple, or null when we have never
 * diffed our numbers against a real startup log for it. Every consumer must label its
 * output accordingly — a prediction with no case behind it says so.
 * @see docs/MATH.md#validation
 */
export function validationFor(model: string, gpu: string, engine: string): ValidationRecord | null {
  const index: ValidationRecord[] = (DATA as any).validation ?? []
  return index.find((v) => v.model === model && v.gpu === gpu && v.engine === engine) ?? null
}

/** Every validation record we ship. @see docs/MATH.md#validation */
export function allValidations(): ValidationRecord[] {
  return ((DATA as any).validation ?? []) as ValidationRecord[]
}
