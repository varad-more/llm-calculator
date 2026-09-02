/** Raised when a HF config omits a field we refuse to guess (see guardrails in README). */
export class IncompleteConfigError extends Error {
  field: string
  modelType: string
  constructor(field: string, modelType: string) {
    super(
      `IncompleteConfigError: missing "${field}" (model_type=${modelType}). ` +
        `Add it to data/arch-defaults.json with a source_url, or pass it explicitly.`,
    )
    this.name = 'IncompleteConfigError'
    this.field = field
    this.modelType = modelType
  }
}

/** Raised when a requested GPU / quant scheme / engine / assumption is not in the shipped data. */
export class UnknownEntityError extends Error {
  constructor(kind: string, id: string, known: readonly string[]) {
    super(`Unknown ${kind} "${id}". Known: ${known.join(', ')}`)
    this.name = 'UnknownEntityError'
  }
}

/** Raised when a plan is arithmetically impossible (e.g. weights alone exceed VRAM). */
export class InfeasibleError extends Error {
  detail: Record<string, number>
  constructor(message: string, detail: Record<string, number> = {}) {
    super(message)
    this.name = 'InfeasibleError'
    this.detail = detail
  }
}
