#!/usr/bin/env node
// Node >= 22.6 runs the TypeScript source directly; no build step.
import { main } from '../src/cli.ts'
process.exitCode = await main(process.argv.slice(2))
