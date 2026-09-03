# Lessons

Corrections worth not repeating. One entry per mistake, with the rule it produced.

## Do not ship an engineering claim the model can be asked to check

Building the machine-comparison feature I wrote, in a docstring and in `docs/MATH.md`, that an
8-GPU machine should be run as 8 replicas rather than one TP=8 server — "throughput scales with
replicas, latency does not" — and only then ran the sweep. The allocator said the opposite, and it
was right: a decode step streams the weights resident on the device once and serves the whole
batch from that read, so TP divides the read while replicas each re-read the whole model. TP=8 was
4.3x eight TP=1 replicas at C=64. The crossover is real but sits past ~1k concurrent sequences.

**Rule:** this repo can evaluate almost any claim about serving in a few lines. Any sentence of the
form "X is faster than Y" goes through `size()` *before* it goes into a docstring, a README or a
commit message. Prose is the last thing written, not the first.

## An unused field is an unenforced field

`reservedVramFraction` shipped with the Apple entries and the Python port never read it. No golden
fixture used an Apple GPU, so the two ports silently disagreed for a week. It surfaced only when an
L4 and an L40S — both in the goldens — gained the same field.

**Rule:** a new field on a shared type needs a golden fixture that exercises it, in the same commit.
Cross-port parity is only as good as the fixtures' coverage of the fields, not of the functions.

## The vendor's capacity is not the tenant's capacity

GDDR6 datacenter cards (A10G, L4, L40S) ship with in-band ECC on, which costs 1/16 of the
framebuffer. We were handing out all 24 GiB of an L4 and calling configurations feasible that OOM
on a real `g6.xlarge`. AWS's own instance table says 22 GiB of 24, and 44 of 48 on the L40S.

**Rule:** when a datasheet number and a platform's own documentation disagree about the same part,
the platform is describing what you actually get. Prefer it, and say in the note why they differ.
