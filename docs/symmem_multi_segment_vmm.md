# Multi-segment VMM symmetric windows

CUDA, Hygon DU, MetaX, and PPU adaptors support symmetric windows backed by
multiple VMM physical allocations created by FlagCX. Set `FLAGCX_VMM_ENABLE=1`
and `FLAGCX_VMM_SEGMENT_BYTES=<bytes>` before `flagcxMemAlloc`. The segment
size must be a positive multiple of the provider's VMM allocation granularity.
FlagCX uses the existing single-allocation path when the setting is absent or
the rounded allocation fits in one segment. CUDA requires a toolkit with
`CUDART_VERSION >= 12010` for this path.

The common symmetric heap exchanges segment sizes and offsets, then transfers
one shareable FD per segment in a fixed rank and segment order. It closes each
exported or received FD before advancing to the next segment, so the number
of live FDs per process does not grow with the number of segments. Each adaptor
retains and exports its own physical handles, maps each peer segment, and owns
imported handles until every alias has been unmapped. Collective registration
publishes a window only after every rank succeeds. Public
one-sided network registration uses VA MR for a tracked multi-segment VMM
allocation; an explicit DMA-BUF-only mode returns `flagcxNotSupported`.

An externally created VMM mapping whose physical offset is unknown returns
`flagcxNotSupported` during symmetric-window registration. FlagCX does not
remap it from physical offset zero. An untracked allocation may use the
ordinary-allocation IPC fallback only when its provider can export a legacy
IPC handle from the allocation base. This also supports windows beginning at
an offset inside an ordinary allocation. The adaptors do not retain unknown
VMM handles to decide between these paths.

The opt-in MPI test `FLAGCX_CI_REQUIRE_LOW_FD_MULTI_SEGMENT=1` uses a
64-segment allocation and a temporarily reduced `RLIMIT_NOFILE` on Linux.
Run it with at least two ranks on one node and `FLAGCX_VMM_ENABLE=1`.
