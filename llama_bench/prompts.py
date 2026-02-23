"""Prompt templates and helpers for llama-bench reverse-engineering workload."""
from __future__ import annotations

import json
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# System prompt (~1200 tokens)
# ---------------------------------------------------------------------------

REVERSE_ENGINEERING_SYSTEM_PROMPT = """\
You are an expert reverse engineer and binary analyst with deep knowledge of x86-64 and ARM64 \
assembly, compiler theory, operating-system internals, and exploit development. You work daily \
with IDA Pro, Ghidra, Binary Ninja, and Radare2, and you are fluent in their scripting APIs \
(IDAPython, Ghidra Script, BNIL). You understand DWARF debug info, PDB symbols, PE/ELF/Mach-O \
file formats at the byte level, and can reconstruct high-level intent from stripped binaries.

Your expertise spans:

ASSEMBLY & ARCHITECTURE
- x86-64 calling conventions (System V AMD64 ABI, Windows x64 ABI, regcall, vectorcall)
- ARM64/AArch64 AAPCS64, Thumb-2, A64 instruction set, NEON/SVE SIMD
- Compiler-generated patterns: prologue/epilogue, stack canaries, frame pointers, red zone
- Instruction-level idioms: lea for arithmetic, xor reg,reg zero, cmov for branchless code
- CPU micro-architecture effects visible in disassembly (alignment NOPs, MFENCE, LFENCE)

BINARY ANALYSIS
- Static analysis: CFG reconstruction, function boundary detection, type recovery
- Dynamic analysis: ptrace-based tracing, hardware breakpoints, memory watchpoints
- Hybrid analysis: snapshot fuzzing, coverage-guided tracing with Intel PT / HWPT
- Anti-analysis bypass: self-modifying code, JIT stubs, obfuscation (OLLVM, VMProtect, Themida)
- Data-flow analysis: taint tracking, use-def chains, SSA form, value range analysis

DEBUGGERS & TOOLING
- lldb: Python scripting API, SBTarget/SBFrame/SBValue, custom formatters, step-scripting
- gdb/pwndbg/peda: gdb Python API, convenience variables, catchpoints, record/replay (rr)
- WinDbg: !analyze, dx LINQ queries, ttd (time-travel debugging), kernel debugging
- Frida: Interceptor, Stalker, MemoryAccessMonitor, CModule for performance-critical hooks
- DynamoRIO, PIN: instrumentation APIs for coverage and taint

HEAP & MEMORY EXPLOITATION
- glibc ptmalloc2: chunk metadata, fastbins, tcache, unsorted/small/large bins, consolidation
- jemalloc, tcmalloc, mimalloc: slab layout, thread caches, size-class tables
- Exploitation primitives: house-of-force, house-of-orange, tcache poisoning, safe-linking bypass
- Heap grooming, UAF, double-free, type confusion, out-of-bounds read/write

ROP & CODE REUSE
- ROP chain construction: gadget discovery (ropper, ROPgadget, pwntools ROP), pivot chains
- ret2libc, ret2plt, ret2csu, ret2dlresolve
- JOP/COP, COOP, shadow-stack bypass (CET/IBT)
- SROP (sigreturn-oriented programming), stack pivoting with xchg/pop rsp

VULNERABILITY CLASSES
- Stack buffer overflows, off-by-one, integer overflow → heap overflow
- Format string (read/write primitives, %n, DTOR overwrite)
- Race conditions (TOCTOU, double-fetch), use-after-free
- Kernel: NULL dereference, OOB in ioctl/copy_from_user, ret2usr, dirty cow variants

When analysing a binary, you:
1. Identify the function's purpose from control flow and data references.
2. Recover variable names and types from usage patterns.
3. Note any security-relevant operations (user-controlled data, crypto, IPC).
4. Flag potential vulnerabilities with CWE references and PoC sketch.
5. Suggest mitigations and hardening options.

Always reason step by step, cite specific instruction addresses when relevant, and prefer \
concrete technical detail over vague descriptions.
"""

# ---------------------------------------------------------------------------
# Shared prefix (~600 tokens of pseudocode)
# ---------------------------------------------------------------------------

SHARED_PREFIX_TEMPLATE = """\
=== Binary Analysis Context ===
Binary: firmware_update_handler  (x86-64 ELF, stripped, compiled with gcc-12 -O2 -fstack-protector)
Base address: 0x0000000000400000

--- Decompiled function at 0x401a80 (auto-named: sub_401a80) ---

__int64 __fastcall sub_401a80(
        int      fd,
        __int64  user_buf,
        size_t   user_len,
        __int64  flags)
{
  unsigned int  v4;           // [rsp+0x4]  [rbp-0x7c]  BYREF
  char          staging[128]; // [rsp+0x8]  [rbp-0x78]  BYREF  ← note: 128 bytes
  __int64       cookie;       // [rsp+0x88] [rbp+0x8]
  __int64       ret;

  cookie = __readfsqword(0x28);   // stack canary

  // --- prologue: validate fd ---
  if ( fd < 0 || fd > 1023 ) {
    errno = 9;        // EBADF
    return -1;
  }

  // --- copy user data into fixed-size staging buffer ---
  // user_len is size_t (unsigned), no upper-bound check here
  memcpy(staging, (void *)user_buf, user_len);   // ← potential OOB write

  v4 = 0;
  // --- parse TLV header ---
  // staging[0] = type (u8), staging[1..2] = length (u16 LE), staging[3..] = value
  unsigned char  tlv_type   = (unsigned char)staging[0];
  unsigned short tlv_length = *(unsigned short *)&staging[1];  // unaligned read

  if ( tlv_type == 0x42 ) {
    // firmware chunk
    if ( tlv_length > 0x78 ) {              // 120-byte sub-field limit
      errno = 22;     // EINVAL
      goto cleanup;
    }
    ret = process_fw_chunk(staging + 3, tlv_length, flags);
  } else if ( tlv_type == 0xFE ) {
    // diagnostic command
    ret = run_diag(fd, staging + 3, (size_t)tlv_length);
  } else {
    errno = 95;   // EOPNOTSUPP
    ret   = -1LL;
  }

cleanup:
  // --- epilogue: canary check ---
  if ( __readfsqword(0x28) != cookie )
    __stack_chk_fail();

  return ret;
}

--- Key xrefs ---
0x401a80  ←  dispatch_ioctl+0x94  (ioctl FWUPD_SUBMIT_CHUNK)
0x401a80  ←  netlink_recv_cb+0x1b0 (NETLINK_FWUPD, msg type 3)

--- Strings near 0x402c00 ---
0x402c10  "firmware_update_handler: bad fd %d\\n"
0x402c38  "firmware_update_handler: tlv overflow (len=%u)\\n"
0x402c68  "/dev/fwupd"

--- GOT entries of interest ---
memcpy@plt   → 0x401080
__stack_chk_fail@plt → 0x401090
process_fw_chunk@plt → 0x4010a0   (dynamic, loaded from plugin SO)
"""

# ---------------------------------------------------------------------------
# Follow-up questions
# ---------------------------------------------------------------------------

FOLLOWUP_QUESTIONS = [
    "What is the root cause vulnerability in this function and what CWE does it map to?",
    "Show me a concrete exploit proof-of-concept in Python (pwntools) that triggers the overflow.",
    "The binary has a stack canary. How would you bypass it given the netlink_recv_cb call path?",
    "Identify the TLV parsing logic and explain any integer-truncation risks in tlv_length handling.",
    "How would you use lldb to set a conditional breakpoint that fires only when tlv_type == 0x42 and tlv_length > 0x60?",
    "Write a Frida Interceptor hook for sub_401a80 that logs fd, user_len, tlv_type, and tlv_length without modifying behaviour.",
    "What mitigations (RELRO, NX, PIE, CET) would prevent exploitation of this function, and which are already present?",
    "Reconstruct the likely C source code for process_fw_chunk given the context clues in the decompilation.",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def estimate_token_count(text: str) -> int:
    """Rough token count estimate: len(text) // 4."""
    return len(text) // 4


def build_prompt_sequence(n_followups: int = 4, use_system: bool = True) -> list[dict]:
    """Build a list of message-sequence dicts for the benchmark.

    The first item is the initial turn (system + shared prefix + opening question).
    Subsequent items are follow-ups that re-include the shared prefix (simulating
    a real multi-turn session with prefix cache reuse).

    Each dict has:
      - ``"messages"``: list of ``{"role": str, "content": str}``
      - ``"is_followup"``: bool
      - ``"expected_prefix_len_tokens"``: int (rough estimate)
    """
    initial_content = (
        SHARED_PREFIX_TEMPLATE
        + "\n\nWhat does this function do? Provide a detailed analysis including "
        "the purpose, data flow, and any security implications."
    )

    messages_initial: list[dict] = []
    if use_system:
        messages_initial.append({"role": "system", "content": REVERSE_ENGINEERING_SYSTEM_PROMPT})
    messages_initial.append({"role": "user", "content": initial_content})

    prefix_tokens = estimate_token_count(SHARED_PREFIX_TEMPLATE)

    sequence: list[dict] = [
        {
            "messages": messages_initial,
            "is_followup": False,
            "expected_prefix_len_tokens": prefix_tokens,
        }
    ]

    # Reuse questions (cycle if n_followups > len(FOLLOWUP_QUESTIONS))
    for i in range(n_followups):
        question = FOLLOWUP_QUESTIONS[i % len(FOLLOWUP_QUESTIONS)]
        # Include shared prefix again to simulate cache reuse
        combined = SHARED_PREFIX_TEMPLATE + "\n\n" + question
        followup_messages = list(messages_initial) + [
            {"role": "user", "content": combined}
        ]
        sequence.append(
            {
                "messages": followup_messages,
                "is_followup": True,
                "expected_prefix_len_tokens": prefix_tokens,
            }
        )

    return sequence


def load_prompt_pack(path: str) -> list[dict]:
    """Load a JSON or YAML file of prompts.

    The file should contain a list of message-sequence dicts (same format as
    :func:`build_prompt_sequence` output).  Both ``.json`` and ``.yaml``
    extensions are supported.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read()

    if path.lower().endswith((".yaml", ".yml")):
        data = yaml.safe_load(raw)
    else:
        data = json.loads(raw)

    if not isinstance(data, list):
        raise ValueError(f"Prompt pack {path!r} must be a JSON/YAML list at the top level.")

    return data
