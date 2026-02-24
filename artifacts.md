# Binary Analysis Artifacts

## Artifact 1: `sub_401a80` — Firmware Update Handler (full decompilation)

```c
__int64 __fastcall sub_401a80(
        int      fd,
        __int64  user_buf,
        size_t   user_len,
        __int64  flags)
{
  unsigned int  v4;           // [rsp+0x4]  [rbp-0x7c]  BYREF
  char          staging[128]; // [rsp+0x8]  [rbp-0x78]  BYREF  ← 128-byte fixed buffer
  __int64       cookie;       // [rsp+0x88] [rbp+0x8]
  __int64       ret;

  cookie = __readfsqword(0x28);   // stack canary read

  // --- prologue: fd validation ---
  if ( fd < 0 || fd > 1023 ) {
    errno = 9;        // EBADF
    return -1LL;
  }

  // --- copy user data into fixed-size staging buffer ---
  // BUG: user_len is size_t (unsigned), no upper-bound check before memcpy
  memcpy(staging, (void *)user_buf, user_len);   // ← stack buffer overflow

  v4 = 0;

  // --- TLV header parse ---
  unsigned char  tlv_type   = (unsigned char)staging[0];
  unsigned short tlv_length = *(unsigned short *)&staging[1];  // unaligned u16 LE

  if ( tlv_type == 0x42 ) {
    // firmware chunk
    if ( tlv_length > 0x78 ) {              // 120-byte limit on sub-field
      errno = 22;     // EINVAL
      goto cleanup;
    }
    ret = process_fw_chunk(staging + 3, tlv_length, flags);
  }
  else if ( tlv_type == 0xFE ) {
    // diagnostic command — reachable via NETLINK_FWUPD without privilege check
    ret = run_diag(fd, staging + 3, (size_t)tlv_length);
  }
  else {
    errno = 95;   // EOPNOTSUPP
    ret   = -1LL;
  }

cleanup:
  // epilogue: canary check
  if ( __readfsqword(0x28) != cookie )
    __stack_chk_fail();

  return ret;
}
```

**Key xrefs**:
- `0x401a80` ← `dispatch_ioctl+0x94` (`ioctl FWUPD_SUBMIT_CHUNK`, requires `CAP_SYS_ADMIN`)
- `0x401a80` ← `netlink_recv_cb+0x1b0` (`NETLINK_FWUPD`, msg type 3, no privilege check)

---

## Artifact 2: `run_diag` — Diagnostic Command Dispatcher (partial)

```c
__int64 __fastcall run_diag(int fd, char *data, size_t len)
{
  char   cmd_buf[64];  // [rsp+0x0] [rbp-0x48]  ← another fixed buffer
  size_t copy_len;

  // Truncate to 63 bytes — but only if len > 63; no check for len == 0
  copy_len = (len > 63) ? 63 : len;
  memcpy(cmd_buf, data, copy_len);
  cmd_buf[copy_len] = '\0';

  // Dispatch on first byte
  switch (cmd_buf[0]) {
    case 0x01:  return diag_get_version(fd);
    case 0x02:  return diag_dump_regs(fd, cmd_buf + 1, copy_len - 1);
    case 0x03:  return diag_exec_selftest(fd);
    default:
      errno = 22;
      return -1LL;
  }
}
```

**Note**: `copy_len - 1` when `copy_len == 0` wraps to `SIZE_MAX` (integer underflow).

---

## Artifact 3: Assembly excerpt — `sub_401a80` prologue (annotated)

```asm
; sub_401a80 prologue (0x401a80–0x401ab0)
0x401a80:  push   rbp
0x401a81:  mov    rbp, rsp
0x401a84:  sub    rsp, 0x90          ; allocate stack frame (144 bytes)
0x401a8b:  mov    [rbp-0x74], edi    ; fd   (int)
0x401a8e:  mov    [rbp-0x80], rsi    ; user_buf
0x401a95:  mov    [rbp-0x88], rdx    ; user_len (size_t)
0x401a9c:  mov    [rbp-0x90], rcx    ; flags
0x401aa3:  mov    rax, QWORD PTR fs:0x28
0x401aac:  mov    [rbp+0x8], rax     ; store canary
0x401ab0:  xor    eax, eax           ; clear eax

; fd bounds check
0x401ab2:  mov    eax, [rbp-0x74]
0x401ab5:  test   eax, eax
0x401ab7:  js     .bad_fd            ; fd < 0
0x401ab9:  cmp    eax, 0x3ff
0x401abe:  jg     .bad_fd            ; fd > 1023

; call memcpy(staging, user_buf, user_len)  ← NO bounds check on user_len
0x401ac0:  mov    rcx, [rbp-0x88]   ; user_len
0x401ac7:  mov    rdx, [rbp-0x80]   ; user_buf
0x401ace:  lea    rax, [rbp-0x78]   ; &staging[0]
0x401ad2:  mov    rsi, rdx          ; src
0x401ad5:  mov    rdi, rax          ; dst
0x401ad8:  call   memcpy@plt        ; ← OVERFLOW possible here
```

---

## Artifact 4: GOT / PLT table (relevant entries)

| PLT address | Symbol | Current GOT value |
|-------------|--------|-------------------|
| `0x401080` | `memcpy` | `0x7f...` (libc) |
| `0x401090` | `__stack_chk_fail` | `0x7f...` (libc) |
| `0x4010a0` | `process_fw_chunk` | `0x7f...` (plugin SO) |
| `0x4010b0` | `run_diag` | `0x401270` (same binary) |
| `0x4010c0` | `netlink_send` | `0x7f...` (libnetlink) |

Partial RELRO: entries above `0x404000` are writable at runtime.

---

## Artifact 5: String table excerpt (relevant)

```
0x402c10  "firmware_update_handler: bad fd %d\n"
0x402c38  "firmware_update_handler: tlv overflow (len=%u)\n"
0x402c68  "/dev/fwupd"
0x402c78  "diag: cmd %02x len %zu\n"
0x402c98  "diag: dump_regs called fd=%d buf=%p len=%zu\n"
0x402cd0  "selftest: PASS\n"
0x402ce0  "selftest: FAIL (code=%d)\n"
```

---

## Artifact 6: ROP gadget candidates (ropper output, binary only)

```
0x40157c: pop rdi; ret
0x401583: pop rsi; ret
0x40158a: pop rdx; ret
0x401591: pop rcx; ret
0x401598: pop rax; ret
0x4015a0: xchg rax, rsp; ret       ; ← stack pivot
0x4015a8: add rsp, 0x40; ret       ; ← stack lift
0x40157a: push rsp; pop rdi; ret   ; ← leak rsp into rdi
```

`libc` base must be leaked before ret2libc; no PIE means binary gadgets have stable
addresses.

---

## Artifact 7: Netlink message structure (`NETLINK_FWUPD`)

Recovered from kernel module source fragment in `/proc/kallsyms` leak:

```c
struct fwupd_nl_msg {
    __u8   msg_type;   // 0x01=status, 0x02=config, 0x03=data
    __u16  payload_len;
    __u8   payload[];  // variable-length, kernel copies into user-space buffer
};
```

When `msg_type == 0x03`, the kernel calls `sub_401a80(fd, payload, payload_len, 0)`.
`payload_len` is a userspace-controlled 16-bit value (range: 0–65535), but `staging` is
only 128 bytes → controlled overflow of up to **65407 bytes** onto the kernel stack.

---

## Artifact 8: `process_fw_chunk` — Dynamic plugin (partial symbols from `.so`)

```c
// process_fw_chunk imported from /opt/fwupd/lib/libfwupd_plugin.so
// Signature recovered via PLT/GOT analysis:
int process_fw_chunk(const char *chunk_data, uint16_t chunk_len, uint64_t flags);
```

The `.so` has PIE enabled and a separate stack canary seed.  Exploitation via this path
is harder; the `run_diag` path is preferred.
