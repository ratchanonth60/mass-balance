# Preserved MATLAB quirks

The rig runs on the *live* `AutoMass_MPC.mlapp` class-method code today, so
this crate replicates its wire behavior byte-exact rather than "fixing" bugs
found along the way. Every entry below cites the MATLAB source; anything not
listed here is assumed to be intentional/correct in both languages.

## `functions/*.m` top-level files are NOT the live code path

`functions/ReadEncoder.m`, `functions/HomeButtonPushed.m`, and friends are
**dead code** — `MainConstantTs11.m` and the in-class LQI loop call
`app.ReadEncoder(...)`, `app.HomeButtonPushed(...)`, etc., which MATLAB
resolves to methods *inside* the `AutoMass_MPC.mlapp` classdef, not the
same-named top-level functions. The class methods are a separate (and in
places materially different) implementation:

- `functions/ReadEncoder.m` and `functions/HomeButtonPushed.m` send a literal
  `0x00` checksum byte instead of computing one. The live class methods
  (`app.ReadEncoder` → `app.MotorCommand`, `app.HomeButtonPushed` →
  `app.MotorCommand`) **do** compute the checksum correctly
  (`mod(sum(bytes),256)`). This crate implements the live (correct) behavior
  — `mks::build_simple` always computes the checksum.

## `mks::parse_encoder_reply` — buggy header validation, preserved

`app.MotorCRCCheck` (called only from `app.ReadEncoder`) rejects a reply only
when **all three** of `byte0!=0xFB`, `byte1!=addr`, `byte2!=0x30` are true
simultaneously (`&&` where `||` was surely intended):

```matlab
if (ReadDataMotor(1) ~= 251) && (ReadDataMotor(2) ~= MotorNum) && (ReadDataMotor(3) ~= 48)
```

A reply with any *one* field correct is accepted even if the other two are
garbage. Replicated as-is in `parse_encoder_reply`. `ReadError`/`ReadRPM` use
proper `||`-based strict validation (all three fields must match) and are
implemented strictly here too.

## `mks::parse_is_moving_reply` — no addr or checksum check

`app.isMotorMoving` only checks `numel(resp)>=4 && resp(1)==0xFB &&
resp(3)==0xF1 && resp(4)~=1` — it never checks the address byte or the
checksum byte at all. Replicated as-is.

## `mks::build_run_abs`/`build_run_rel` — unconditional negation, multiply-then-clamp

- `d = -d` runs unconditionally in the live methods; the `if addr ~= 3` /
  `if addr == 3` guards mentioned in the comments are commented out in the
  code itself.
- Speed is scaled *before* clamping: `spd = spd*16; spd = clamp(0,spd,3000)`,
  giving a usable caller range of ~0-187.5 RPM before saturation. (The
  `functino test/mksRunRelAxis.m` copy clamps first, then multiplies — that
  copy is not on the live call path, so it's not what's replicated here.)

## `mks::parse_encoder_reply` — asymmetric total-angle branch

```matlab
if (degree < 360) && (degree > 0)
    rotation = rotation*360;
    TotalAngle = rotation + degree;
else
    TotalAngle = rotation;   % NOT rotation*360 here
end
```

The `else` branch does not multiply `rotation` by 360, unlike the `if`
branch. This asymmetry is preserved exactly.

## `imu::parse_combined_reply` — no CRC16 verification on receive

`readHWT9053.m` (the function actually called by both control loops for IMU
data) sends a pre-baked TX command with a hardcoded CRC16, and on receive only
checks `rx(1) == 0x50` — it never computes or checks a CRC16 on the reply.
Replicated as-is; `imu::parse_angle_only_reply` (unused by the live control
loop, ported for completeness/future UI use) is the one reader in the
original codebase (`ReadAngles.m`) that *does* verify CRC16 on receive, and
this crate does the same for that path.

## `imu::CMD_READ_YAW` / `CMD_READ_YAW_VELO` — stale CRC16 in the source itself

`app.CommandReadYaw` and `app.CommandReadYawVelo` (HWT101CL slave `0x60`,
registers `0x3D`/`0x37`) have trailing CRC16 bytes that do **not** match a
CRC16/Modbus computation over their own preceding 6 bytes — every other fixed
command in the class (including `app.CollectAll`, also slave `0x60`, verified
against the standalone `HWT101Test.m`) does check out. This looks like a
copy-paste bug in the original MATLAB (register offset copied from the slave
`0x50` `CommandReadAngle` constant without recomputing CRC for slave `0x60`).
Neither constant is used by the live MPC or LQI control loop — only by
individual-read UI methods — so sending them to real HWT101 hardware would
likely just be silently ignored by the device (Modbus slaves drop
invalid-CRC frames). Kept byte-exact per policy; not "corrected."

## `imu::read_int32_triplet_swapped` — non-standard word order

32-bit angle/yaw fields are two 16-bit Modbus registers where the **second**
register holds the high word, and each register's own two bytes are
big-endian: `value = (hi(reg1)<<8 | lo(reg1)) | (hi(reg2)<<24 | lo(reg2)<<16)`.
Matches `charToInt`/`HexToAng` in the live code exactly.
