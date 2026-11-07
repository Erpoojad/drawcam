# BUSH-07 (lathe, MS)

## OP1: O1001_OP1.nc
- lines: 101
- estimated cycle time: 5.8 min (excluding loading)
- G-code check: no rapids into material, feeds and RPM within limits

## Review items

- [warn] scale: overall length and largest diameter imply different scales; check the profile
- [warn] profile: diameter read as 59.82, set to 60: no matching dimension text
- [warn] profile: step positions read from a scan and rounded to 0.5 mm: check every Z against the drawing

Total estimated machine time per part: 5.8 min
