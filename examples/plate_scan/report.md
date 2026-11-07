# PLATE-12 (mill, AL)

## OP1: O2001_OP1.nc
- lines: 375
- estimated cycle time: 5.3 min (excluding loading)
- G-code check: no rapids into material, feeds and RPM within limits

## Review items

- [warn] machine: machine type 'mill' guessed with 75% confidence (ml); pass --machine to force
- [warn] stock: stock assumed 2 mm oversize per side and 2 mm on top; edit stock_* if blanks are pre-sized
- [warn] geometry: positions read from a scan: verify every hole X/Y against the drawing
- [warn] contour: outer contour is cut to Z-20.3: part must sit on parallels clear of the vise, or delete this section and finish the sides in a second setup
- [warn] holes: through holes break out under the part: use parallels with clearance below every hole

Total estimated machine time per part: 5.3 min
